/**
 * MELD Backend (Rebuilt) — stability-first
 * - OpenAI 2-pass: composer -> answer
 * - Optional xAI scout for Multi/Ultimate run modes
 * - Redis sessions (optional)
 * - Token auth: X-MELD-TOKEN (preferred) or Authorization: Bearer
 *
 * Node 18+ (CommonJS)
 */

require('dotenv').config();

const express = require('express');
const cors = require('cors');
const JSZip = require('jszip');
const OpenAI = require('openai');
const { createClient } = require('redis');

const app = express();

// -------------------- Config --------------------

const VERSION = '1.0.0-rebuild';
const PORT = process.env.PORT || 10000;

const ALLOWED_ORIGIN = (process.env.ALLOWED_ORIGIN || '').trim();
const MAX_BODY_SIZE = process.env.MAX_BODY_SIZE || '2mb';

const MAX_SESSION_TURNS = Number(process.env.MAX_SESSION_TURNS || 12); // turns per lane
const SESSION_TTL_SEC = Number(process.env.MELD_SESSION_TTL_SEC || 7 * 24 * 60 * 60);

const OPENAI_FAST_MODEL = process.env.MELD_OPENAI_FAST_MODEL || 'gpt-5-nano';
const OPENAI_DEEP_MODEL = process.env.MELD_OPENAI_DEEP_MODEL || 'gpt-5-nano';

const XAI_API_KEY = (process.env.XAI_API_KEY || '').trim();
const XAI_MODEL = process.env.MELD_XAI_MODEL || 'grok-4';
const XAI_BASE_URL = (process.env.MELD_XAI_BASE_URL || 'https://api.x.ai').replace(/\/+$/, '');

const BACKEND_TOKEN = (process.env.MELD_BACKEND_TOKEN || '').trim();

const REDIS_URL = (process.env.REDIS_URL || '').trim();

// -------------------- Middleware --------------------

app.use(express.json({ limit: MAX_BODY_SIZE }));

const corsOptions = ALLOWED_ORIGIN
  ? { origin: ALLOWED_ORIGIN, credentials: false }
  : { origin: '*', credentials: false };

app.use(cors(corsOptions));

// minimal logging
app.use((req, res, next) => {
  const uid = (req.headers['x-meld-user'] || '').toString();
  const intent = req.body && req.body.intent;
  const runMode = req.body && (req.body.runMode || req.body.mode);
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path} user=${uid} intent=${intent} runMode=${runMode}`);
  next();
});

// -------------------- OpenAI client --------------------

let openai = null;
if (process.env.OPENAI_API_KEY) {
  openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
}

// -------------------- Redis session store --------------------

let redis = null;
let redisReady = false;

async function initRedis() {
  if (!REDIS_URL) return;
  try {
    redis = createClient({ url: REDIS_URL });
    redis.on('error', (err) => {
      console.warn('Redis error:', err && err.message ? err.message : err);
      redisReady = false;
    });
    await redis.connect();
    redisReady = true;
    console.log('✅ Redis connected; session persistence enabled.');
  } catch (err) {
    console.warn('⚠️ Redis connection failed; falling back to in-memory sessions:', err && err.message ? err.message : err);
    redis = null;
    redisReady = false;
  }
}

// in-memory fallback
const memSessions = Object.create(null);

function nowMs() {
  return Date.now();
}

function generateSessionId() {
  return 'sess_' + Math.random().toString(36).slice(2) + '_' + Date.now().toString(36);
}

function sanitizeIntent(intent) {
  const VALID = ['chat', 'ideas', 'recipes', 'spec', 'code', 'scaffold', 'repo'];
  if (!intent || typeof intent !== 'string') return 'chat';
  const s = intent.trim().toLowerCase();
  return VALID.includes(s) ? s : 'chat';
}

function sanitizeRunMode(runMode) {
  if (!runMode || typeof runMode !== 'string') return 'direct';
  const s = runMode.trim().toLowerCase();
  // We allow unknown ids; backend will fall back to direct safely.
  return s;
}

function sanitizeText(s, maxLen) {
  const t = (s == null ? '' : String(s)).trim();
  if (!maxLen) return t;
  return t.length > maxLen ? t.slice(0, maxLen) : t;
}

function roughTokenEstimate(text) {
  if (!text) return 0;
  return Math.ceil(String(text).length / 4);
}

function blankSession(id) {
  return {
    id,
    createdAt: nowMs(),
    updatedAt: nowMs(),
    global: {
      // small cross-intent breadcrumb; user-only
      lastUser: []
    },
    lanes: {}
  };
}

function getLane(session, laneId) {
  if (!session.lanes) session.lanes = {};
  if (!session.lanes[laneId]) {
    session.lanes[laneId] = {
      id: laneId,
      summary: '',
      messages: []
    };
  }
  return session.lanes[laneId];
}

function pushUserBreadcrumb(session, text) {
  const t = sanitizeText(text, 500);
  if (!t) return;
  if (!session.global) session.global = { lastUser: [] };
  if (!Array.isArray(session.global.lastUser)) session.global.lastUser = [];
  session.global.lastUser.push({ ts: nowMs(), text: t });
  // Keep only last 6
  if (session.global.lastUser.length > 6) {
    session.global.lastUser = session.global.lastUser.slice(-6);
  }
}

async function loadSession(sessionId) {
  const sid = sessionId && typeof sessionId === 'string' ? sessionId : '';
  const id = sid && sid.startsWith('sess_') ? sid : '';
  const finalId = id || generateSessionId();

  if (redisReady && redis) {
    const key = `meld:sess:${finalId}`;
    try {
      const raw = await redis.get(key);
      if (raw) {
        const parsed = JSON.parse(raw);
        // Safety: never allow null / weird
        if (parsed && typeof parsed === 'object' && parsed.id) {
          return parsed;
        }
      }
    } catch (err) {
      // ignore and fall through to create new
    }
  } else if (memSessions[finalId]) {
    return memSessions[finalId];
  }

  return blankSession(finalId);
}

async function saveSession(session) {
  if (!session || !session.id) return;
  session.updatedAt = nowMs();

  if (redisReady && redis) {
    const key = `meld:sess:${session.id}`;
    try {
      await redis.set(key, JSON.stringify(session), { EX: SESSION_TTL_SEC });
    } catch (err) {
      // ignore
    }
  } else {
    memSessions[session.id] = session;
  }
}

// -------------------- Auth --------------------

function extractToken(req) {
  const hdr1 = req.headers['x-meld-token'];
  if (hdr1) return String(hdr1).trim();

  const hdr2 = req.headers['x-meld-backend-token'];
  if (hdr2) return String(hdr2).trim();

  const auth = req.headers['authorization'];
  if (auth && typeof auth === 'string') {
    const m = auth.match(/^Bearer\s+(.+)$/i);
    if (m) return m[1].trim();
  }
  return '';
}

function requireAuth(req, res, next) {
  if (!BACKEND_TOKEN) {
    return res.status(500).json({
      error: 'Server misconfigured',
      message: 'MELD_BACKEND_TOKEN is not set on the server.'
    });
  }
  const tok = extractToken(req);
  if (!tok || tok !== BACKEND_TOKEN) {
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Missing or invalid Authorization token.'
    });
  }
  next();
}

// -------------------- Providers / runModes --------------------

function getProvidersStatus() {
  return {
    openai: !!openai,
    xai: !!XAI_API_KEY
  };
}

function buildRunModes() {
  const providers = getProvidersStatus();
  const modes = [];

  modes.push({ id: 'direct', label: 'Direct · Fast', kind: 'single', available: providers.openai, scouts: [] });
  modes.push({ id: 'orchestrated', label: 'Orchestrated · Deep', kind: 'single', available: providers.openai, scouts: [] });

  if (providers.openai && providers.xai) {
    modes.push({ id: 'multi-xai', label: 'Multi · OpenAI + xAI', kind: 'multi', available: true, scouts: ['xai'] });
    modes.push({ id: 'multi-all', label: 'Multi · All Scouts', kind: 'multi', available: true, scouts: ['xai'] });
    modes.push({ id: 'ultimate', label: 'Ultimate · All Scouts + Verifier', kind: 'ultimate', available: true, scouts: ['xai'] });
  }

  return modes;
}

function resolveRunMode(runModeId) {
  const id = sanitizeRunMode(runModeId);
  const available = buildRunModes();
  const found = available.find((m) => m.id === id);
  if (found) return found;
  // fallback
  return available.find((m) => m.id === 'direct') || { id: 'direct', label: 'Direct · Fast', kind: 'single', available: true, scouts: [] };
}

// -------------------- Prompting --------------------

function buildSystemPrompt(intent) {
  switch (intent) {
    case 'ideas':
      return [
        'You are MELD.',
        'Intent: IDEAS.',
        'Return a numbered list of strong, concrete ideas. Each idea must be actionable.',
        'No meta commentary.'
      ].join('\n');
    case 'recipes':
      return [
        'You are MELD.',
        'Intent: RECIPES.',
        'Return restaurant-quality recipes with precise ingredients and steps. Include timing.',
        'No meta commentary.'
      ].join('\n');
    case 'spec':
      return [
        'You are MELD.',
        'Intent: SPEC.',
        'Return a structured spec with headings: Summary, Goals, Non-Goals, Requirements, Architecture/Design, Milestones, Risks, Open Questions.',
        'No meta commentary.'
      ].join('\n');
    case 'code':
      return [
        'You are MELD.',
        'Intent: CODE.',
        'Return complete runnable code (no pseudocode, no "..."). Keep explanation short and practical.',
        'No meta commentary.'
      ].join('\n');
    case 'scaffold':
      return [
        'You are MELD.',
        'Intent: SCAFFOLD.',
        'Return a project file tree first, then brief file descriptions (and small stubs if helpful).',
        'No meta commentary.'
      ].join('\n');
    default:
      return [
        'You are MELD.',
        'Intent: CHAT.',
        'Answer clearly and directly. Use Markdown if helpful.',
        'No meta commentary.'
      ].join('\n');
  }
}

function formatLaneContext(session, lane) {
  const laneObj = getLane(session, lane);
  const summary = laneObj.summary ? `Lane summary:\n${laneObj.summary}\n\n` : '';
  const recent = Array.isArray(laneObj.messages) ? laneObj.messages.slice(-8) : [];
  const convo = recent
    .map((m) => {
      const who = m.role === 'assistant' ? 'MELD' : 'User';
      return `${who}: ${m.content}`;
    })
    .join('\n');

  const breadcrumbs = (session.global && Array.isArray(session.global.lastUser) && session.global.lastUser.length)
    ? '\n\nRecent user breadcrumb (cross-intent, user-only):\n' +
      session.global.lastUser.map((x) => `- ${x.text}`).join('\n')
    : '';

  return `${summary}${convo}${breadcrumbs}`.trim();
}

function buildComposerPrompt(question, intent, runMode, contextText) {
  return [
    'You are MELD’s internal prompt composer.',
    'Task: rewrite the user’s request into a single improved prompt that will yield the best possible answer.',
    'Rules: output ONLY the improved prompt string. No labels, no bullets, no quotes.',
    '',
    `Run mode: ${runMode}`,
    `Intent: ${intent}`,
    '',
    'User question:',
    question,
    '',
    'Context:',
    contextText || '(none)'
  ].join('\n');
}

function buildAnswerPrompt(question, refinedPrompt, intent, runMode, contextText) {
  const sys = buildSystemPrompt(intent);
  return [
    sys,
    '',
    `Run mode: ${runMode}`,
    '',
    'Refined task:',
    refinedPrompt || question,
    '',
    'Context:',
    contextText || '(none)'
  ].join('\n');
}

function buildScoutPrompt(question, refinedPrompt, intent, contextText) {
  return [
    'You are a specialist assistant helping MELD.',
    `Intent: ${intent}`,
    'Answer the user directly and practically. Do not talk about being a model. Do not mention MELD.',
    '',
    'User question:',
    question,
    '',
    'Refined task:',
    refinedPrompt || question,
    '',
    'Context:',
    contextText || '(none)'
  ].join('\n');
}

function buildConductorPrompt(question, refinedPrompt, intent, contextText, scoutAnswers, verifierNotes) {
  const sys = buildSystemPrompt(intent);
  const scoutsBlock = scoutAnswers && scoutAnswers.length
    ? scoutAnswers.map((s, i) => `Scout ${i + 1} (${s.provider}):\n${s.text}`).join('\n\n-----\n\n')
    : '(no scouts)';

  const verifierBlock = verifierNotes ? `\n\nVerifier notes:\n${verifierNotes}` : '';

  return [
    sys,
    '',
    'You are the conductor. Synthesize the best final answer.',
    'Do NOT mention scouts or providers. Do NOT mention your process.',
    'If something is uncertain, say so briefly and propose a safe next step.',
    '',
    'User question:',
    question,
    '',
    'Refined task:',
    refinedPrompt || question,
    '',
    'Context:',
    contextText || '(none)',
    '',
    'Scout outputs:',
    scoutsBlock,
    verifierBlock
  ].join('\n');
}

function buildVerifierPrompt(question, intent, scoutAnswers) {
  const scoutsBlock = scoutAnswers && scoutAnswers.length
    ? scoutAnswers.map((s, i) => `Scout ${i + 1} (${s.provider}):\n${s.text}`).join('\n\n-----\n\n')
    : '(no scouts)';

  return [
    'You are a careful verifier.',
    `Intent: ${intent}`,
    'Task: identify likely errors, missing steps, contradictions, or unsafe assumptions in the scout outputs.',
    'Return a short bullet list of corrections and clarifications to apply in the final answer.',
    'Do not write the final answer. Only verification notes.',
    '',
    'User question:',
    question,
    '',
    'Scout outputs:',
    scoutsBlock
  ].join('\n');
}

// -------------------- OpenAI / xAI calls --------------------

async function callOpenAIText(model, promptText) {
  if (!openai) throw new Error('OpenAI not configured');
  const resp = await openai.responses.create({
    model,
    input: promptText
  });
  const text = (resp && resp.output_text) ? resp.output_text.trim() : '';
  const usage = resp && resp.usage ? resp.usage : null;
  const tokens = usage && typeof usage.total_tokens === 'number' ? usage.total_tokens : roughTokenEstimate(text);
  return { text, tokens };
}

async function callXAIText(model, promptText) {
  if (!XAI_API_KEY) throw new Error('xAI not configured');
  const url = `${XAI_BASE_URL}/v1/chat/completions`;
  const body = {
    model: model || XAI_MODEL,
    messages: [{ role: 'user', content: promptText }]
  };

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'authorization': `Bearer ${XAI_API_KEY}`
    },
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`xAI HTTP ${res.status}: ${txt.slice(0, 200)}`);
  }

  const data = await res.json();
  const choice = data && Array.isArray(data.choices) ? data.choices[0] : null;
  const text = choice && choice.message && choice.message.content ? String(choice.message.content).trim() : '';
  const tokens = roughTokenEstimate(text);
  return { text, tokens };
}

// -------------------- Summarization (optional safety) --------------------

async function maybeSummarizeLane(session, laneId) {
  // Keep it simple & safe: summarize only when lane grows big.
  const lane = getLane(session, laneId);
  if (!Array.isArray(lane.messages)) lane.messages = [];
  const maxMsgs = MAX_SESSION_TURNS * 2;
  if (lane.messages.length <= maxMsgs) return;

  // Summarize older part
  const keep = 8; // keep last 8 messages
  const older = lane.messages.slice(0, Math.max(0, lane.messages.length - keep));
  const recent = lane.messages.slice(-keep);

  const olderText = older
    .map((m) => `${m.role === 'assistant' ? 'MELD' : 'User'}: ${m.content}`)
    .join('\n');

  const prompt = [
    'Summarize the following conversation briefly for future context.',
    'Rules: 6-10 bullet points max. Include key requirements, constraints, and decisions.',
    'Do not include filler.',
    '',
    olderText
  ].join('\n');

  try {
    const r = await callOpenAIText(OPENAI_FAST_MODEL, prompt);
    lane.summary = sanitizeText(r.text, 2000);
    lane.messages = recent;
  } catch (err) {
    // If summarization fails, just trim to recent safely.
    lane.messages = recent;
  }
}

// -------------------- Routes --------------------

app.get('/', (req, res) => {
  res.type('text/plain').send('MELD backend is running. Use GET /health, POST /chat, POST /repo.');
});

app.get('/health', (req, res) => {
  const providers = getProvidersStatus();
  const runModes = buildRunModes();
  res.json({
    status: 'ok',
    service: 'meld-backend',
    version: VERSION,
    time: new Date().toISOString(),
    models: {
      openai: { fast: OPENAI_FAST_MODEL, deep: OPENAI_DEEP_MODEL },
      xai: { default: XAI_MODEL }
    },
    providers,
    runModes,
    auth: {
      requiredOn: ['/chat', '/repo'],
      acceptedHeaders: ['Authorization: Bearer <token>', 'X-MELD-TOKEN: <token>', 'X-MELD-BACKEND-TOKEN: <token>']
    },
    openaiAvailable: !!openai,
    sessionStore: { type: redisReady ? 'redis' : 'memory', enabled: redisReady }
  });
});

app.post('/chat', requireAuth, async (req, res) => {
  try {
    const body = req.body || {};
    const question = typeof body.question === 'string' ? body.question.trim() : '';
    const intent = sanitizeIntent(body.intent);
    const runMode = resolveRunMode(body.runMode || body.mode || 'direct');
    const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';

    if (!question) {
      return res.status(400).json({ error: 'Missing question', message: 'The "question" field is required.' });
    }

    if (!openai) {
      return res.status(500).json({ error: 'OpenAI not configured', message: 'OPENAI_API_KEY is required.' });
    }

    const session = await loadSession(sessionId);

    // Lane per intent
    const laneId = intent;
    const lane = getLane(session, laneId);

    const contextText = formatLaneContext(session, laneId);

    // 1) Composer
    const composerPrompt = buildComposerPrompt(question, intent, runMode.id, contextText);
    const composer = await callOpenAIText(OPENAI_FAST_MODEL, composerPrompt);
    const refinedPrompt = sanitizeText(composer.text || question, 6000);

    // 2) Answer
    let answerText = '';
    let usedProviders = [`openai:${OPENAI_FAST_MODEL}`];

    if (runMode.kind === 'single') {
      const answerPrompt = buildAnswerPrompt(question, refinedPrompt, intent, runMode.label, contextText);
      const ans = await callOpenAIText(OPENAI_DEEP_MODEL, answerPrompt);
      answerText = ans.text;
      usedProviders.push(`openai:${OPENAI_DEEP_MODEL}`);
    } else {
      // Scouts (currently xAI only in this rebuild)
      const scouts = [];
      if (runMode.scouts.includes('xai') && XAI_API_KEY) {
        const scoutPrompt = buildScoutPrompt(question, refinedPrompt, intent, contextText);
        const scout = await callXAIText(XAI_MODEL, scoutPrompt);
        scouts.push({ provider: 'xai', text: scout.text });
        usedProviders.push(`xai:${XAI_MODEL}`);
      }

      let verifierNotes = '';
      if (runMode.kind === 'ultimate' && scouts.length) {
        const verifierPrompt = buildVerifierPrompt(question, intent, scouts);
        const ver = await callOpenAIText(OPENAI_FAST_MODEL, verifierPrompt);
        verifierNotes = sanitizeText(ver.text, 2000);
        usedProviders.push(`openai:${OPENAI_FAST_MODEL}:verifier`);
      }

      const conductorPrompt = buildConductorPrompt(question, refinedPrompt, intent, contextText, scouts, verifierNotes);
      const cond = await callOpenAIText(OPENAI_DEEP_MODEL, conductorPrompt);
      answerText = cond.text;
      usedProviders.push(`openai:${OPENAI_DEEP_MODEL}:conductor`);
    }

    // Persist turn
    lane.messages.push({ role: 'user', content: sanitizeText(question, 8000), ts: nowMs() });
    lane.messages.push({ role: 'assistant', content: sanitizeText(answerText, 12000), ts: nowMs() });

    // Trim per-lane turns
    const maxMsgs = MAX_SESSION_TURNS * 2;
    if (lane.messages.length > maxMsgs) lane.messages = lane.messages.slice(-maxMsgs);

    pushUserBreadcrumb(session, question);

    await maybeSummarizeLane(session, laneId);
    await saveSession(session);

    res.json({
      answer: answerText,
      sessionId: session.id,
      intent,
      runMode: runMode.id,
      meta: {
        usedProviders,
        tokensEstimate: composer.tokens + roughTokenEstimate(answerText),
        refinedPrompt
      }
    });
  } catch (err) {
    console.error('Error in /chat:', err);
    res.status(500).json({
      error: 'MELD backend error',
      message: err && err.message ? err.message : 'Unknown error'
    });
  }
});

app.post('/repo', requireAuth, async (req, res) => {
  try {
    const body = req.body || {};
    const question = typeof body.question === 'string' ? body.question.trim() : '';
    const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';

    if (!question) {
      return res.status(400).json({ error: 'Missing question', message: 'The "question" field is required.' });
    }
    if (!openai) {
      return res.status(500).json({ error: 'OpenAI not configured', message: 'OPENAI_API_KEY is required.' });
    }

    const session = await loadSession(sessionId);
    const contextText = formatLaneContext(session, 'repo');

    const prompt = [
      'You are MELD’s repository architect.',
      'Write the full content of README.md for a small starter project that satisfies the user request.',
      'Include: name, overview, stack, folder tree, setup/run, next steps.',
      'Output valid Markdown only.',
      '',
      'User request:',
      question,
      '',
      'Context:',
      contextText || '(none)'
    ].join('\n');

    const r = await callOpenAIText(OPENAI_DEEP_MODEL, prompt);
    const readme = r.text || '# MELD Project\n';

    const zip = new JSZip();
    zip.file('README.md', readme);
    zip.file('QUESTION.txt', question);

    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `meld-project-${stamp}.zip`;

    const buf = await zip.generateAsync({ type: 'nodebuffer' });
    const zipBase64 = buf.toString('base64');

    // persist a breadcrumb
    pushUserBreadcrumb(session, `[repo] ${question}`);
    await saveSession(session);

    res.json({ filename, zipBase64, sessionId: session.id });
  } catch (err) {
    console.error('Error in /repo:', err);
    res.status(500).json({
      error: 'MELD repo error',
      message: err && err.message ? err.message : 'Unknown error'
    });
  }
});

// -------------------- Start --------------------

initRedis().finally(() => {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`MELD backend listening on port ${PORT}`);
    console.log(`- OpenAI models: fast=${OPENAI_FAST_MODEL}, deep=${OPENAI_DEEP_MODEL}`);
    console.log(`- xAI scout: ${XAI_API_KEY ? 'enabled' : 'disabled'}`);
    console.log(`- Session store: ${redisReady ? 'redis' : 'memory'} (enabled=${redisReady})`);
  });
});
