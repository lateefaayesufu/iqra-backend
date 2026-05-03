
'use strict';

const express = require('express');
const cors    = require('cors');

const app = express();

const PORT           = process.env.PORT || 3000;
const CONTENT_MAX    = 10000;
const RATE_WINDOW_MS = 60000;
const RATE_LIMIT_MAX = 20;

// ── CORS ─────────────────────────────────────────────────────
app.use(cors({
  origin: (origin, cb) => {
    if (!origin || origin.startsWith('chrome-extension://') || origin === 'null') {
      cb(null, true);
    } else {
      cb(new Error('Not allowed by CORS'));
    }
  },
  methods: ['POST', 'GET'],
  allowedHeaders: ['Content-Type', 'x-iqra-token']
}));

app.use(express.json({ limit: '50kb' }));

// ── Rate Limiter ──────────────────────────────────────────────
const rateLimitMap = new Map();

function rateLimit(req, res, next) {
  const ip  = req.headers['x-forwarded-for']?.split(',')[0]?.trim() || req.socket.remoteAddress || 'unknown';
  const now = Date.now();
  if (!rateLimitMap.has(ip)) rateLimitMap.set(ip, []);
  const timestamps = rateLimitMap.get(ip).filter(ts => now - ts < RATE_WINDOW_MS);
  timestamps.push(now);
  rateLimitMap.set(ip, timestamps);
  if (timestamps.length > RATE_LIMIT_MAX) {
    const waitSecs = Math.ceil((RATE_WINDOW_MS - (now - timestamps[0])) / 1000);
    return res.status(429).json({ ok: false, error: `Rate limit reached. Please wait ${waitSecs}s.` });
  }
  if (Math.random() < 0.01) {
    for (const [key, tsList] of rateLimitMap.entries()) {
      if (tsList.every(ts => now - ts > RATE_WINDOW_MS)) rateLimitMap.delete(key);
    }
  }
  next();
}

// ── Token Guard ───────────────────────────────────────────────
function tokenGuard(req, res, next) {
  const secret = process.env.IQRA_SECRET;
  if (!secret) return next();
  if (req.headers['x-iqra-token'] !== secret)
    return res.status(401).json({ ok: false, error: 'Unauthorized.' });
  next();
}

// ── Prompt Builder ────────────────────────────────────────────
function buildPrompt(content, mode, wordCount, readingTime) {
  const truncated = content.slice(0, CONTENT_MAX);
  const modeInstructions = {
    full:       'Provide 4 to 6 comprehensive key insights. Each should be a complete, informative sentence.',
    '3bullets': 'Provide exactly 3 concise, high-impact bullet points capturing the most important takeaways.',
    quotes:     'Extract 3 of the most notable sentences directly from the text (preserve exact wording).'
  };
  return `You are an expert summarizer. Analyze this webpage content and ${modeInstructions[mode] || modeInstructions.full}

Page content (${wordCount} words, ~${readingTime} min read):
---
${truncated}
---

You MUST respond ONLY with a valid JSON object in this exact format, nothing else:
{"insights":["sentence 1","sentence 2","sentence 3"],"readingTime":${readingTime},"wordCount":${wordCount},"keyPhrases":["phrase1","phrase2","phrase3"]}

Rules:
- "insights": array of strings (the summary points)
- "keyPhrases": 3-5 short phrases verbatim from the text (for in-page highlighting)
- "readingTime": use ${readingTime}
- "wordCount": use ${wordCount}`;
}

// ── Gemini 2.5 Flash ──────────────────────────────────────────
async function callGemini(prompt) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY not configured on server.');

  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`;

  const res = await fetch(url, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: {
        maxOutputTokens: 1024,
        temperature: 0.3,
        responseMimeType: 'application/json',
        thinkingConfig: {
          thinkingBudget: 0
        }
      }
    })
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error?.message || `Gemini error (${res.status})`);
  }
  const data = await res.json();
  return data.candidates?.[0]?.content?.parts?.[0]?.text || '';
}

// ── Anthropic (fallback) ──────────────────────────────────────
async function callAnthropic(prompt) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error('ANTHROPIC_API_KEY not configured on server.');
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' },
    body: JSON.stringify({ model: 'claude-haiku-4-5-20251001', max_tokens: 1024, messages: [{ role: 'user', content: prompt }] })
  });
  if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.error?.message || `Anthropic error (${res.status})`); }
  const data = await res.json();
  return data.content?.[0]?.text || '';
}

// ── OpenAI (fallback) ─────────────────────────────────────────
async function callOpenAI(prompt) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('OPENAI_API_KEY not configured on server.');
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({ model: 'gpt-4o-mini', max_tokens: 1024, messages: [{ role: 'user', content: prompt }] })
  });
  if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.error?.message || `OpenAI error (${res.status})`); }
  const data = await res.json();
  return data.choices?.[0]?.message?.content || '';
}

// ── Parser ────────────────────────────────────────────────────
function parseAIResponse(raw) {
  let text = raw.replace(/```json\s*/gi, '').replace(/```\s*/g, '').trim();
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error('AI returned unexpected format.');
  let parsed;
  try { parsed = JSON.parse(match[0]); }
  catch (_) { throw new Error('Failed to parse AI response.'); }
  if (!Array.isArray(parsed.insights) || parsed.insights.length === 0)
    throw new Error('AI returned empty summary.');
  const sanitize = s => String(s).replace(/</g, '&lt;').replace(/>/g, '&gt;').slice(0, 500);
  return {
    insights:    parsed.insights.map(sanitize).filter(Boolean),
    readingTime: Number(parsed.readingTime) || 1,
    wordCount:   Number(parsed.wordCount)   || 0,
    keyPhrases:  Array.isArray(parsed.keyPhrases)
      ? parsed.keyPhrases.map(sanitize).filter(s => s.length > 2).slice(0, 6)
      : []
  };
}

// ── Routes ────────────────────────────────────────────────────
app.get('/', (_req, res) => {
  res.json({ ok: true, service: 'Iqra AI Summarizer Backend', version: '1.0.0' });
});

app.post('/api/summarize', rateLimit, tokenGuard, async (req, res) => {
  try {
    const { content, mode, wordCount, readingTime } = req.body;
    if (!content || typeof content !== 'string')
      return res.status(400).json({ ok: false, error: 'Missing or invalid content.' });
    if (content.trim().length < 50)
      return res.status(400).json({ ok: false, error: 'Content too short to summarize.' });
    if (content.length > 50000)
      return res.status(400).json({ ok: false, error: 'Content too long.' });

    const safeMode  = ['full', '3bullets', 'quotes'].includes(mode) ? mode : 'full';
    const safeWords = Math.max(0, parseInt(wordCount) || 0);
    const safeTime  = Math.max(1, parseInt(readingTime) || 1);
    const prompt    = buildPrompt(content, safeMode, safeWords, safeTime);

    let rawText = '';
    if      (process.env.GEMINI_API_KEY)    rawText = await callGemini(prompt);
    else if (process.env.ANTHROPIC_API_KEY) rawText = await callAnthropic(prompt);
    else if (process.env.OPENAI_API_KEY)    rawText = await callOpenAI(prompt);
    else throw new Error('No AI provider API key configured on server.');

    const result = parseAIResponse(rawText);
    return res.json({ ok: true, data: result });

  } catch (err) {
    console.error('[Iqra] Error:', err.message);
    return res.status(500).json({ ok: false, error: err.message || 'Internal server error.' });
  }
});

app.use((_req, res) => res.status(404).json({ ok: false, error: 'Route not found.' }));

app.listen(PORT, () => console.log(`✦ Iqra backend running on port ${PORT}`));

module.exports = app;
