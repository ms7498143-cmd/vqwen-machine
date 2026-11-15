'use strict';
// Inference device: runs the registered model inside the function, on Vercel's
// servers. The development workspace is not involved at any point.
const { generate, MODEL_ID } = require('../lib/infer');

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();

  const url = new URL(req.url, `https://${req.headers.host || 'localhost'}`);
  let prompt = url.searchParams.get('prompt');
  let maxTokens = parseInt(url.searchParams.get('max_tokens') || '48', 10);

  if (req.method === 'POST' && req.body) {
    try {
      const body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
      if (body.prompt) prompt = body.prompt;
      if (body.max_tokens) maxTokens = parseInt(body.max_tokens, 10);
    } catch (e) { /* fall through to query params */ }
  }

  if (!prompt) {
    return res.status(200).json({
      device: 'infer-device',
      model: MODEL_ID,
      usage: 'GET or POST with ?prompt=...&max_tokens=48',
      runtime: 'Vercel serverless function',
      region: process.env.VERCEL_REGION || 'unknown',
    });
  }

  try {
    const started = Date.now();
    const result = await generate(prompt, Math.min(Math.max(maxTokens, 1), 256));
    res.status(200).json({
      device: 'infer-device',
      runtime: 'Vercel serverless function',
      region: process.env.VERCEL_REGION || 'unknown',
      wall_ms: Date.now() - started,
      ...result,
    });
  } catch (err) {
    res.status(500).json({
      device: 'infer-device',
      model: MODEL_ID,
      error: String(err && err.message ? err.message : err),
    });
  }
};
