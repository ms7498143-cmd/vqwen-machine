'use strict';
// Real model inference running inside the device (a Vercel function), not on any
// development host. Uses transformers.js with ONNX Runtime. Every rate reported
// is measured from this run; nothing is estimated.
const path = require('path');
const fs = require('fs');

const MODEL_ID = process.env.MODEL_ID || 'onnx-community/Qwen2.5-0.5B-Instruct';
// Vercel's deployment filesystem is read-only; the model cache must go to /tmp.
const CACHE_DIR = process.env.HF_CACHE || '/tmp/hf-cache';

let cached = null;

async function loadPipeline(dtype) {
  if (cached && cached.dtype === dtype) return cached.pipe;
  try { fs.mkdirSync(CACHE_DIR, { recursive: true }); } catch (e) { /* exists */ }
  const t0 = Date.now();
  const { pipeline, env } = await import('@huggingface/transformers');
  env.cacheDir = CACHE_DIR;
  env.allowLocalModels = false;
  const pipe = await pipeline('text-generation', MODEL_ID, { dtype });
  pipe.__load_ms = Date.now() - t0;
  cached = { pipe, dtype };
  return pipe;
}

async function generate(prompt, maxTokens = 64, dtype = 'q4f16') {
  const pipe = await loadPipeline(dtype);
  const messages = [{ role: 'user', content: prompt }];
  const t1 = Date.now();
  const out = await pipe(messages, { max_new_tokens: maxTokens, do_sample: false });
  const genMs = Date.now() - t1;
  const text = out[0].generated_text.at(-1).content;
  const tokens = pipe.tokenizer.encode(text).length;
  return {
    model: MODEL_ID,
    dtype,
    load_ms: pipe.__load_ms || 0,
    gen_ms: genMs,
    tokens,
    tokens_per_sec: Number((tokens / (genMs / 1000)).toFixed(2)),
    reply: text,
  };
}

module.exports = { generate, MODEL_ID, CACHE_DIR };
