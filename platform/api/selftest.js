'use strict';
// Diagnostic: report exactly how far the inference stack gets inside the function.
module.exports = async function handler(req, res) {
  const steps = [];
  const t0 = Date.now();
  try {
    steps.push({ step: 'start', ms: 0 });
    const fs = require('fs');
    const cacheDir = process.env.HF_CACHE || '/tmp/hf-cache';
    fs.mkdirSync(cacheDir, { recursive: true });
    steps.push({ step: 'cache dir ok', dir: cacheDir, ms: Date.now() - t0 });
    const tf = await import('@huggingface/transformers');
    steps.push({ step: 'import ok', version: tf.env && tf.env.version, ms: Date.now() - t0 });
    tf.env.cacheDir = cacheDir;
    tf.env.allowLocalModels = false;
    const pipe = await tf.pipeline('text-generation', 'onnx-community/Qwen2.5-0.5B-Instruct', { dtype: 'q4f16' });
    steps.push({ step: 'pipeline ok', ms: Date.now() - t0 });
    const out = await pipe([{ role: 'user', content: 'Say hi.' }], { max_new_tokens: 8, do_sample: false });
    steps.push({ step: 'generate ok', reply: out[0].generated_text.at(-1).content.slice(0, 80), ms: Date.now() - t0 });
    res.status(200).json({ ok: true, steps, total_ms: Date.now() - t0 });
  } catch (err) {
    res.status(200).json({ ok: false, steps, error: String(err && err.stack ? err.stack : err).slice(0, 1200), total_ms: Date.now() - t0 });
  }
};
