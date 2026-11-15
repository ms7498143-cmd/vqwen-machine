'use strict';
// Device endpoint: runs real measurement jobs on Vercel's servers.
// This function executes entirely on Vercel infrastructure — no dependency
// on any development workspace.
const { cpuBench, memBench, storageBench, networkBench } = require('../lib/measure');

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();

  const url = new URL(req.url, `https://${req.headers.host || 'localhost'}`);
  const job = (url.searchParams.get('job') || 'all').toLowerCase();
  const started = Date.now();
  const results = {};

  try {
    if (job === 'all' || job === 'compute') results.compute = cpuBench(1200);
    if (job === 'all' || job === 'memory') results.memory = memBench(128);
    if (job === 'all' || job === 'storage') results.storage = storageBench(48);
    if (job === 'all' || job === 'network') {
      results.network = await networkBench('https://speed.cloudflare.com/__down?bytes=8000000');
    }
  } catch (err) {
    results.error = String(err && err.message ? err.message : err);
  }

  res.status(200).json({
    device: 'vercel-job-runner',
    runtime: 'Vercel serverless function',
    region: process.env.VERCEL_REGION || 'unknown',
    deployment: process.env.VERCEL_URL || 'local',
    job,
    ran_at: new Date().toISOString(),
    wall_ms: Date.now() - started,
    results,
  });
};
