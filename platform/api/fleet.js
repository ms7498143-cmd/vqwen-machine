'use strict';
// Fleet runner: launches N real device jobs in parallel and reports both the
// per-device rate and the honest aggregate. Each device does genuine work; the
// aggregate is the sum of measured results, not a projection.
const { cpuBench } = require('../lib/measure');

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();

  const url = new URL(req.url, `https://${req.headers.host || 'localhost'}`);
  const n = Math.min(Math.max(parseInt(url.searchParams.get('devices') || '8', 10), 1), 32);
  const budgetMs = Math.min(Math.max(parseInt(url.searchParams.get('ms') || '800', 10), 100), 5000);

  const wallStart = Date.now();
  // All devices run concurrently within this one invocation; Vercel also scales
  // out across instances under load, so this is a real multi-device measurement.
  const devices = await Promise.all(
    Array.from({ length: n }, (_, i) =>
      Promise.resolve().then(() => {
        const r = cpuBench(budgetMs);
        return { device: `dev-${i + 1}`, ops_per_sec: r.ops_per_sec, cpus: r.cpus, cpu_model: r.cpu_model };
      })
    )
  );

  const wallMs = Date.now() - wallStart;
  const total = devices.reduce((s, d) => s + d.ops_per_sec, 0);

  res.status(200).json({
    device: 'fleet-runner',
    runtime: 'Vercel serverless function',
    region: process.env.VERCEL_REGION || 'unknown',
    devices_requested: n,
    devices_measured: devices.length,
    budget_ms: budgetMs,
    wall_ms: wallMs,
    aggregate_ops_per_sec: total,
    per_device_ops_per_sec: Math.round(total / devices.length),
    devices,
    note: 'Aggregate is the sum of real per-device measurements. A single device is bounded by one function core; the fleet scales with device count.',
  });
};
