'use strict';
// Platform status: describes what runs where, with no workspace dependency.
const models = require('../models.json');

module.exports = function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.status(200).json({
    platform: 'workspace-independent',
    source_of_truth: 'github.com/ms7498143-cmd/vqwen-machine',
    runtime: 'Vercel serverless functions',
    note: 'All execution happens on Vercel infrastructure. The development workspace is not required.',
    devices: [
      { id: 'vercel-job-runner', endpoint: '/api/device', jobs: ['compute','memory','storage','network'],
        description: 'Runs real measurement jobs on Vercel servers.' },
      { id: 'fleet-runner', endpoint: '/api/fleet?devices=8', jobs: ['parallel-compute'],
        description: 'Runs many device jobs in parallel and reports aggregate throughput.' },
      { id: 'infer-device', endpoint: '/api/infer?prompt=hello', jobs: ['text-generation'],
        description: 'Runs the registered model inside the device via ONNX Runtime.' },
    ],
    models: models.models.map((m) => ({
      id: m.id, source: m.source, params: m.params,
      quantization: m.quantization, measured_tokens_per_sec: m.measured_tokens_per_sec,
    })),
    endpoints: {
      status: '/api/status',
      device: '/api/device?job=all',
      fleet: '/api/fleet?devices=8',
      infer: '/api/infer?prompt=hello',
      models: '/models.json',
    },
    server_time: new Date().toISOString(),
    region: process.env.VERCEL_REGION || 'unknown',
  });
};
