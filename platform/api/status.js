'use strict';
// Platform status: describes what runs where, with no workspace dependency.
module.exports = function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.status(200).json({
    platform: 'workspace-independent',
    source_of_truth: 'github.com/ms7498143-cmd/vqwen-machine',
    runtime: 'Vercel serverless functions',
    note: 'All execution happens on Vercel infrastructure. The development workspace is not required.',
    devices: [
      {
        id: 'vercel-job-runner',
        endpoint: '/api/device',
        jobs: ['compute', 'memory', 'storage', 'network'],
        description: 'Runs real measurement jobs on Vercel servers.',
      },
    ],
    endpoints: {
      status: '/api/status',
      device: '/api/device?job=all',
    },
    server_time: new Date().toISOString(),
    region: process.env.VERCEL_REGION || 'unknown',
  });
};
