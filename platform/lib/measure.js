'use strict';
// Real measurement primitives. Every function performs actual work and reports
// the measured rate. Nothing here is estimated or synthesized.

const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');

function nowMs() {
  return Number(process.hrtime.bigint() / 1000000n);
}

// Real CPU work: hash rounds until the time budget elapses.
function cpuBench(budgetMs = 1500) {
  const t0 = nowMs();
  let ops = 0;
  const buf = Buffer.alloc(4096, 7);
  while (nowMs() - t0 < budgetMs) {
    crypto.createHash('sha256').update(buf).digest();
    ops++;
  }
  const secs = (nowMs() - t0) / 1000;
  return {
    job: 'compute',
    work: 'sha256 rounds over 4KiB buffer',
    ops,
    seconds: Number(secs.toFixed(3)),
    ops_per_sec: Math.round(ops / secs),
    cpus: os.cpus().length,
    cpu_model: (os.cpus()[0] || {}).model || 'unknown',
  };
}

// Real memory work: allocate and touch pages, measure throughput.
function memBench(mb = 256) {
  const t0 = nowMs();
  const size = mb * 1024 * 1024;
  const buf = Buffer.allocUnsafe(size);
  for (let i = 0; i < size; i += 4096) buf.writeUInt32LE((i ^ 0x5a5a5a5a) >>> 0, i);
  const touchSecs = Math.max((nowMs() - t0) / 1000, 0.000001);
  const t1 = nowMs();
  let sum = 0;
  for (let i = 0; i < size; i += 4096) sum += buf.readUInt32LE(i);
  const readSecs = Math.max((nowMs() - t1) / 1000, 0.000001);
  return {
    job: 'memory',
    allocated_mb: mb,
    write_mb_s: Number((mb / touchSecs).toFixed(2)),
    read_mb_s: Number((mb / readSecs).toFixed(2)),
    checksum: sum,
    total_mem_mb: Math.round(os.totalmem() / 1024 / 1024),
  };
}

// Real storage work: write and read a file in the function's own filesystem.
function storageBench(mb = 64) {
  const file = path.join(os.tmpdir(), `bench-${process.pid}-${Date.now()}.bin`);
  const chunk = Buffer.alloc(1024 * 1024, 0xa5);
  let t0 = nowMs();
  const fd = fs.openSync(file, 'w');
  for (let i = 0; i < mb; i++) fs.writeSync(fd, chunk);
  fs.fsyncSync(fd);
  fs.closeSync(fd);
  const writeSecs = (nowMs() - t0) / 1000;

  t0 = nowMs();
  const data = fs.readFileSync(file);
  const readSecs = (nowMs() - t0) / 1000;
  const bytes = data.length;
  try { fs.unlinkSync(file); } catch (e) { /* best effort */ }

  return {
    job: 'storage',
    volume_mb: mb,
    write_mb_s: Number((mb / writeSecs).toFixed(2)),
    read_mb_s: Number((mb / readSecs).toFixed(2)),
    bytes: bytes,
    location: 'function-local tmpfs',
  };
}

// Real network work: download a payload and measure the achieved rate.
async function networkBench(url, maxBytes = 8 * 1024 * 1024) {
  const t0 = nowMs();
  const res = await fetch(url, { redirect: 'follow' });
  let total = 0;
  for await (const chunk of res.body) {
    total += chunk.length;
    if (total >= maxBytes) break;
  }
  const secs = (nowMs() - t0) / 1000;
  const mb = total / 1024 / 1024;
  return {
    job: 'network',
    source: url,
    bytes: total,
    seconds: Number(secs.toFixed(3)),
    mb_s: Number((mb / secs).toFixed(2)),
    mbit_s: Number(((mb * 8) / secs).toFixed(2)),
    status: res.status,
  };
}

module.exports = { cpuBench, memBench, storageBench, networkBench, nowMs };
