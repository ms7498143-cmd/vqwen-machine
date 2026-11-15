# vqwen platform — runs on GitHub + Vercel, no workspace required

A permanent home for devices, servers and jobs that **does not depend on any
development workspace**. Source lives in GitHub; execution happens on Vercel's
servers. Reset or close the workspace and everything here keeps running.

## Why this is workspace-independent

| Layer | Runs on | Survives workspace reset |
|---|---|---|
| Source of truth | GitHub (`ms7498143-cmd/vqwen-machine`) | yes |
| Web pages | Vercel edge/CDN | yes |
| Jobs / servers | Vercel serverless functions | yes |
| Auto-deploy | Vercel Git integration | yes |

Nothing here reads from a local disk, a local port, or a tunnel. If a piece of
work needs to run forever, it belongs in one of the two layers above.

## Live endpoints

Base: `https://vqwen-platform.vercel.app`

| Endpoint | Purpose |
|---|---|
| `GET /api/status` | What runs where, region, server time |
| `GET /api/device?job=all` | Run all real measurement jobs, return results |
| `GET /api/device?job=compute` | CPU throughput |
| `GET /api/device?job=memory` | Memory bandwidth |
| `GET /api/device?job=storage` | Storage throughput |
| `GET /api/device?job=network` | Network download rate |

Every number returned is produced by real work at request time. Nothing is
estimated or hard-coded.

## Add a new device or job (future tasks)

1. Create `platform/api/<name>.js` exporting a handler `(req, res) => {}`.
2. Put reusable real-work code in `platform/lib/`.
3. Commit and push to `main`. Vercel auto-deploys.

The function runs on Vercel's servers with 1024 MB and up to 60s. That is the
budget to design within.

## Adding continuous / scheduled work

GitHub Actions runs on GitHub's servers — use it for anything scheduled or
long-running that does not need to answer an HTTP request:

- Add `.github/workflows/<name>.yml` with a `schedule:` or `workflow_dispatch:` trigger.
- Requires the push token to carry the `workflow` scope.

Vercel Cron (`vercel.json` → `crons`) is the alternative when the work is a
Vercel function to be called on a schedule.

## Layout

```
platform/
  api/device.js     real measurement jobs (Vercel function)
  api/status.js     platform description
  lib/measure.js    real measurement primitives (CPU, memory, storage, network)
  public/index.html dashboard that calls the functions
  vercel.json       function memory/duration + no-store on /api
```

## Honest boundaries

- Vercel functions are **ephemeral**: each request gets a fresh, short-lived
  sandbox. Storage written there does not persist between requests. For durable
  data use a database or object store, not the function filesystem.
- Hobby plan caps function memory at 2048 MB and has no GPU. Heavy models do not
  fit; light jobs do.
- These jobs measure the **Vercel runtime**, which is real infrastructure but not
  a dedicated physical machine owned by you.

## Security

Revoke the tokens used to build this when done:
- GitHub: https://github.com/settings/personal-access-tokens
- Vercel: https://vercel.com/account/tokens
