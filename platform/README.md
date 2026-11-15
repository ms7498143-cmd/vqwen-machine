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
| `GET /api/cpu` | CPU-clone capability report and AI provenance |
| `POST /api/cpu` `{"example":"sum_1_to_100"}` | Run a pre-assembled program on the x86-64 clone |
| `POST /api/cpu` `{"code":"<hex>"}` | Run arbitrary machine code on the clone |
| `POST /api/cpu` `{"generate":"the ","n":80}` | Run the neural network on the clone |

Every number returned is produced by real work at request time. Nothing is
estimated or hard-coded.

### The CPU clone (`/cpu/`)

An independently implemented x86-64 engine runs on Vercel. It decodes real
machine code and executes it against its own registers, flags and memory.
Correctness is verified against the physical CPU: 54 fixed and 14,500
randomized cases, 0 mismatches.

The neural network at `/api/cpu` runs on that clone. Its whole forward pass is
x86-64 machine code that the clone decodes and executes instruction by
instruction; no math library performs the inference. See
`device/CPU_CLONE_REPORT.md` for the full, honest report — including the
measured ~8,300x slowdown versus native execution, which is the unavoidable
consequence of running a CPU in software.

## Add a new device or job (future tasks)

1. Create `platform/api/<name>.js` exporting a handler `(req, res) => {}`.
2. Put reusable real-work code in `platform/lib/`.
3. Commit and push to `main`. Vercel auto-deploys.

The function runs on Vercel's servers with 1024 MB and up to 60s. That is the
budget to design within.

## Deploying without this workspace

Two ways, neither requires a development workspace:

**1. Automatic (one-time browser setup, then forever hands-off).**
Vercel's Git integration must be connected to the GitHub account that owns the
repo. This needs a browser OAuth step once:
   - https://vercel.com/account/integrations -> connect GitHub -> authorize
     the account `ms7498143-cmd`.
   - Then link the project: Vercel dashboard -> `vqwen-platform` -> Settings ->
     Git -> connect `ms7498143-cmd/vqwen-machine`, production branch `main`.
After that, every push to `main` deploys automatically with no token and no
workspace. This is the intended permanent path.

**2. Manual, from any machine.**
```bash
git clone https://github.com/ms7498143-cmd/vqwen-machine
cd vqwen-machine/platform
VERCEL_TOKEN=... ./deploy.sh
```
`deploy.sh` needs only `VERCEL_TOKEN`; it links and deploys production.

## Continuous / scheduled work

GitHub Actions runs on GitHub's servers. To use it, the push token needs the
`workflow` scope (the current token does not have it, which blocks adding
workflow files). Once added:

- Add `.github/workflows/<name>.yml` with a `schedule:` or `workflow_dispatch:` trigger.

Vercel Cron (`vercel.json` -> `crons`) is the alternative when the work is a
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
