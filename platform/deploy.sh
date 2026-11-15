#!/usr/bin/env bash
# Redeploy the platform to Vercel from any environment.
# Requires only the VERCEL_TOKEN environment variable. No workspace dependency:
# clone the repo anywhere and run this.
set -euo pipefail

: "${VERCEL_TOKEN:?set VERCEL_TOKEN before running}"
PROJECT="${VERCEL_PROJECT:-vqwen-platform}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$HERE"
echo "Deploying $PROJECT from $HERE"

if ! command -v vercel >/dev/null 2>&1; then
  npm install -g vercel >/dev/null 2>&1 || npx vercel --version >/dev/null
fi

VERCEL="$(command -v vercel || echo 'npx vercel')"

$VERCEL link --yes --project "$PROJECT" --token "$VERCEL_TOKEN" >/dev/null
$VERCEL deploy --prod --yes --token "$VERCEL_TOKEN"
