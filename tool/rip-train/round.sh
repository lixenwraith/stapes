#!/usr/bin/env bash
# One round for a machine nobody watches, every line kept in $CORPUS/round.log: the
# corpus, the models and the bench, then the fit where its lists are named. Run it
# again to resume: each stage goes on from what it left. The knobs are
# pipeline.sh's; the README's "A round on a development machine" is the guide.
set -euo pipefail
CORPUS=$(realpath -m "${CORPUS:-${XDG_STATE_HOME:-$HOME/.local/state}/stapes/corpus}")
export CORPUS
cd "$(dirname "$0")/../.."
mkdir -p "$CORPUS"
log=$CORPUS/round.log
stage() { tool/rip-train/pipeline.sh "$@" 2>&1 | tee -a "$log"; }
[ -f "$CORPUS/corpus.json" ] || stage corpus
stage train
[ -z "${NOTES:-}" ] || stage notes
stage bench
[ -z "${ROWS:-}" ] || stage rows judge
[ -z "${TRUTH:-}" ] || stage fit
[ -z "${READBACK:-}" ] || stage readback table
[ -z "${WEIGHTS:-}" ] || stage keep
