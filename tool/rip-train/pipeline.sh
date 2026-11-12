#!/usr/bin/env bash
# The frame model's training pipeline: corpus, train, bench and install, or all four
# in order; shard and merge run a training round over several workers. Run from
# the stapes checkout; every knob is an environment variable, and the README is the
# guide.
set -euo pipefail

CORPUS=${CORPUS:-${XDG_STATE_HOME:-$HOME/.local/state}/stapes/corpus}   # where the loops go
LOOPS=${LOOPS:-20000}                # loops to render; a smoke test is 200
PARTS=${PARTS:-1}                    # exports at once, each rendering every PARTS-th loop: a core each
BARS=${BARS:-4}                      # bars a loop
AUG=${AUG:-aug}                      # aug: a room, a mix, random voices, saturation, a tuning off A440; empty for clean
GEN=${GEN:-gen}                      # gen: beats, melodies and bass lines the bank never had; empty for the bank alone
GLIDE=${GLIDE:-}                     # glide: a share of the notes bend, scoop or sway, each labelled with its cents; empty for none
WAV=${WAV:-nowav}                    # nowav: frames and labels only; empty to keep each loop's WAV
CODEC=${CODEC:-}                     # codec: a third of the loops through a lossy codec and back, by ffmpeg; empty for none
TEMPOS=${TEMPOS:-}                   # e.g. 80-120 or 96,128,160; empty for the default spread
SAMPLED=${SAMPLED:-}                 # a directory of NSynth notes half the tonal lines play on; empty for the engine's voices
FOCUS=${FOCUS:-}                     # a .slab run before corpus: voices and patterns to draw from
STEPS=${STEPS:-20000}                # training steps
BATCH=${BATCH:-8}                    # windows a step
DIM=${DIM:-256}                      # the model's width, attention heads, layers and feed-forward width
HEADS=${HEADS:-8}
LAYERS=${LAYERS:-4}
FFN=${FFN:-1024}
CONVS=${CONVS:-}                     # convolutions before the input layer, e.g. 16x3x3/1,16x3x3/2,32x3x3/2; empty for none
TOWERS=${TOWERS:-}                   # stacks of their own, layers:heads each, e.g. 2:2,4:10; empty for one of LAYERS
MIX=${MIX:-}                         # another corpus and the share of windows drawn from it, dir:share; empty for none
OVER=${OVER:-}                       # a corpus of drums alone and the share of windows that lay it over a loop with none, dir:share; empty for none
CHANNEL=${CHANNEL:-}                 # a noise bank and the share of windows heard through a small speaker, a room and a phone, file.npz:share; empty for none
PCEN=${PCEN:-}                       # pcen: a second input channel, each band's PCEN; empty for none
HOLDOUT=${HOLDOUT:-200}              # loops held out, scored every EVERY steps
EVERY=${EVERY:-1000}
VALIDATE=${VALIDATE:-}               # a corpus of recordings (corpus <dir> from) whose best step OUT gets; empty for the held-out loops'
OUT=${OUT:-$CORPUS/ripframe.bin}       # the weight file
CHECKPOINT=${CHECKPOINT:-$CORPUS/ripframe.pt}
ROOT=${ROOT:-${XDG_CONFIG_HOME:-$HOME/.config}/stapes}   # the stapes root rip hears with
NOTES=${NOTES:-}                     # a notes model's weight file, a --pitchnet run's OUT, bench and install put beside OUT as ripnotes.bin; empty for none
BENCH_LOOPS=${BENCH_LOOPS:-12}
BENCH_BARS=${BENCH_BARS:-4}
INIT=${INIT:-}                       # shard: the .pt the round starts from, the last merge's
SHARD=${SHARD:-0}                    # shard: this worker's stream of windows, its own in every round
MERGE_FILES=${MERGE_FILES:-}         # merge: the round's shard checkpoints, blank-separated, globs too
TRIM=${TRIM:-0}                      # merge: share of the files dropped at each end of each weight
MERGED=${MERGED:-$CORPUS/merged.pt}  # merge: the mean as a .pt, the next round's INIT; OUT gets its weights
PYTHON=${PYTHON:-python3}

here=$(cd "$(dirname "$0")" && pwd)
stage=${1:-all}

stapes() { go run ./cmd/stapes -headless "$@"; }

# corpus_slab writes the export's script to $1, its corpus line ending in $2
corpus_slab() {
	{
		[ -n "$FOCUS" ] && printf 'run %s\n' "$(realpath "$FOCUS")"
		printf 'corpus %s %s %s %s %s %s %s %s %s %s %s\n' "$(realpath "$CORPUS")" "$LOOPS" "$BARS" "$AUG" "$GEN" "$GLIDE" "$WAV" "$CODEC" "${TEMPOS:+tempos $TEMPOS}" "${SAMPLED:+sampled $(realpath "$SAMPLED")}" "${2:-}"
	} > "$1"
}

# The parts render at once; the last export, alone, renders any loop they left
# and writes corpus.json. Every export keeps the loops on disk, so a stopped
# corpus run again goes on where it stopped.
corpus() {
	mkdir -p "$CORPUS"
	local k
	corpus_slab "$CORPUS/corpus.slab"
	echo "== corpus: $(cat "$CORPUS/corpus.slab"), in $PARTS part(s)"
	if [ "$PARTS" -gt 1 ]; then
		for ((k = 0; k < PARTS; k++)); do
			corpus_slab "$CORPUS/corpus$k.slab" "part $k/$PARTS"
			stapes -s "$CORPUS/corpus$k.slab" &
		done
		wait
	fi
	stapes -s "$CORPUS/corpus.slab"
}

train() {
	echo "== train: $STEPS steps of batch $BATCH on $CORPUS, $HOLDOUT loops held out${VALIDATE:+, the step picked on $VALIDATE}, width $DIM, $HEADS heads, $LAYERS layers, feed-forward $FFN"
	(cd "$here" && "$PYTHON" train.py --corpus "$(realpath "$CORPUS")" --out "$(realpath -m "$OUT")" --steps "$STEPS" --batch "$BATCH" --every "$EVERY" --holdout "$HOLDOUT" --checkpoint "$(realpath -m "$CHECKPOINT")" \
		--dim "$DIM" --heads "$HEADS" --layers "$LAYERS" --ffn "$FFN" --convs "$CONVS" --towers "$TOWERS" ${MIX:+--mix "$MIX"} ${OVER:+--over "$OVER"} ${CHANNEL:+--channel "$CHANNEL"} ${PCEN:+--pcen} ${VALIDATE:+--validate "$(realpath "$VALIDATE")"} "$@")
}

# A shard's files are named by SHARD, so a round never resumes an earlier one's checkpoint.
shard() {
	[ -f "$INIT" ] || { echo "shard: INIT names the .pt the round starts from; '$INIT' is no file" >&2; exit 2; }
	echo "== shard $SHARD from $INIT"
	OUT=$CORPUS/shard$SHARD.bin CHECKPOINT=$CORPUS/shard$SHARD.pt train --init "$(realpath "$INIT")" --shard-seed "$SHARD"
}

merge() {
	[ -n "$MERGE_FILES" ] || { echo "merge: MERGE_FILES names the round's shard checkpoints" >&2; exit 2; }
	echo "== merge: $MERGE_FILES into $MERGED and $OUT"
	# shellcheck disable=SC2086 # blank-separated, and a glob expands
	"$PYTHON" "$here/average.py" --pt "$MERGED" --out "$OUT" --trim "$TRIM" $MERGE_FILES
}

# The bench's own root holds the weights and ROOT's rip.toml, when it has one, the
# thresholds rip hears with there
bench() {
	local root
	root=$(mktemp -d)
	cp "$OUT" "$root/ripframe.bin"
	if [ -n "$NOTES" ]; then cp "$NOTES" "$root/ripnotes.bin"; fi
	if [ -f "$ROOT/rip.toml" ]; then cp "$ROOT/rip.toml" "$root/rip.toml"; fi
	local slab="$root/bench.slab"
	{
		[ -n "$FOCUS" ] && printf 'run %s\n' "$(realpath "$FOCUS")"
		printf 'bench %s %s\n' "$BENCH_LOOPS" "$BENCH_BARS"
	} > "$slab"
	echo "== bench: the analysis and $OUT${NOTES:+ with the notes of $NOTES} on $BENCH_LOOPS fresh loops, clean and augmented"
	stapes -config-dir "$root" -s "$slab"
	rm -rf "$root"
}

install() {
	mkdir -p "$ROOT"
	cp "$OUT" "$ROOT/ripframe.bin"
	echo "== installed $OUT as $ROOT/ripframe.bin; stapes' rip hears with it, remove the file to hear with the analysis"
	if [ -n "$NOTES" ]; then
		cp "$NOTES" "$ROOT/ripnotes.bin"
		echo "== installed $NOTES as $ROOT/ripnotes.bin; it gives the notes, remove the file to take them from the frame model"
	elif [ -f "$ROOT/ripnotes.bin" ]; then
		echo "== $ROOT/ripnotes.bin still gives the notes; remove it to take them from the frame model"
	fi
}

case "$stage" in
corpus | train | shard | merge | bench | install) "$stage" ;;
all) corpus; train; bench; install ;;
*) echo "usage: $0 corpus|train|shard|merge|bench|install|all" >&2; exit 2 ;;
esac
