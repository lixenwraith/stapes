#!/usr/bin/env bash
# The rip's training pipeline, a stage a word and several in a row: corpus, train,
# notes, bench and install make the models; rows, judge, fit, readback, table and
# keep fit them and say where the rip stands; shard and merge spread a round. Run
# from the stapes checkout; every knob is an environment variable, the README the guide.
set -euo pipefail

CORPUS=${CORPUS:-${XDG_STATE_HOME:-$HOME/.local/state}/stapes/corpus}   # where the loops go
LOOPS=${LOOPS:-20000}                # loops to render; a smoke test is 200
PARTS=${PARTS:-1}                    # exports at once, each rendering every PARTS-th loop: a core each
BARS=${BARS:-4}                      # bars a loop
AUG=${AUG-aug}                       # aug: a room, a mix, random voices, saturation, a tuning off A440; empty for clean
GEN=${GEN-gen}                       # gen: music arranged over the whole keyboard in place of the bank's loops; empty for the bank alone
METERS=${METERS-meters}              # meters: the arranged loops in seven meters, some starting part-way into a bar; empty for 4/4 alone
OVERLAPS=${OVERLAPS-overlaps}        # overlaps: a quarter of the loops single notes overlapping in no meter, or overlaps <share>; empty for none
GLIDE=${GLIDE:-}                     # glide: a share of the notes bend, scoop or sway, each labelled with its cents; empty for none
WAV=${WAV-nowav}                     # nowav: frames and labels only; empty to keep each loop's WAV
CODEC=${CODEC:-}                     # codec: a third of the loops through a lossy codec and back, by ffmpeg; empty for none
TEMPOS=${TEMPOS:-}                   # e.g. 80-120 or 96,128,160; empty for the default spread
SAMPLED=${SAMPLED:-}                 # a directory of NSynth notes half the tonal lines play on; empty for the engine's voices
SEED=${SEED:-}                       # every loop drawn from this seed; empty for the verb's own, 1
FOCUS=${FOCUS:-}                     # a .slab run before corpus: voices and patterns to draw from
STAPES=${STAPES:-}                   # a stapes binary every stage runs, such as a corpus's build; empty for the checkout's
STEPS=${STEPS:-20000}                # training steps
BATCH=${BATCH:-8}                    # windows a step
DIM=${DIM:-256}                      # the model's width, attention heads, layers and feed-forward width
HEADS=${HEADS:-8}
LAYERS=${LAYERS:-4}
FFN=${FFN:-1024}
CONVS=${CONVS:-}                     # convolutions before the input layer, e.g. 16x3x3/1,16x3x3/2,32x3x3/2; empty for none
TOWERS=${TOWERS:-}                   # stacks of their own, layers:heads each, e.g. 2:2,4:11; empty for one of LAYERS
MIX=${MIX:-}                         # another corpus and the share of windows drawn from it, dir:share; empty for none
OVER=${OVER:-}                       # a corpus of drums alone and the share of windows that lay it over a loop where nothing is struck, dir:share; empty for none
CHANNEL=${CHANNEL:-}                 # a noise bank and the share of windows heard through a small speaker, a room and a phone, file.npz:share; empty for none
PCEN=${PCEN:-}                       # pcen: two more input planes, each band's PCEN and their mean, fine and sharp; empty for none
HOLDOUT=${HOLDOUT:-200}              # loops held out, scored every EVERY steps
EVERY=${EVERY:-1000}
VALIDATE=${VALIDATE:-}               # a corpus of recordings (corpus <dir> from) whose best step OUT gets; empty for the held-out loops'
OUT=${OUT:-$CORPUS/ripframe.bin}       # the weight file
CHECKPOINT=${CHECKPOINT:-$CORPUS/ripframe.pt}
ROOT=${ROOT:-${XDG_CONFIG_HOME:-$HOME/.config}/stapes}   # the stapes root rip hears with
NOTES=${NOTES:-}                     # a notes model's weight file: notes trains it, and the stages after put it beside OUT as ripnotes.bin; empty for none
DILATIONS=${DILATIONS:-}             # notes: its long convolutions' frames apart, e.g. 3,9,27; empty for none
LR=${LR:-}                           # the learning rate; empty for train.py's own
EXTRA=${EXTRA:-}                     # more corpora and the share of windows each draws, dir:share by commas, such as recorded sets; empty for none
CARRY=${CARRY:-}                     # carry: a pass that carries on from INIT with its optimizer's state; empty for a fresh run
FIT=${FIT:-$CORPUS/fit}              # where a pair's fit goes: its root, rows, sets, logs and table; a new pair takes a new folder
ROWS=${ROWS:-}                       # rows: the truth lists and corpus directories the judge is fitted on, blank-separated
LEAVE_OUT=${LEAVE_OUT:-}             # judge: conditions its fit leaves out and counts, blank-separated
TRUTH=${TRUTH:-}                     # fit: the truth list the thresholds are searched on, a kind of sound a condition, or a corpus directory
READBACK=${READBACK:-}               # readback: the truth list every set is read on, of recordings no fit saw, or a corpus directory
FRAMES=${FRAMES:-$FIT/frames}        # what the models heard of each recording, kept between runs
START=${START:-}                     # fit: the rip.toml every search starts from; empty for the built-in thresholds
POOL=${POOL:-12}                     # fit: the sets a generation
GENERATIONS=${GENERATIONS:-5}        # fit: the generations a search runs
GUARD=${GUARD:-}                     # fit: the score a set may take from a condition; empty for the verb's own
HOLD=${HOLD:-}                       # fit, readback: the GB of recordings held between rips; empty for the verb's own
WEIGH=${WEIGH-ends}                  # fit: the readings a search weighs beside the notes, by commas; empty for the notes alone
GENES=${GENES:-}                     # fit: the genes a search moves, by commas; empty for all the list can score
JUDGE=${JUDGE:-}                     # install: a judge's file, put beside the notes model as ripjudge.bin; empty for none
SET=${SET:-}                         # install: a rip.toml, one of FIT's sets, put in the root; empty for none
WEIGHTS=${WEIGHTS:-}                 # keep: the folder a round's files are copied under, each a line in its MANIFEST.txt
NAME=${NAME:-$(basename "$CORPUS")}  # keep: the round's folder under WEIGHTS
BENCH_LOOPS=${BENCH_LOOPS:-12}
BENCH_BARS=${BENCH_BARS:-4}
INIT=${INIT:-}                       # the .pt a training starts from: a shard's round, the last merge's, or a pass's, the pass before it
SHARD=${SHARD:-0}                    # shard: this worker's stream of windows, its own in every round
MERGE_FILES=${MERGE_FILES:-}         # merge: the round's shard checkpoints, blank-separated, globs too
TRIM=${TRIM:-0}                      # merge: share of the files dropped at each end of each weight
MERGED=${MERGED:-$CORPUS/merged.pt}  # merge: the mean as a .pt, the next round's INIT; OUT gets its weights
PYTHON=${PYTHON:-python3}

here=$(cd "$(dirname "$0")" && pwd)

stapes() { if [ -n "$STAPES" ]; then "$STAPES" -headless "$@"; else go run ./cmd/stapes -headless "$@"; fi; }

# scratch makes the run's temporary folder once; it goes when the run ends
scratch() {
	if [ -z "${tmp:-}" ]; then
		tmp=$(mktemp -d)
		trap 'rm -rf -- "$tmp"' EXIT
	fi
}

# corpus_slab is the export's script: its corpus line is the words the knobs set, a
# blank apart, ending in $1, so a line written by hand is written again exactly
corpus_slab() {
	local w words=()
	[ -z "$FOCUS" ] || printf 'run %s\n' "$(realpath -m "$FOCUS")"
	for w in "$(realpath -m "$CORPUS")" "$LOOPS" "$BARS" "$AUG" "$GEN" "$METERS" "$OVERLAPS" "$GLIDE" "$WAV" "$CODEC" \
		"${TEMPOS:+tempos $TEMPOS}" "${SAMPLED:+sampled $(realpath -m "$SAMPLED")}" "${SEED:+seed $SEED}" "${1:-}"; do
		[ -z "$w" ] || words+=("$w")
	done
	printf 'corpus %s\n' "${words[*]}"
}

# The parts render at once, a core each; the last export, alone, renders any loop
# they left and writes corpus.json. Each runs on an empty root and config folder, so
# no model loads and no override of the bank is read, and the checkout is built
# first, since go reads its own settings from the config folder. Every export keeps
# the loops on disk, so a stopped corpus run again goes on where it stopped.
corpus() {
	local k STAPES=$STAPES
	mkdir -p "$CORPUS"
	corpus_slab > "$CORPUS/corpus.slab"
	echo "== corpus: $(cat "$CORPUS/corpus.slab"), in $PARTS part(s)${STAPES:+ by $STAPES}"
	scratch
	bare=$tmp/corpus
	mkdir -p "$bare/root" "$bare/config"
	if [ -z "$STAPES" ]; then
		go build -o "$bare/stapes" ./cmd/stapes
		STAPES=$bare/stapes
	fi
	if [ "$PARTS" -gt 1 ]; then
		for ((k = 0; k < PARTS; k++)); do
			corpus_slab "part $k/$PARTS" > "$CORPUS/corpus$k.slab"
			GOMAXPROCS=1 XDG_CONFIG_HOME=$bare/config stapes -config-dir "$bare/root" -s "$CORPUS/corpus$k.slab" &
		done
		wait
	fi
	XDG_CONFIG_HOME=$bare/config stapes -config-dir "$bare/root" -s "$CORPUS/corpus.slab"
}

train() {
	echo "== train: $STEPS steps of batch $BATCH on $CORPUS, $HOLDOUT loops held out${VALIDATE:+, the step picked on $VALIDATE}, width $DIM, $HEADS heads, $LAYERS layers, feed-forward $FFN"
	(cd "$here" && "$PYTHON" train.py --corpus "$(realpath "$CORPUS")" --out "$(realpath -m "$OUT")" --steps "$STEPS" --batch "$BATCH" --every "$EVERY" --holdout "$HOLDOUT" --checkpoint "$(realpath -m "$CHECKPOINT")" \
		--dim "$DIM" --heads "$HEADS" --layers "$LAYERS" --ffn "$FFN" --convs "$CONVS" --towers "$TOWERS" ${MIX:+--mix "$MIX"} ${OVER:+--over "$OVER"} ${CHANNEL:+--channel "$CHANNEL"} ${PCEN:+--pcen} ${VALIDATE:+--validate "$(realpath "$VALIDATE")"} \
		${LR:+--lr "$LR"} ${EXTRA:+--extra "$EXTRA"} ${INIT:+--init "$(realpath "$INIT")"} ${CARRY:+--carry} "$@")
}

# A shard's files are named by SHARD, so a round never resumes an earlier one's checkpoint.
shard() {
	[ -f "$INIT" ] || { echo "shard: INIT names the .pt the round starts from; '$INIT' is no file" >&2; exit 2; }
	echo "== shard $SHARD from $INIT"
	OUT=$CORPUS/shard$SHARD.bin CHECKPOINT=$CORPUS/shard$SHARD.pt train --shard-seed "$SHARD"
}

# The notes model trains as the frame model does, by the same knobs, into NOTES
notes() {
	[ -n "$NOTES" ] || { echo "notes: NOTES names the notes model's weight file" >&2; exit 2; }
	echo "== notes: the notes model, into $NOTES"
	OUT=$NOTES CHECKPOINT=${NOTES%.bin}.pt train --pitchnet ${DILATIONS:+--dilations "$DILATIONS"}
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
	if [ -n "$JUDGE" ]; then
		cp "$JUDGE" "$ROOT/ripjudge.bin"
		echo "== installed $JUDGE as $ROOT/ripjudge.bin; it rules which strikes are notes, remove the file for the installed scores"
	fi
	if [ -n "$SET" ]; then
		cp "$SET" "$ROOT/rip.toml"
		echo "== installed $SET as $ROOT/rip.toml; rip hears with its thresholds, remove the file for the built-in ones"
	fi
}

# The fit's root holds the pair, then the judge fitted on it. A script reads its
# paths from its own folder, so the fit's are made whole; the stages that run
# stapes many times build the checkout once
fitroot() {
	FIT=$(realpath -m "$FIT") FRAMES=$(realpath -m "$FRAMES")
	mkdir -p "$FIT/root" "$FIT/logs" "$FIT/lists"
	cmp -s "$OUT" "$FIT/root/ripframe.bin" || cp "$OUT" "$FIT/root/ripframe.bin"
	[ -z "$NOTES" ] || cmp -s "$NOTES" "$FIT/root/ripnotes.bin" || cp "$NOTES" "$FIT/root/ripnotes.bin"
	if [ -z "$STAPES" ]; then
		scratch
		go build -o "$tmp/stapes" ./cmd/stapes
		STAPES=$tmp/stapes
	fi
}

# slab runs stapes from root $2 on the lines after it; its words are
# $FIT/logs/$1.log once it has ended, so a log there is a run that ended
slab() {
	local name=$1 root=$2
	shift 2
	printf '%s\n' "$@" > "$FIT/logs/$name.slab"
	stapes -config-dir "$root" -s "$FIT/logs/$name.slab" < /dev/null | tee "$FIT/logs/$name.part"
	mv "$FIT/logs/$name.part" "$FIT/logs/$name.log"
}

# listed prints the truth list $1 names: a file's lines, or a corpus directory's
# loops that kept their audio, each against its label under the folder's name, a
# loop under four seconds left out, as rip refuses one
listed() {
	local w rate
	if [ ! -d "$1" ]; then
		cat "$1"
		return
	fi
	for w in "$(realpath "$1")"/loop_*.wav; do
		[ -e "$w" ] || { echo "$1 holds no loop's audio: render it with WAV= empty" >&2; exit 2; }
		rate=$(od -An -tu4 -j28 -N4 "$w")
		[ "$(wc -c < "$w")" -lt $((rate * 4 + 44)) ] || printf '%s\t%s\t%s\ttimed\n' "$w" "${w%.wav}.json" "$(basename "$1")"
	done
}

rows() {
	[ -n "$ROWS" ] || { echo "rows: ROWS names the truth lists and corpus directories the judge is fitted on" >&2; exit 2; }
	fitroot
	local from name
	for from in $ROWS; do
		name=$(basename "$from")
		name=${name%.*}
		echo "== rows: $from through the pair, into $FIT/rows/$name"
		slab "rows-$name" "$FIT/root" "rows $FIT/rows/$name from $(realpath "$from") frames $FRAMES"
	done
}

# A judge is its notes model's: one that stands in the root is kept
judge() {
	local dirs=("$FIT"/rows/*/)
	[ -d "${dirs[0]}" ] || { echo "judge: no rows under $FIT/rows; rows writes them" >&2; exit 2; }
	if [ -s "$FIT/root/ripjudge.bin" ]; then
		echo "== judge: $FIT/root/ripjudge.bin stands; remove it to fit again"
		return
	fi
	echo "== judge: fitted on ${dirs[*]}"
	# shellcheck disable=SC2086 # blank-separated
	"$PYTHON" "$here/judge.py" --rows "${dirs[@]}" ${LEAVE_OUT:+--leave-out $LEAVE_OUT} --out "$FIT/root/ripjudge.part" | tee "$FIT/logs/judge.log"
	mv "$FIT/root/ripjudge.part" "$FIT/root/ripjudge.bin"
}

# Each kind of sound is searched apart, from START, in a copy of the root, since
# tune writes the set it finds there: the set a search leaves is the kind's,
# the start's own where nothing beat it. A set stands while what it was searched
# from does: the root, the start, the kind's lines and the search's words
fit() {
	[ -e "$TRUTH" ] || { echo "fit: TRUTH names the truth list or corpus directory the thresholds are searched on; '$TRUTH' is neither" >&2; exit 2; }
	fitroot
	mkdir -p "$FIT/sets"
	listed "$TRUTH" > "$FIT/lists/fit.tsv"
	local kind name from line="tune $POOL $GENERATIONS truth LIST${GUARD:+ guard $GUARD} frames $FRAMES${HOLD:+ hold $HOLD}${WEIGH:+ weigh $WEIGH}${GENES:+ genes $GENES}"
	while IFS= read -r kind; do
		name=${kind//[^A-Za-z0-9]/-}
		name=${name:-all}
		awk -F'\t' -v k="$kind" '$3 == k' "$FIT/lists/fit.tsv" > "$FIT/lists/fit-$name.tsv"
		rm -rf "$FIT/search"
		cp -r "$FIT/root" "$FIT/search"
		[ -z "$START" ] || cp "$START" "$FIT/search/rip.toml"
		from=$(cat "$FIT/search"/* "$FIT/lists/fit-$name.tsv" | sha256sum)$line
		if [ -e "$FIT/sets/$name.toml" ] && [ "$from" = "$(cat "$FIT/sets/$name.from" 2> /dev/null)" ]; then
			continue
		fi
		echo "== fit: the thresholds for ${kind:-every recording}, on $(wc -l < "$FIT/lists/fit-$name.tsv") recording(s)${START:+, from $START}"
		slab "fit-$name" "$FIT/search" "${line/LIST/$FIT/lists/fit-$name.tsv}"
		[ -f "$FIT/search/rip.toml" ] || : > "$FIT/search/rip.toml"
		mv "$FIT/search/rip.toml" "$FIT/sets/$name.toml"
		echo "$from" > "$FIT/sets/$name.from"
	done < <(cut -f3 "$FIT/lists/fit.tsv" | sort -u)
	rm -rf "$FIT/search"
}

# reading reads set $2 on the read-back's list as $1, under the judge unless
# $3 is alone: tune's smallest search, whose figures before are the set's own.
# A reading stands while the root it read from and the list do
reading() {
	local from
	rm -rf "$FIT/read"
	cp -r "$FIT/root" "$FIT/read"
	[ "${3:-}" != alone ] || rm -f "$FIT/read/ripjudge.bin"
	[ -z "$2" ] || cp "$2" "$FIT/read/rip.toml"
	from=$(cat "$FIT/read"/* "$FIT/lists/readback.tsv" | sha256sum)
	if [ ! -s "$FIT/logs/readback-$1.log" ] || [ "$from" != "$(cat "$FIT/logs/readback-$1.from" 2> /dev/null)" ]; then
		echo "== readback: $1"
		slab "readback-$1" "$FIT/read" "tune 2 1 truth $FIT/lists/readback.tsv frames $FRAMES${HOLD:+ hold $HOLD}"
		echo "$from" > "$FIT/logs/readback-$1.from"
	fi
	rm -rf "$FIT/read"
}

# The read-back's rows: the models alone, under their judge, under START, and
# under each kind's set that is not the start's own; a row whose set is gone
# leaves the logs, so the table holds what stands
readback() {
	[ -e "$READBACK" ] || { echo "readback: READBACK names the truth list or corpus directory every set is read on; '$READBACK' is neither" >&2; exit 2; }
	fitroot
	listed "$READBACK" > "$FIT/lists/readback.tsv"
	local set rows=" models "
	reading models "" alone
	if [ -s "$FIT/root/ripjudge.bin" ]; then
		reading judge ""
		rows+="judge "
	fi
	if [ -n "$START" ]; then
		reading start "$START"
		rows+="start "
	fi
	for set in "$FIT"/sets/*.toml; do
		if [ -s "$set" ] && ! cmp -s "$set" "${START:-/dev/null}"; then
			reading "fit-$(basename "$set" .toml)" "$set"
			rows+="fit-$(basename "$set" .toml) "
		fi
	done
	for set in "$FIT"/logs/readback-*.log; do
		set=$(basename "$set" .log)
		[[ $rows == *" ${set#readback-} "* ]] || rm -f "$FIT/logs/$set".*
	done
}

table() {
	local set
	"$PYTHON" "$here/table.py" "$FIT"/logs/readback-*.log > "$FIT/table.txt"
	for set in "$FIT"/sets/*.toml; do
		if [ -e "$set" ] && { [ ! -s "$set" ] || cmp -s "$set" "${START:-/dev/null}"; }; then
			echo "fit-$(basename "$set" .toml): no set beat the one its search started from" >> "$FIT/table.txt"
		fi
	done
	cat "$FIT/table.txt"
}

# keep copies what a round made under WEIGHTS, a file not there yet or changed, each
# with its line in MANIFEST.txt: its path, bytes, sha256, when, what it is
keep() {
	[ -n "$WEIGHTS" ] || { echo "keep: WEIGHTS names the folder the round's files are kept under" >&2; exit 2; }
	local file as what
	mkdir -p "$WEIGHTS/$NAME/sets"
	[ -s "$WEIGHTS/MANIFEST.txt" ] || printf '# one line a file, tab-separated: its path here, bytes, sha256, when copied (UTC), what it is\n' > "$WEIGHTS/MANIFEST.txt"
	while IFS=$'\t' read -r file as what; do
		if [ ! -s "$file" ] || cmp -s "$file" "$WEIGHTS/$NAME/$as"; then
			continue
		fi
		cp -p "$file" "$WEIGHTS/$NAME/$as"
		printf '%s\t%s\t%s\t%s\t%s\n' "$NAME/$as" "$(wc -c < "$file")" "$(sha256sum "$file" | cut -d' ' -f1)" "$(date -u '+%F %T UTC')" "$what" >> "$WEIGHTS/MANIFEST.txt"
		echo "== kept $file as $WEIGHTS/$NAME/$as"
	done < <(
		printf '%s\t%s\t%s\n' "$OUT" ripframe.bin "the frame model" \
			"$CHECKPOINT" ripframe.pt "the frame model's checkpoint, the step a later pass starts from" \
			"$NOTES" ripnotes.bin "the notes model" \
			"${NOTES%.bin}.pt" ripnotes.pt "the notes model's checkpoint, the step a later pass starts from" \
			"$FIT/root/ripjudge.bin" ripjudge.bin "the judge fitted on this notes model's rows" \
			"$FIT/table.txt" table.txt "the read-back's table: where the rip stands under each set" \
			"$CORPUS/round.log" round.log "the round's log"
		for file in "$FIT"/sets/*.toml; do
			printf '%s\tsets/%s\tthe thresholds fitted for %s\n' "$file" "$(basename "$file")" "$(basename "$file" .toml)"
		done
	)
}

[ $# -gt 0 ] || set -- all
for stage in "$@"; do
	case "$stage" in
	corpus | train | notes | shard | merge | bench | install | rows | judge | fit | readback | table | keep) "$stage" ;;
	line) corpus_slab ;;
	all) corpus; train; bench; install ;;
	*) echo "usage: $0 corpus|line|train|notes|shard|merge|bench|install|rows|judge|fit|readback|table|keep|all ..." >&2; exit 2 ;;
	esac
done
