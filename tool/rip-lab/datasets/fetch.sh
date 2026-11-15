#!/usr/bin/env bash
# fetch.sh nsynth test|valid|train1..train6: NSynth in batches by HTTP byte range, its training split in six slices
# fetch.sh egmd meta|E01..E12|ET: E-GMD's metadata and MIDI, then its audio in batches
# fetch.sh status: what is on disk
# Every command resumes where it stopped: after a drop, run it again. DATA=dir is where the sets land (default:
# the current directory), JOBS=n how many E-GMD files download at once (default 4).
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${DATA:-$PWD}
par=${JOBS:-4}
gcs=https://storage.googleapis.com
nsynth_url=$gcs/download.magenta.tensorflow.org/datasets/nsynth
egmd_url=$gcs/magentadata/datasets/e-gmd/v1.0.0
egmd=$root/e-gmd-v1.0.0
train_total=23815298079
slice=4000000000
nsynth_wav=128044 # every NSynth note: four seconds of 16-bit mono at 16 kHz

declare -A size=(
	[nsynth-test.jsonwav.tar.gz]=349501546
	[nsynth-valid.jsonwav.tar.gz]=1068767009
	[nsynth-train.jsonwav.tar.gz]=$train_total
	[e-gmd-v1.0.0.csv]=9614713
	[e-gmd-v1.0.0-midi.zip]=107076192
)
declare -A md5=(
	[nsynth-test.jsonwav.tar.gz]=5e6f8719bf7e16ad0a00d518b78af77d
	[nsynth-valid.jsonwav.tar.gz]=87e94a00a19b6dbc99cf6d4c0c0cae87
	[nsynth-train.jsonwav.tar.gz]=fde6665a93865503ba598b9fac388660
	[e-gmd-v1.0.0.csv]=a212cb4b1aec205ef1882d9f9bb6150a
	[e-gmd-v1.0.0-midi.zip]=c41050781bd6f4e240efb7a71c924cee
)

die() { echo "fetch.sh: $*" >&2; exit 1; }
bytes() { stat -c %s -- "$1" 2>/dev/null || echo 0; }

# grow FILE with bytes FROM..TO of URL, resuming from what FILE already holds;
# bytes are appended as they arrive, so a drop or a kill keeps them
grow() {
	local url=$1 file=$2 from=$3 to=$4 have code
	while :; do
		have=$(bytes "$file")
		((from + have > to)) && return 0
		code=$(curl -sf --connect-timeout 30 --speed-limit 2048 --speed-time 60 \
			-r "$((from + have))-$to" -w '%{stderr}%{http_code}' -- "$url" 2>&1 >>"$file") || true
		case $code in
		206) (($(bytes "$file") > have)) || sleep 10 ;;
		4??) truncate -s "$have" -- "$file"; echo "$url: HTTP $code" >&2; return 1 ;;
		*) truncate -s "$have" -- "$file"; sleep 10 ;; # no reply, or one that is not the range
		esac
	done
}

# whole NAME from URL into DIR, checked against its MD5 once complete
whole() {
	local url=$1 dir=$2 name=$3
	grow "$url/$name" "$dir/$name" 0 $((size[$name] - 1))
	if [[ ! -e $dir/$name.ok ]]; then
		echo "checking $name"
		[[ $(md5sum -- "$dir/$name" | cut -d' ' -f1) == "${md5[$name]}" ]] ||
			die "$name: MD5 mismatch; delete it and run again"
		: >"$dir/$name.ok"
	fi
}

lock() {
	exec 9>"$root/.fetch-$1.lock"
	flock -n 9 || die "another fetch.sh $1 is running"
}

nsynth() {
	local d=$root/nsynth f
	mkdir -p "$d"
	lock "nsynth-$1"
	case $1 in
	test | valid)
		f=nsynth-$1.jsonwav.tar.gz
		whole "$nsynth_url" "$d" "$f"
		tar -xzf "$d/$f" -C "$d" --skip-old-files
		;;
	train[1-6])
		local k=${1#train} end
		f=nsynth-train.jsonwav.tar.gz
		end=$((k == 6 ? train_total - 1 : k * slice - 1))
		grow "$nsynth_url/$f" "$d/$f" 0 "$end"
		if (($(bytes "$d/$f") == train_total)); then
			whole "$nsynth_url" "$d" "$f"
			tar -xzf "$d/$f" -C "$d" --skip-old-files
		else
			# a gzip stream decodes only from its start; the cut member is removed
			tar -xzf "$d/$f" -C "$d" --skip-old-files 2>/dev/null || true
			find "$d/nsynth-train" -type f -name '*.wav' ! -size "${nsynth_wav}c" -delete
			rm -f -- "$d/nsynth-train/examples.json"
		fi
		;;
	*) die "nsynth: want test, valid or train1 to train6" ;;
	esac
	local split=${1%%[1-6]}
	echo "nsynth $1: $(find "$d/nsynth-$split" -name '*.wav' | wc -l) notes on disk in nsynth-$split"
}

egmd_file() {
	local off=$1 end=$2 crc=$3 usz=$4 dst=$egmd/$5
	(($(bytes "$dst") == usz)) && return 0
	mkdir -p -- "${dst%/*}"
	grow "$egmd_url/e-gmd-v1.0.0.zip" "$dst.part" "$off" "$end" &&
		python3 -I "$here/egmd.py" extract "$dst.part" "$dst" "$crc" "$usz"
}

egmd() {
	mkdir -p "$egmd"
	lock "egmd-$1"
	if [[ $1 == meta ]]; then
		whole "$egmd_url" "$egmd" e-gmd-v1.0.0.csv
		whole "$egmd_url" "$root" e-gmd-v1.0.0-midi.zip
		python3 -I -m zipfile -e "$root/e-gmd-v1.0.0-midi.zip" "$root"
		echo "egmd meta: $(find "$egmd" -name '*.midi' | wc -l) MIDI files"
		return
	fi
	grep -q "^$1	" "$here/egmd/batches.tsv" || die "egmd: no batch $1 (want meta, E01 to E12, ET)"
	[[ -e $egmd/e-gmd-v1.0.0.csv.ok ]] || die "egmd: run 'fetch.sh egmd meta' first"
	local n=0 bad=0 total batch off end crc usz path
	total=$(grep -c "^$1	" "$here/egmd/batches.tsv")
	while IFS=$'\t' read -r batch off end crc usz path; do
		[[ $batch == "$1" ]] || continue
		while (($(jobs -rp | wc -l) >= par)); do wait -n || bad=$((bad + 1)); done
		egmd_file "$off" "$end" "$crc" "$usz" "$path" &
		n=$((n + 1))
		((n % 100 == 0)) && echo "$1: $n/$total"
	done <"$here/egmd/batches.tsv"
	while (($(jobs -rp | wc -l) > 0)); do wait -n || bad=$((bad + 1)); done
	echo "egmd $1: $n files, $bad failed$( ((bad)) && echo '; run again to retry them')"
	((bad == 0))
}

status() {
	local d=$root/nsynth
	for s in test valid train; do
		[[ -d $d/nsynth-$s ]] && echo "nsynth $s: $(find "$d/nsynth-$s" -name '*.wav' | wc -l) notes"
	done
	[[ -e $d/nsynth-train.jsonwav.tar.gz ]] &&
		echo "nsynth train archive: $(($(bytes "$d/nsynth-train.jsonwav.tar.gz") / 1000000)) of $((train_total / 1000000)) MB"
	[[ -d $egmd ]] && python3 -I "$here/egmd.py" status "$root" "$here/egmd/batches.tsv"
	return 0
}

for tool in curl python3 md5sum flock tar truncate; do
	command -v "$tool" >/dev/null || die "needs $tool"
done
case ${1-} in
nsynth) nsynth "${2-}" ;;
egmd) egmd "${2-}" ;;
status) status ;;
*) sed -n '2,6p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
