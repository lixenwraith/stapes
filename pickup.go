package stapes

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"slices"

	"github.com/lixenwraith/audio"
)

// readChannels is a WAV's channels apart, at its rate: audio reads the samples
// as one channel's once the header says one, and they are dealt out by its count
func readChannels(file string) ([][]float64, int, error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, 0, err
	}
	n := 0
	for off := 12; off+8 <= len(data); {
		size := int(binary.LittleEndian.Uint32(data[off+4:]))
		if string(data[off:off+4]) == "fmt " && off+12 <= len(data) {
			n = int(binary.LittleEndian.Uint16(data[off+10:]))
			binary.LittleEndian.PutUint16(data[off+10:], 1)
			break
		}
		off += 8 + size + size&1
	}
	if n < 1 {
		return nil, 0, fmt.Errorf("%s: no channels", file)
	}
	x, rate, err := audio.ReadWAV(bytes.NewReader(data))
	if err != nil {
		return nil, 0, fmt.Errorf("%s: %w", file, err)
	}
	out := make([][]float64, n)
	for c := range out {
		out[c] = make([]float64, len(x)/n)
		for i := range out[c] {
			out[c][i] = x[i*n+c]
		}
	}
	return out, rate, nil
}

// stringsOf is the string each note is played on, the place of the track whose
// notes hold it: one of its key starting within a frame of it, the nearest, two
// notes alike taking two strings; -1 for a note no string holds
func stringsOf(tracks []pitchTrack, notes []audio.RipNote) []int {
	of, taken := make([]int, len(notes)), map[[2]int]bool{}
	for i, n := range notes {
		of[i] = -1
		best, at := math.Inf(1), [2]int{}
		for k, tr := range tracks {
			for m, a := range tr.notes {
				if d := math.Abs(a[0] - n.Start); d <= audio.RipFrameSec && d < best && math.Floor(a[2]+0.5) == float64(n.MIDI) && !taken[[2]int{k, m}] {
					best, at, of[i] = d, [2]int{k, m}, k
				}
			}
		}
		if of[i] >= 0 {
			taken[at] = true
		}
	}
	return of
}

// heardString is a note as its strings say it: its string, the sample its
// string stops sounding it at, and its level on the microphone where it sounds
// alone by the ends its strings give (lo 0 otherwise)
type heardString struct {
	string, end int
	mic         heardLevel
}

// hearStrings is each note's end, read on the channel of ch its string in of
// has, ch a channel a string in the order the pitch track names them, and its
// level on the microphone where it sounds alone by those ends, which say that
// nothing else rings into it; or why the channels say nothing, of unmeasuredWhy
func hearStrings(mic []float64, ch [][]float64, rate int, notes []heardNote, of []int) ([]heardString, string) {
	strs := slices.Max(of) + 1
	if strs < 1 {
		return nil, unmeasuredWhy[3]
	} else if strs > len(ch) {
		return nil, unmeasuredWhy[4]
	}
	out := make([]heardString, len(notes))
	for i, n := range notes {
		out[i] = heardString{end: n.end, string: of[i]}
	}
	for s := range strs {
		var on []int
		for i := range notes {
			if of[i] == s {
				on = append(on, i)
			}
		}
		slices.SortStableFunc(on, func(a, b int) int { return cmp.Compare(notes[a].start, notes[b].start) })
		x, line, next := ch[s], make([]heardNote, len(on)), make([]int, len(on))
		for k, i := range on {
			line[k], next[k] = notes[i], len(x)
			if k+1 < len(on) {
				next[k] = notes[on[k+1]].start
			}
		}
		for k, e := range heardEnds(x, rate, line, next) {
			out[on[k]].end = e
		}
	}
	for i, n := range notes {
		n.end = out[i].end
		span, with := heldSpan(n, rate, len(mic)), false
		for j := range notes {
			with = with || j != i && notes[j].start < n.start+span && out[j].end > n.start
		}
		if !with && n.start >= 0 && span >= int(sampledWindow*float64(rate)) {
			out[i].mic = hearLevels(mic, rate, []heardNote{n})[0]
		}
	}
	return out, ""
}

// hearPickup is each of a score's notes as its string says it, read from a
// recording's by-string file, a channel a string, beside the recording itself
// at its rate; or why it says nothing (unmeasured)
func hearPickup(wav, hex string, score *audio.MIDIScore, tracks []pitchTrack) ([]heardString, int, *unmeasured) {
	x, rate, failed := readSound(wav)
	if failed != nil {
		return nil, 0, failed
	}
	ch, at, err := readChannels(hex)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, 0, &unmeasured{unmeasuredWhy[0], hex}
	} else if err != nil {
		return nil, 0, &unmeasured{unmeasuredWhy[1], err.Error()}
	}
	if at != rate {
		return nil, 0, &unmeasured{unmeasuredWhy[5], fmt.Sprintf("%s: %d Hz, %s %d", hex, at, wav, rate)}
	}
	heard, why := hearStrings(x, ch, rate, heardNotes(score, tracks, rate), stringsOf(tracks, score.Notes))
	if why != "" {
		return nil, 0, &unmeasured{why, hex}
	}
	for i, h := range heard {
		if !finite(h.mic.lo) || !finite(h.mic.hi) {
			return nil, 0, &unmeasured{unmeasuredWhy[2], fmt.Sprintf("%s: the note at %.3f s", hex, score.Notes[i].Start)}
		}
		heard[i].end = min(h.end, len(x))
	}
	return heard, rate, nil
}
