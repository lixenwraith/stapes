package stapes

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/lixenwraith/audio"
)

// rowsManifest is a rows directory's rows.json: the row's version and size and
// the checksum of the notes weights the rows were read through, which a judge
// fitted on them names; a directory holds one kind
type rowsManifest struct {
	Version    int    `json:"version"`
	RowVersion int    `json:"row_version"`
	RowSize    int    `json:"row_size"`
	Notes      string `json:"notes"`
}

// rowsRecording is a recording's label beside its rows: its condition, its
// truth's count, how many of those notes no strike answers, and a strike a row
type rowsRecording struct {
	Source     string      `json:"source"`
	Condition  string      `json:"condition"`
	Given      int         `json:"given"`
	Unanswered int         `json:"unanswered"`
	Strikes    []rowStrike `json:"strikes"`
}

// rowStrike is a row's strike: its start in the rip's frames, its pitch, its
// stage, whether a truth note answers it, and the installed verdict
type rowStrike struct {
	Frame    int  `json:"frame"`
	MIDI     int  `json:"midi"`
	Reported bool `json:"reported"`       // a note the decoder reported, else a strike it dropped
	Held     bool `json:"held,omitempty"` // a held note the accounting keeps unjudged
	Right    bool `json:"right"`
	Kept     bool `json:"kept"`
}

const rowsFile = "rows.json"

func cmdRows(s *Session, a []string) error {
	usage := errors.New("want rows <dir> from <list|corpus dir> [part k/n] [frames <dir>]")
	if a[1] != "from" {
		return usage
	}
	var o tuneOptions
	for w := a[3:]; len(w) > 0; w = w[2:] {
		switch {
		case len(w) > 1 && w[0] == "part":
			var err error
			if o.part, o.parts, err = parsePart(w[1]); err != nil {
				return err
			}
		case len(w) > 1 && w[0] == "frames":
			o.frames = s.path(w[1])
		default:
			return usage
		}
	}
	if s.frontEnd == nil {
		return fmt.Errorf("rows reads a strike's row off the models' heads, and no %s is loaded", frameFile)
	}
	return s.rows(s.path(a[0]), s.path(a[2]), o, s.frontEnd, s.frontEnd.NotesSum)
}

// rows rips each recording from names, a truth list's or a corpus directory's
// loops that kept their audio, through fe with no judge, and writes every
// strike's row into dir with its label (rightStrikes): <checksum>.rows as a
// loop's frames are written, then <checksum>.json. A recording on disk is kept.
func (s *Session) rows(dir, from string, o tuneOptions, fe audio.RipFrontEnd, notes uint64) error {
	lines, silent, err := truthSources(from)
	if err != nil {
		return err
	}
	for _, d := range []string{dir, o.frames} {
		if d != "" {
			if err := os.MkdirAll(d, 0o755); err != nil {
				return err
			}
		}
	}
	m := rowsManifest{Version: 1, RowVersion: audio.RipRowVersion, RowSize: audio.RipRowSize, Notes: fmt.Sprintf("%016x", notes)}
	manifest := filepath.Join(dir, rowsFile)
	if data, err := os.ReadFile(manifest); err == nil {
		var was rowsManifest
		if json.Unmarshal(data, &was) != nil || was != m {
			return fmt.Errorf("%s holds rows of version %d read through the notes weights %s, and these are of version %d through %s: a directory holds one kind", dir, was.RowVersion, was.Notes, m.RowVersion, m.Notes)
		}
	} else {
		// whole or not there, as parts started together each write it
		part := fmt.Sprintf("%s.%d.part", manifest, os.Getpid())
		if err := writeJSON(part, m); err != nil {
			return err
		}
		if err := os.Rename(part, manifest); err != nil {
			return err
		}
	}
	var written, kept, read, strikes, right, given, unanswered int
	var failed []string
	for i, l := range lines {
		if o.parts > 0 && i%o.parts != o.part {
			continue
		}
		recording, err := os.ReadFile(l.at(l.wav))
		if err != nil {
			return err
		}
		sum := sha256.Sum256(recording)
		base := filepath.Join(dir, fmt.Sprintf("%x", sum[:8]))
		if _, err := os.Stat(base + ".json"); err == nil { // written last, so its rows are whole
			kept++
			continue
		}
		t, err := l.take(recording, sum, fe, o.frames, s.earSum)
		if err != nil {
			return err
		}
		r, err := audio.Ripper{Params: s.rip, FrontEnd: t.ear, Lead: s.lead, Strikes: true, Rows: true}.Samples(t.x, audio.RipRate)
		if err != nil {
			failed = append(failed, fmt.Sprintf("%s: %v", l.wav, err))
			continue
		}
		truth, err := t.truth(r)
		if err != nil {
			return fmt.Errorf("%s: %w", l.truth, err)
		}
		if slices.ContainsFunc(truth, func(n audio.RipNote) bool { return n.MIDI < 0 }) {
			return fmt.Errorf("%s: a rhythm gives no pitch to label a strike by", l.truth)
		}
		rec := rowsRecording{Source: l.wav, Condition: l.cond, Given: len(truth), Strikes: make([]rowStrike, len(r.Strikes))}
		labels, rows := rec.Strikes, make([][]float32, len(r.Strikes))
		var answers []bool
		answers, rec.Unanswered = rightStrikes(r, truth, t.timed)
		for k, st := range r.Strikes {
			labels[k], rows[k] = rowStrike{st.Frame, st.MIDI, st.Reported, st.Held, answers[k], st.Kept}, st.Row
			if answers[k] {
				right++
			}
		}
		if err := writeMel(base+".rows", rows); err != nil {
			return err
		}
		if err := writeJSON(base+".json", rec); err != nil {
			return err
		}
		written, strikes, given, unanswered = written+1, strikes+len(rows), given+rec.Given, unanswered+rec.Unanswered
		if t.ear.read {
			read++
		}
	}
	what := fmt.Sprintf("wrote the rows of %d recording(s) to %s from %s: %d strike(s), %d of them right, against %d note(s) given, %d of which no strike answers", written, dir, from, strikes, right, given, unanswered)
	if kept > 0 {
		what += fmt.Sprintf("; %d were there", kept)
	}
	if silent > 0 {
		what += fmt.Sprintf("; %d loop(s) kept no audio", silent)
	}
	if o.frames != "" {
		what += fmt.Sprintf("; their frames kept in %s, %d read back from it", o.frames, read)
	}
	if o.parts > 0 {
		what += fmt.Sprintf("; part %d of %d", o.part, o.parts)
	}
	fmt.Fprintln(s.out, what)
	if len(failed) > 0 {
		fmt.Fprintf(s.out, "no rip of %d, as %s\n", len(failed), failed[0])
	}
	return nil
}

// truthSources is the recordings a rows run reads: a truth list's lines, or for
// a directory its corpus loops that kept their audio, each against its label on
// its own clock, the corpus a condition of its own under its folder's name, and
// how many loops kept none
func truthSources(from string) (lines []truthLine, silent int, err error) {
	if info, err := os.Stat(from); err != nil {
		return nil, 0, err
	} else if !info.IsDir() {
		lines, err := readTruthLines(from)
		return lines, 0, err
	}
	labels, _ := filepath.Glob(filepath.Join(from, "loop_*.json"))
	for _, label := range labels {
		wav := strings.TrimSuffix(filepath.Base(label), ".json") + ".wav"
		if _, err := os.Stat(filepath.Join(from, wav)); err != nil {
			silent++
			continue
		}
		lines = append(lines, truthLine{dir: from, wav: wav, truth: filepath.Base(label), cond: filepath.Base(from), timed: true})
	}
	if len(lines) == 0 {
		return nil, 0, fmt.Errorf("%s: no loop with its audio beside its label (a corpus written without nowav)", from)
	}
	return lines, silent, nil
}

// rightStrikes labels a rip's strikes against a truth: right where a truth note
// of its pitch starts within the matcher's window, each truth note answering
// one strike, the nearest, the reported before the dropped; a truth off the
// recording's clock is placed by the map the rip's notes fit. It returns each
// strike's label and how many truth notes no strike answers.
func rightStrikes(r *audio.Rip, truth []audio.RipNote, timed bool) ([]bool, int) {
	ag := audio.MatchNotes(r.Notes, truth, timed)
	left := make([]audio.RipNote, len(truth))
	for i, n := range truth {
		left[i] = audio.RipNote{Start: ag.At(n.Start), MIDI: n.MIDI + ag.Shift}
	}
	right := make([]bool, len(r.Strikes))
	for _, reported := range [...]bool{true, false} {
		var stage []audio.RipNote
		for _, st := range r.Strikes {
			if st.Reported == reported {
				stage = append(stage, audio.RipNote{Start: float64(st.Frame) * audio.RipFrameSec, MIDI: st.MIDI})
			}
		}
		all := audio.MatchNotes(stage, left, true).All
		unmatched := map[[2]int]int{}
		for _, n := range all.OnlyA {
			unmatched[[2]int{int(math.Round(n.Start / audio.RipFrameSec)), n.MIDI}]++
		}
		for i, st := range r.Strikes {
			if st.Reported != reported {
				continue
			}
			if k := [2]int{st.Frame, st.MIDI}; unmatched[k] > 0 {
				unmatched[k]--
			} else {
				right[i] = true
			}
		}
		left = all.OnlyB
	}
	return right, len(left)
}
