package stapes

import (
	"bytes"
	"cmp"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/lixenwraith/audio"
)

// cmdAgree compares the transcripts of two rips of one piece, two witnesses
// where no score is to hand: what both hear is likely, what one hears is to
// doubt
func cmdAgree(s *Session, a []string) error {
	ra, ok := s.rips[a[0]]
	if !ok {
		return fmt.Errorf("%q: nothing ripped under that name this session (rip first)", a[0])
	}
	rb, ok := s.rips[a[1]]
	if !ok {
		return fmt.Errorf("%q: nothing ripped under that name this session (rip first)", a[1])
	}
	ag := audio.MatchNotes(ra.rip.Notes, rb.rip.Notes, false)
	fmt.Fprintf(s.out, "%s against %s, %s transposed %+d: %s's seconds = %.3f × %s's %+.2f\n", a[0], a[1], a[1], ag.Shift, a[0], ag.Stretch, a[1], ag.Offset)
	reportAgreement(s.out, ag, a[0], a[1], "heard by both")
	return nil
}

// cmdTruth scores a rip's transcript against one given as text in the notes
// verb's own form, as a MIDI file, whose drum channel scores the drums, or as a
// corpus loop's label: the notes given it heard by pitch and onset, then line by
// line, and the hits; a MIDI file's bars, a label's and a JAMS file's beats, a
// truth of beats alone, score its bar lines, and a label's readings what it says
// past them; a label's or a MIDI file's strikes score what the rip heard struck.
// Timed, the truth's seconds are the recording's: no map is fitted.
func cmdTruth(s *Session, a []string) error {
	rp, ok := s.rips[a[0]]
	if !ok {
		return fmt.Errorf("%q: nothing ripped under that name this session (rip first)", a[0])
	}
	var timed bool
	for w := a[2:]; len(w) > 0; w = w[1:] {
		switch {
		case w[0] == "timed":
			timed = true
		default:
			return fmt.Errorf("%q: want timed, for a truth whose seconds are the recording's, or nothing", w[0])
		}
	}
	label := strings.EqualFold(filepath.Ext(a[1]), ".json")
	data, err := os.ReadFile(s.path(a[1]))
	if err != nil {
		return err
	}
	var truth []audio.RipNote
	var hits, strikes []audio.RipHit
	var score *audio.MIDIScore
	var lab *corpusLoop
	switch {
	case isMIDI(a[1]):
		if score, err = audio.ReadMIDI(bytes.NewReader(data)); err == nil {
			truth, hits, strikes = score.Notes, score.Hits, slices.Concat(score.Hits, score.Struck)
		}
	case label:
		lab = new(corpusLoop)
		if err = json.Unmarshal(data, lab); err == nil {
			truth, hits, strikes = lab.notes(), lab.hits(), lab.strikes()
		}
	case strings.EqualFold(filepath.Ext(a[1]), ".jams"):
		beats, downbeats, err := audio.ReadJAMSBeats(bytes.NewReader(data))
		if err != nil {
			return fmt.Errorf("%s: %w", a[1], err)
		}
		reportBars(s.out, a[0], rp.rip, beats, downbeats)
		return nil
	default:
		truth, err = transcript(string(data), rp.rip)
	}
	if err != nil {
		return fmt.Errorf("%s: %w", a[1], err)
	}
	// a score's bars are placed on the recording as its notes matched, or with
	// none as its hits did
	var at func(float64) float64
	var ag audio.Agreement
	if len(truth) > 0 {
		ag = audio.MatchNotes(rp.rip.Notes, truth, timed)
		if rp.x != nil { // nothing heard ends past the recording
			ag = ag.Until(float64(len(rp.x)) / float64(rp.rate))
		}
		if timed {
			fmt.Fprintf(s.out, "%s against %d note(s) given on the recording's own clock: no map fitted, no transposition tried\n", a[0], len(truth))
		} else {
			fmt.Fprintf(s.out, "%s against %d note(s) given, transposed %+d: the rip's seconds = %.3f × the given %+.2f\n", a[0], len(truth), ag.Shift, ag.Stretch, ag.Offset)
		}
		c, by := everyNote(ag), "by pitch and onset"
		if ag.All.Rhythm {
			by = "by onset, a rhythm giving no pitch"
		}
		fmt.Fprintf(s.out, "notes  %d heard as given, given %d, %s %d, F1 %.3f: %s, whatever line holds a note\n", c.hits, c.given, a[0], c.heard, c.f1(), by)
		reportAgreement(s.out, ag, a[0], "given", "heard as given")
		if slices.ContainsFunc(truth, func(n audio.RipNote) bool { return n.Further }) {
			c := agreed(ag)
			fmt.Fprintf(s.out, "every part %s %d, given %d, %d heard as given, F1 %.2f: the parts past the lines in the melody register as one line, or the bass under E3\n", a[0], c.heard, c.given, c.hits, c.f1())
		}
		reportLevels(s.out, rp, ag, truth)
		at = ag.At
	}
	var hg *audio.HitsAgreement
	if len(hits) > 0 {
		hg = new(reportHits(s.out, a[0], rp.rip, hits, timed))
		if at == nil {
			at = func(sec float64) float64 { return hg.Stretch*sec + hg.Offset }
		}
	}
	if lab != nil || len(strikes) > 0 {
		reportStruck(s.out, a[0], rp, strikes, timed)
	}
	if score != nil {
		beats, downbeats := score.Beats(at)
		reportBars(s.out, a[0], rp.rip, beats, downbeats)
	}
	if lab != nil {
		if beats, downbeats := lab.grid(); len(downbeats) > 0 {
			reportBars(s.out, a[0], rp.rip, beats, downbeats)
		}
		var c truthCount
		c.notes, c.drums.given = everyNote(ag), len(hits)
		c.read(rp.rip, ag, hg, lab.said())
		reportReadings(s.out, c)
	}
	for _, l := range stageLines(audio.ReadStages(rp.rip, ag, hg, 30/float64(rp.rip.BPM)/audio.StepsPerBeat)) {
		fmt.Fprintf(s.out, "%-8s %s\n", l[0], l[1])
	}
	return nil
}

// stageLines is a stage reading as truth says it, a line a kind: the notes
// given the rip lacks, with why the strikes refused those lost there, the notes
// it adds, the ends it puts wrong, and the hits lost and added, each by stage,
// the largest first; none for a rip that kept no stages
func stageLines(r audio.StageReading) [][2]string {
	if r.Lost == nil {
		return nil
	}
	lost := fmt.Sprintf("%d of the notes given, by stage: %s", total(r.Lost), countWords(r.Lost))
	if len(r.Refused) > 0 {
		lost += "; at the strikes, " + countWords(r.Refused)
	}
	return [][2]string{{"lost", lost},
		{"added", fmt.Sprintf("%d, by stage: %s", total(r.Added), countWords(r.Added))},
		{"ends", fmt.Sprintf("%d wrong, by stage: %s", total(r.Ends), countWords(r.Ends))},
		{"hits", fmt.Sprintf("%d lost, by stage: %s; %d added, by stage: %s", total(r.HitsLost), countWords(r.HitsLost), total(r.HitsAdded), countWords(r.HitsAdded))}}
}

// countWords is a count by name as words, the largest first, "none" for none
func countWords(c map[string]int) string {
	names := slices.Collect(maps.Keys(c))
	slices.SortFunc(names, func(a, b string) int { return cmp.Or(c[b]-c[a], strings.Compare(a, b)) })
	var out []string
	for _, n := range names {
		out = append(out, fmt.Sprintf("%s %d", n, c[n]))
	}
	if len(out) == 0 {
		return "none"
	}
	return strings.Join(out, ", ")
}

func total(c map[string]int) (n int) {
	for _, v := range c {
		n += v
	}
	return n
}

// reading is a row of readings: the word tune truth weighs it by, its name,
// whether a truth gives what it is read over (its notes, its hits or itself),
// whether a count gives the reading, its score, a share, what it is of and
// what it says on a line beside, nil for nothing
type reading struct {
	word, name string
	over       func(truthCount) bool
	given      func(truthCount) bool
	score      func(truthCount) float64
	of, beside func(truthCount) string
}

// readings are what a label says past pitch and onset as the rip reads it
// (audio's readings), read where a label gives it, weighed by tune truth alone
// where its words ask
var readings = []reading{
	{"ends", "ends", notesGiven, func(c truthCount) bool { return c.stated.ends > 0 }, func(c truthCount) float64 { return share(c.ends.Right, c.ends.Given) },
		func(c truthCount) string {
			var past string
			if c.ends.Past > 0 {
				past = fmt.Sprintf(", %d given past the recording's end and read at it, %d of them wrong as given", c.ends.Past, c.ends.PastWrong)
			}
			return fmt.Sprintf("%d note(s) matched whose end is given%s, each right within %g of its length or %g ms, the median end %+.2f of a length off",
				c.ends.Given, past, audio.EndShare, 1000*audio.EndFloor, median(c.ends.Off))
		}, nil},
	{"loudness", "loudness", notesGiven, func(c truthCount) bool { return c.stated.loud > 0 }, func(c truthCount) float64 { return share(c.loud.Right, c.loud.Given) },
		func(c truthCount) string {
			return fmt.Sprintf("%d note(s) matched whose velocity is given, each right within %g dB once the recording's level is out, the median %.1f dB off",
				c.loud.Given, audio.LevelBound, medianOff(c.loud.Off))
		}, nil},
	{"loudness", "drums' loudness", func(c truthCount) bool { return c.drums.given > 0 }, func(c truthCount) bool { return c.stated.hitLoud > 0 },
		func(c truthCount) float64 { return share(c.hitLoud.Right, c.hitLoud.Given) },
		func(c truthCount) string {
			return fmt.Sprintf("%d hit(s) matched whose velocity is given, each right within %g dB once its drum's level is out, the median %.1f dB off",
				c.hitLoud.Given, audio.LevelBound, medianOff(c.hitLoud.Off))
		}, nil},
	{"parts", "parts", notesGiven, func(c truthCount) bool { return c.stated.parts > 0 },
		func(c truthCount) float64 {
			if c.parts.Given == 0 {
				return 0
			}
			in, on := c.parts.InPart/float64(c.parts.Given), c.parts.OnLine/float64(c.parts.Given)
			return 2 * in * on / (in + on)
		},
		func(c truthCount) string {
			n := float64(max(c.parts.Given, 1))
			return fmt.Sprintf("%d note(s) matched in a part given, the F1 of %.3f of a line's notes of their part and %.3f of a part's on their line",
				c.parts.Given, c.parts.InPart/n, c.parts.OnLine/n)
		}, nil},
	{"key", "key", func(truthCount) bool { return true }, func(c truthCount) bool { return c.stated.keys > 0 },
		func(c truthCount) float64 { return share(c.keys.miss[audio.KeyRight], c.keys.given) },
		func(c truthCount) string {
			return fmt.Sprintf("%d recording(s) whose key is given, right in both its tonic and its scale: %d tonic(s) right, %d scale(s)", c.keys.given, c.keys.tonic, c.keys.scale)
		}, keyMissed},
	{"chords", "chords", func(truthCount) bool { return true }, func(c truthCount) bool { return c.stated.chords > 0 },
		func(c truthCount) float64 { return c.chords.Right / max(c.chords.Given, 1e-9) },
		func(c truthCount) string {
			return fmt.Sprintf("%.1f s of chords given, right where a bar of the rip's names the chord's root", c.chords.Given)
		}, nil},
}

// keyMisses says each kind of a key's miss, by audio's KeyMiss
var keyMisses = [audio.KeyMisses]string{"", "at the given tonic in a scale holding or held by the given's", "at the given's tones under another tonic", "else"}

// keyMissed is how many keys a count missed, by the kind of each miss, and
// how many of them no rip named, failing
func keyMissed(c truthCount) string {
	unnamed := c.keys.given
	var kinds []string
	for k, n := range c.keys.miss {
		if unnamed -= n; audio.KeyMiss(k) != audio.KeyRight {
			kinds = append(kinds, fmt.Sprintf("%d %s", n, keyMisses[k]))
		}
	}
	if unnamed > 0 {
		kinds = append(kinds, fmt.Sprintf("%d named by no rip", unnamed))
	}
	return fmt.Sprintf("%d missed: %s", c.keys.given-c.keys.miss[audio.KeyRight], strings.Join(kinds, ", "))
}

func notesGiven(c truthCount) bool { return c.notes.given > 0 }

func share(right, given int) float64 { return float64(right) / float64(max(given, 1)) }

// medianOff is the median of how far each is off, its sign left out
func medianOff(off []float64) float64 {
	abs := make([]float64, len(off))
	for i, v := range off {
		abs[i] = math.Abs(v)
	}
	return median(abs)
}

// readingWords are the words the readings are weighed by, each once
func readingWords() []string {
	var out []string
	for _, rd := range readings {
		if !slices.Contains(out, rd.word) {
			out = append(out, rd.word)
		}
	}
	return out
}

// read reads what a label says past pitch and onset as r reads it: over the
// notes ag matched, the hits hg matched (nil where its truth knows no drum),
// and against its key and chords
func (c *truthCount) read(r *audio.Rip, ag audio.Agreement, hg *audio.HitsAgreement, said *labelSaid) {
	c.ends, c.loud, c.parts, c.stated = ag.Ends(), ag.Levels(), ag.Parts(), said.stated
	if hg != nil {
		c.hitLoud = hg.Levels()
	}
	if said.key != nil {
		one := func(b bool) int {
			if b {
				return 1
			}
			return 0
		}
		tonic, scale, miss := audio.MatchKey(r, said.key[0], audio.ScaleID(said.key[1]))
		c.keys = keyCount{given: 1, tonic: one(tonic), scale: one(scale)}
		c.keys.miss[miss] = 1
	}
	c.chords = audio.MatchChords(r, said.chords)
}

// reportReadings prints a row a reading a label gives, and names those it
// gives none of where it gives what they are read over
func reportReadings(w io.Writer, c truthCount) {
	var lacks []string
	for _, rd := range readings {
		switch {
		case rd.given(c):
			fmt.Fprintf(w, "%-8s %.3f of %s\n", rd.name, rd.score(c), rd.of(c))
			if rd.beside != nil {
				fmt.Fprintf(w, "%-8s %s\n", "", rd.beside(c))
			}
		case rd.over(c):
			lacks = append(lacks, rd.name)
		}
	}
	if len(lacks) > 0 {
		fmt.Fprintf(w, "the label gives no %s\n", strings.Join(lacks, ", "))
	}
}

// matchHeard matches a rip's notes to a truth's as MatchNotes does, the notes
// given at pitches the rip's front end does not tell apart, under the
// transposition found, left out: no rip of its weights could report them, so a
// search and a judge's rows are not held to them. It returns the agreement and
// the notes it counted.
func matchHeard(r *audio.Rip, given []audio.RipNote, timed bool) (audio.Agreement, []audio.RipNote) {
	ag := audio.MatchNotes(r.Notes, given, timed)
	if r.Range == (audio.RipRange{}) {
		return ag, given
	}
	heard := slices.DeleteFunc(slices.Clone(given), func(n audio.RipNote) bool { return n.MIDI >= 0 && !r.Range.Holds(n.MIDI+ag.Shift) })
	if len(heard) == len(given) {
		return ag, given
	}
	return audio.MatchNotes(r.Notes, heard, timed), heard
}

// transcript reads notes given as text against a rip: at its tempo, and a rhythm
// tapped out without a start from where its melody register first sounds
func transcript(text string, r *audio.Rip) ([]audio.RipNote, error) {
	origin := math.Inf(1)
	for _, n := range r.Notes {
		if !n.Bass {
			origin = math.Min(origin, n.Start)
		}
	}
	if math.IsInf(origin, 1) {
		origin = 0
	}
	return audio.ParseTranscript(text, r.BPM, origin)
}

// reportBars prints the rip's bar lines scored against the beats and downbeats
// given (MatchBars)
func reportBars(w io.Writer, name string, r *audio.Rip, beats, downbeats []float64) {
	b := audio.MatchBars(r, beats, downbeats)
	same := "the same length"
	if !b.Same {
		same = "not the same length, so no downbeat counts"
	}
	fmt.Fprintf(w, "bars   %s's bar %.2f of the given's, %s: %.2f of %d downbeat(s) given on a bar line, phase %+.2f bar, %.2f of %d beat(s) given on a beat; of those before the first downbeat %d of %d lost, after the last %d of %d\n",
		name, b.Ratio, same, b.Downbeats, len(downbeats), b.Phase, b.Beats, len(beats), b.BeforeLost, b.Before, b.AfterLost, b.After)
}

// reportHits prints each drum's count of a rip's hits against the hits given,
// matched within half a step, and the hits only one side has; timed, the hits
// given are on the recording's own clock and no map is fitted
func reportHits(w io.Writer, name string, r *audio.Rip, given []audio.RipHit, timed bool) audio.HitsAgreement {
	hg := audio.MatchHits(r.Hits(), given, 30/float64(r.BPM)/audio.StepsPerBeat, timed)
	if timed {
		fmt.Fprintf(w, "%s against %d hit(s) given on the recording's own clock: no map fitted\n", name, len(given))
	} else {
		fmt.Fprintf(w, "%s against %d hit(s) given: the rip's seconds = %.3f × the given %+.2f\n", name, len(given), hg.Stretch, hg.Offset)
	}
	brief := func(hits []audio.RipHit) string {
		notes := make([]audio.RipNote, len(hits))
		for i, h := range hits {
			notes[i] = audio.RipNote{Start: h.Start, MIDI: -1}
		}
		return notesInBrief(notes)
	}
	for d, l := range hg.Drums {
		if l.A > 0 || l.B > 0 {
			fmt.Fprintf(w, "%-6s %s %d, given %d, %d heard as given, only %s %d, only given %d\n", audio.InstrumentName(audio.InstrumentType(d)), name, l.A, l.B, l.Matched, name, len(l.OnlyA), len(l.OnlyB))
		}
	}
	for d, l := range hg.Drums {
		if len(l.OnlyA) > 0 {
			fmt.Fprintf(w, "  %s only %s: %s\n", audio.InstrumentName(audio.InstrumentType(d)), name, brief(l.OnlyA))
		}
		if len(l.OnlyB) > 0 {
			fmt.Fprintf(w, "  %s only given: %s\n", audio.InstrumentName(audio.InstrumentType(d)), brief(l.OnlyB))
		}
	}
	return hg
}

// reportStruck prints how a rip's strikes answer those given by their time alone,
// its lines' hits counted beside them, or how many it heard where none is given;
// nothing for a rip that heard none
func reportStruck(w io.Writer, name string, rp *ripped, given []audio.RipHit, timed bool) {
	r, minutes := rp.rip, 0.0
	if rp.rate > 0 {
		minutes = float64(len(rp.x)) / float64(rp.rate) / 60
	}
	c := countStruck(r, given, 30/float64(r.BPM)/audio.StepsPerBeat, timed, minutes)
	switch {
	case c.read == 0:
	case len(given) > 0:
		fmt.Fprintf(w, "struck %s %d, given %d instant(s), %d answered by time alone, F1 %.3f, the lines' hits %.3f; each of their %d strike(s) its own hit, %.3f, the lines' %.3f, and one hit an instant %.3f at most\n",
			name, c.by.heard, c.by.given, c.by.hits, c.by.f1(), c.lines.f1(), c.own.given, c.own.f1(), c.linesOwn.f1(), c.most())
	default:
		fmt.Fprintf(w, "struck %s %d, %.1f a minute where none plays, the lines' hits %d\n", name, c.own.heard, float64(c.own.heard)/max(minutes, 1e-9), c.linesOwn.heard)
	}
}

// levelBands part the notes given by how far, in dB, the recording plays each
// under the loudest tenth of them
var levelBands = [...]float64{-6, -12, -18}

// reportLevels prints per line, and for the melody register as one, how many of
// the notes given the rip heard by how loud the recording plays them and whether
// another note given sounds at their onset; a rhythm has no pitch to read at,
// and a score no recording to read it in
func reportLevels(w io.Writer, rp *ripped, ag audio.Agreement, given []audio.RipNote) {
	if rp.x == nil || slices.ContainsFunc(given, func(n audio.RipNote) bool { return n.MIDI < 0 }) {
		return
	}
	level := audio.NoteLevels(rp.x, rp.rate, given, ag.At, rp.rip.Tuning)
	sorted := slices.Sorted(slices.Values(level))
	top := sorted[len(sorted)*9/10]
	type key struct {
		start float64
		midi  int
	}
	fmt.Fprintf(w, "heard by level under the loudest tenth given: from %g dB, %g to %g, %g to %g, under %g; alone or under others\n",
		levelBands[0], levelBands[0], levelBands[1], levelBands[1], levelBands[2], levelBands[2])
	row := func(name string, keep func(audio.RipNote) bool, missed []audio.RipNote) {
		miss := map[key]bool{}
		for _, n := range missed {
			miss[key{n.Start, n.MIDI}] = true
		}
		var heard, total [len(levelBands) + 1]int
		var alone, under [2]int // heard, given
		for i, n := range given {
			if !keep(n) {
				continue
			}
			band := 0
			for band < len(levelBands) && level[i]-top < levelBands[band] {
				band++
			}
			h := 0
			if !miss[key{n.Start, n.MIDI}] {
				h = 1
			}
			heard[band], total[band] = heard[band]+h, total[band]+1
			by := &alone
			if slices.ContainsFunc(given, func(o audio.RipNote) bool { return o != n && o.Start <= n.Start+0.03 && o.End > n.Start }) {
				by = &under
			}
			by[0], by[1] = by[0]+h, by[1]+1
		}
		if alone[1]+under[1] == 0 {
			return
		}
		fmt.Fprintf(w, "%-8s", name)
		for b := range total {
			fmt.Fprintf(w, " %d/%d", heard[b], total[b])
		}
		fmt.Fprintf(w, "; alone %d/%d, under others %d/%d\n", alone[0], alone[1], under[0], under[1])
	}
	row("melody", func(n audio.RipNote) bool { return !n.Bass && !n.Second }, ag.Lines[0].OnlyB)
	row("under", func(n audio.RipNote) bool { return n.Second && !n.Further }, ag.Lines[1].OnlyB)
	row("bass", func(n audio.RipNote) bool { return n.Bass }, ag.Lines[2].OnlyB)
	row("register", func(n audio.RipNote) bool { return !n.Bass }, ag.Register.OnlyB)
}

// isMIDI is whether a file names a Standard MIDI File by its extension
func isMIDI(file string) bool {
	ext := strings.ToLower(filepath.Ext(file))
	return ext == ".mid" || ext == ".midi"
}

// reportAgreement prints each line's counts and the notes only one side has
func reportAgreement(w io.Writer, ag audio.Agreement, nameA, nameB, both string) {
	words := [...]string{"melody", "under", "bass"}
	for i, l := range ag.Lines {
		if l.A == 0 && l.B == 0 {
			continue
		}
		fmt.Fprintf(w, "%-6s %s %d, %s %d, %d %s, only %s %d, only %s %d", words[i], nameA, l.A, nameB, l.B, l.Matched, both, nameA, len(l.OnlyA), nameB, len(l.OnlyB))
		if l.Rhythm {
			fmt.Fprintf(w, "; by onset: %d of the given cut in two or more, %d of %s's span two or more", l.Split, l.Merged, nameA)
		}
		fmt.Fprintln(w)
	}
	if r, apart := ag.Register, ag.Lines[0].Matched+ag.Lines[1].Matched; r.Matched > apart && r.B == ag.Lines[0].B+ag.Lines[1].B {
		fmt.Fprintf(w, "melody register as one line: %d %s, %d more than the lines apart, so the two part the lines differently\n", r.Matched, both, r.Matched-apart)
	}
	for i, l := range ag.Lines {
		if len(l.OnlyA) > 0 {
			fmt.Fprintf(w, "  %s only %s: %s\n", words[i], nameA, notesInBrief(l.OnlyA))
		}
		if len(l.OnlyB) > 0 {
			fmt.Fprintf(w, "  %s only %s: %s\n", words[i], nameB, notesInBrief(l.OnlyB))
		}
		if len(l.Cut) > 0 {
			fmt.Fprintf(w, "  %s %s cut in two or more: %s\n", words[i], nameB, notesInBrief(l.Cut))
		}
		if len(l.Spanning) > 0 {
			fmt.Fprintf(w, "  %s %s spanning two or more: %s\n", words[i], nameA, notesInBrief(l.Spanning))
		}
	}
}

// notesInBrief names notes by pitch and second, a rhythm's by second alone
func notesInBrief(notes []audio.RipNote) string {
	var b strings.Builder
	for i, n := range notes {
		if i > 0 {
			b.WriteString(" ")
		}
		if n.MIDI < 0 {
			fmt.Fprintf(&b, "%.2f", n.Start)
		} else {
			fmt.Fprintf(&b, "%s@%.2f", noteName(n.MIDI), n.Start)
		}
	}
	return b.String()
}
