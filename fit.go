package stapes

import (
	"bytes"
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/lixenwraith/audio"
	"github.com/lixenwraith/genetic"
)

// fitOptions sizes a voice fit: the notes of a register it listens to, the
// generations run over a pool of candidate voices, the seed every draw follows
// and the candidates scored at once, the machine's cores at 0
type fitOptions struct {
	notes, generations, pool int
	seed                     uint64
	cores                    int
}

var defaultFit = fitOptions{notes: 6, generations: 8, pool: 16, seed: 1}

// fitWords are the words that size a fit on a verb's line, each before its
// number, with the numbers it takes
var fitWords = [...]struct {
	word   string
	lo, hi int
	set    func(*fitOptions, int)
}{
	{"notes", 1, 64, func(o *fitOptions, n int) { o.notes = n }},
	{"pool", 4, 1024, func(o *fitOptions, n int) { o.pool = n }},
	{"generations", 1, 500, func(o *fitOptions, n int) { o.generations = n }},
	{"seed", 0, math.MaxInt32, func(o *fitOptions, n int) { o.seed = uint64(n) }},
}

// size sets the size a word of fitWords names to its number
func (o *fitOptions) size(word, number string) error {
	for _, f := range fitWords {
		if f.word != word {
			continue
		}
		n, err := strconv.Atoi(number)
		if err != nil || n < f.lo || n > f.hi {
			return fmt.Errorf("%s %q: want %d to %d", word, number, f.lo, f.hi)
		}
		f.set(o, n)
		return nil
	}
	return fmt.Errorf("%q: want notes, pool, generations or seed, each before its number", word)
}

// fitTail is how long past a note's end its sample runs, for the release
const fitTail = 0.25

// ripped is a recording ripped this session: what was heard, and the samples it
// was heard in, so fit and compress work on it without the file; a score read
// from a MIDI file has no samples
type ripped struct {
	rip   *audio.Rip
	x     []float64
	rate  int
	file  string
	sound *audio.RipSound // the recording as Fidelity hears it, heard on first use
}

// errScore refuses a verb that hears the recording a score lacks
func errScore(rp *ripped) error {
	return fmt.Errorf("%s is a score, with no recording to hear or fit to", rp.file)
}

func cmdFit(s *Session, a []string) error {
	rp, ok := s.rips[a[0]]
	if !ok {
		return fmt.Errorf("%q: nothing ripped under that name this session (rip first)", a[0])
	}
	o, w := s.fit, a[1:]
	if len(w)%2 == 1 { // a number alone is the generations
		if err := o.size("generations", w[0]); err != nil {
			return err
		}
		w = w[1:]
	}
	for ; len(w) > 0; w = w[2:] {
		if err := o.size(w[0], w[1]); err != nil {
			return err
		}
	}
	if rp.x == nil {
		return errScore(rp)
	}
	s.fitVoices(rp, o)
	for _, v := range rp.rip.Voices(a[0]) {
		if _, err := s.eng.DefineVoice(v.Clone()); err != nil {
			return fmt.Errorf("voice %q: %w", v.Name, err)
		}
		s.voices.put(v, false)
		fmt.Fprintf(s.out, "voice %s: %s, registered; the patterns naming it play it\n", v.Name, cmp.Or(v.Wave, "sine"))
	}
	return nil
}

// cmdNotes lists a rip's transcript as heard before the grid, so a listener checks
// it by eye: each note's bar and step, its pitch as the key's degree with any
// accidental, length, loudness and support, a melody-register note's brightness
// where measured (the header counts the rest), and the wave and sure family of a
// note told its voice; or writes the transcript to a MIDI file at the rip's tempo.
func cmdNotes(s *Session, a []string) error {
	rp, ok := s.rips[a[0]]
	if !ok {
		return fmt.Errorf("%q: nothing ripped under that name this session (rip first)", a[0])
	}
	r := rp.rip
	if len(a) > 1 {
		if !isMIDI(a[1]) {
			return fmt.Errorf("%s: want a .mid file", a[1])
		}
		var b bytes.Buffer
		hits := r.Hits()
		if err := audio.WriteMIDI(&b, r.Notes, hits, r.BPM, r.Meter); err != nil {
			return err
		}
		if err := os.WriteFile(s.path(a[1]), b.Bytes(), 0o644); err != nil {
			return err
		}
		fmt.Fprintf(s.out, "%s: %d note(s) and %d hit(s) at %d bpm in %s written to %s, a track each for melody, under, bass and drums\n", a[0], len(r.Notes), len(hits), r.BPM, r.Meter, a[1])
		return nil
	}
	// support is how much of the recording's salience over a note's frames is
	// its pitch's: one for a note alone in its frames, little for one heard
	// beside a louder, a note to doubt; a score's notes are given, not heard
	var snd *audio.RipSound
	if rp.x != nil {
		var err error
		if snd, err = s.soundOf(rp); err != nil {
			return err
		}
	}
	support := func(n audio.RipNote) string {
		if snd == nil {
			return ""
		}
		f, pitches := snd.Frames, snd.Frames.Pitches()
		var own, all float64
		for t := int(n.Start / f.Hop); t < int(n.End/f.Hop) && (t+1)*pitches.Len() <= len(f.Salience); t++ {
			row := f.Salience[t*pitches.Len() : (t+1)*pitches.Len()]
			if i := n.MIDI - pitches.Min; i >= 0 && i < len(row) {
				own += float64(row[i])
			}
			var peak float32
			for _, v := range row {
				peak = max(peak, v)
			}
			all += float64(peak)
		}
		if all > 0 {
			own /= all
		}
		return fmt.Sprintf(", support %.2f", own)
	}
	var upper, unmeasured int
	for _, n := range r.Notes {
		if !n.Bass {
			upper++
			if n.Bright == audio.RipUnmeasured {
				unmeasured++
			}
		}
	}
	fmt.Fprintf(s.out, "%s: %d note(s) at %d bpm in %s, %s %s, tuned %+.0f cents, the first bar from %.2fs, brightness unmeasured on %d of %d melody-register note(s)\n", a[0], len(r.Notes), r.BPM, r.Meter, noteName(r.Root), scaleName(r.Scale), r.Tuning, r.Downbeat, unmeasured, upper)
	marks := [...]string{"bb", "b", "", "#", "x"}
	beat, per := r.Meter.Beat(), r.Meter.Beats()
	for _, n := range r.Notes {
		line := "melody"
		switch {
		case n.Bass:
			line = "bass"
		case n.Second:
			line = "under"
		}
		where, span := "before the first bar", 60/float64(max(r.BPM, 1))/audio.StepsPerBeat
		if i := sort.SearchFloat64s(r.Beats, n.Start+1e-9) - 1; i >= 0 && i+1 < len(r.Beats) {
			span = (r.Beats[i+1] - r.Beats[i]) / float64(beat)
			where = fmt.Sprintf("bar %d step %2d", i/per+1, beat*(i%per)+int(math.Round((n.Start-r.Beats[i])/span)))
		}
		deg, oct, acc := audio.PitchStep(r.Root, r.Scale, n.MIDI)
		bright := ""
		if n.Bright != audio.RipUnmeasured {
			bright = fmt.Sprintf(", bright %.2f", n.Bright)
		}
		bend := ""
		if n.Bend != 0 {
			bend = fmt.Sprintf(", bend %+.2f from %.2f", n.Bend, n.BendAt)
		}
		voice := ""
		if v := n.Voice; v != nil {
			voice = ", wave " + v.Wave
			if v.Family != "" {
				voice += ", family " + v.Family
			}
		}
		fmt.Fprintf(s.out, "%-6s %7.3fs %-20s %-4s degree %d%s octave %d, %.1f step(s) at %.2f%s%s%s%s\n", line, n.Start, where, noteName(n.MIDI), deg, marks[acc+2], oct, (n.End-n.Start)/span, n.Vel, support(n), bright, bend, voice)
	}
	return nil
}

// ripLines are a rip's three lines in the order their voices are fitted: the
// melody, the line under it and the bass, each with the notes that are its
var ripLines = [3]struct {
	word string
	of   func(audio.RipNote) bool
}{
	{"melody", func(n audio.RipNote) bool { return !n.Bass && !n.Second }},
	{"under", func(n audio.RipNote) bool { return !n.Bass && n.Second }},
	{"bass", func(n audio.RipNote) bool { return n.Bass }},
}

// lineVoices are the rip's voices in ripLines' order
func lineVoices(r *audio.Rip) [3]**audio.VoiceDef {
	return [3]**audio.VoiceDef{&r.MelodyVoice, &r.UnderVoice, &r.BassVoice}
}

// fitVoices fits a voice to each line of a rip, a voice the front end heard
// for it held as a seed, and says how far the fit came
func (s *Session) fitVoices(rp *ripped, o fitOptions) {
	for reg, at := range lineVoices(rp.rip) {
		notes := lineNotes(rp.rip, reg)
		if len(notes) == 0 {
			continue
		}
		word := ripLines[reg].word
		fmt.Fprintf(s.out, "fitting the %s voice over %d note(s), %d generation(s) of %d...\n", word, min(len(notes), o.notes), o.generations, o.pool)
		f := s.fitVoice(rp.x, rp.rate, rp.rip.Tuning, notes, *at, o)
		if f == nil {
			continue
		}
		*at = f.voice
		fmt.Fprintf(s.out, "%s voice: %s, distance %.3f from the best seed's %.3f\n", word, cmp.Or(f.voice.Wave, "sine"), f.to, f.from)
	}
}

// fitSamples picks the notes a register is fitted on: the longest and loudest
// first, no pitch twice until every pitch has a note, each at least 80 ms
func fitSamples(notes []audio.RipNote, n int) []audio.RipNote {
	var cands []audio.RipNote
	for _, note := range notes {
		if note.End-note.Start >= 0.08 {
			cands = append(cands, note)
		}
	}
	sort.SliceStable(cands, func(i, j int) bool {
		return (cands[i].End-cands[i].Start)*cands[i].Vel > (cands[j].End-cands[j].Start)*cands[j].Vel
	})
	var out []audio.RipNote
	seen := map[int]bool{}
	for pass := range 2 {
		for _, c := range cands {
			if len(out) >= n {
				return out
			}
			if (pass == 0 && !seen[c.MIDI]) || (pass == 1 && !slices.Contains(out, c)) {
				out = append(out, c)
				seen[c.MIDI] = true
			}
		}
	}
	return out
}

// voiceGenome is a voice as a search sees it: its genes and, last, its wave as a
// fraction of the wave table; voiceFromGenome reads one back
func voiceGenome(v *audio.VoiceDef) []float64 {
	w := max(slices.Index(audio.VoiceWaves[:], cmp.Or(v.Wave, "sine")), 0)
	return append(v.Genes(), (float64(w)+0.5)/float64(len(audio.VoiceWaves)))
}

func voiceFromGenome(g []float64) *audio.VoiceDef {
	genes := audio.VoiceGeneCount
	w := min(max(int(g[genes]*float64(len(audio.VoiceWaves))), 0), len(audio.VoiceWaves)-1)
	v := audio.VoiceFromGenes(audio.VoiceWaves[w], g[:genes])
	v.Name = "fitted"
	return v
}

// genomeKey names a genome to three decimals, for a cache of its score
func genomeKey(g []float64) string {
	var b strings.Builder
	for _, v := range g {
		fmt.Fprintf(&b, "%.3f,", v)
	}
	return b.String()
}

// lineNotes are a rip's notes of one of ripLines' registers
func lineNotes(r *audio.Rip, reg int) []audio.RipNote {
	var notes []audio.RipNote
	for _, n := range r.Notes {
		if ripLines[reg].of(n) {
			notes = append(notes, n)
		}
	}
	return notes
}

// fitTarget is one note of the recording a voice is fitted on: its seconds, the
// mel frames of the recording around it and the note the voice plays to match
type fitTarget struct {
	start, end float64
	mel        [][]float32
	midi       int
	vel, dur   float64
}

// fitTargets picks a register's sample notes, n of them, and hears the recording
// around each
func fitTargets(x []float64, rate int, notes []audio.RipNote, n int) []fitTarget {
	samples := fitSamples(notes, max(n, 1))
	targets := make([]fitTarget, len(samples))
	for i, n := range samples {
		lo := min(int(n.Start*float64(rate)), len(x))
		hi := min(int((n.End+fitTail)*float64(rate)), len(x))
		targets[i] = fitTarget{start: n.Start, end: n.End, mel: audio.MelFrames(x[lo:hi], rate), midi: n.MIDI, vel: n.Vel, dur: n.End - n.Start}
	}
	return targets
}

// fitDistance is how far a voice's renders of the targets' notes, at the
// recording's tuning, sound from the recording around them: the fit's measure,
// which the refinement holds a voice to
func fitDistance(v *audio.VoiceDef, targets []fitTarget, cents float64) float64 {
	var sum float64
	for _, t := range targets {
		note := audio.RenderNoteAt(v, t.midi, cents, t.vel, t.dur, fitTail)
		sum += audio.MelDistance(audio.MelFrames(note, audio.AudioSampleRate), t.mel)
	}
	return sum / float64(max(len(targets), 1))
}

// fitted is a fit's outcome: the voice, its level set, the seed the measure put
// nearest, and the distance of each on the notes listened to
type fitted struct {
	voice, seed *audio.VoiceDef
	to, from    float64
}

// fitVoice searches for the voice whose engine-rendered notes sound most like a
// register's notes in the recording: a genome of genes and wave, seeded with every
// registered voice and the one given, scored by mel distance from the recording
// around each sample note to the voice playing it; nil when the register has no
// note long enough.
func (s *Session) fitVoice(x []float64, rate int, cents float64, notes []audio.RipNote, prior *audio.VoiceDef, o fitOptions) *fitted {
	targets := fitTargets(x, rate, notes, o.notes)
	if len(targets) == 0 {
		return nil
	}
	genes := audio.VoiceGeneCount
	decode, encode := voiceFromGenome, voiceGenome
	eval := genetic.Memoize(genomeKey, func(g []float64) float64 {
		v := decode(g)
		if audio.ValidateVoice(v) != nil {
			return math.Inf(-1)
		}
		return -fitDistance(v, targets, cents)
	})
	var seeds [][]float64
	if prior != nil {
		seeds = append(seeds, encode(prior))
	}
	for _, v := range audio.RegisteredVoices() {
		seeds = append(seeds, encode(v))
	}
	best := math.Inf(-1)
	var seed *audio.VoiceDef // decoded here, before the engine owns the seeds
	for _, g := range seeds {
		d := eval(g)
		if seed == nil || d > best {
			seed = decode(g)
		}
		best = max(best, d)
	}
	bounds := make([]genetic.ParameterBounds, genes+1)
	for i := range bounds {
		bounds[i] = genetic.ParameterBounds{Min: 0, Max: 1}
	}
	cfg := genetic.DefaultConfig()
	cfg.PoolSize, cfg.EliteCount, cfg.MaxIterations = max(o.pool, 4), max(o.pool/8, 1), o.generations
	cfg.Parallelism, cfg.Seed = cmp.Or(o.cores, runtime.NumCPU()), o.seed
	cfg.PerturbationRate, cfg.PerturbationStrength = 0.15, 0.1
	e := genetic.NewEngine[[]float64, float64](eval, func(rng *rand.Rand) []float64 {
		g := make([]float64, genes+1)
		for i := range g {
			g[i] = rng.Float64()
		}
		return g
	}, &genetic.TournamentSelector[[]float64, float64]{TournamentSize: 3},
		&genetic.UniformCombiner[[]float64, float64, float64]{MixProbability: 0.5},
		&genetic.BoundedPerturbator{Bounds: bounds, Boundary: genetic.BoundaryReflect}, cfg)
	e.SetSeeds(seeds)
	e.SetSchedule(genetic.AdaptiveStrength[float64](0.15, 0.1, 0.01, 0.4, 1.5))
	e.SetTerminator(func(p *genetic.Pool[[]float64, float64], gen int) bool {
		fmt.Fprintf(s.out, "generation %d: distance %.3f\n", gen, -p.Stats.BestScore)
		return false
	})
	pool, err := e.Run(context.Background())
	if err != nil || len(pool.Members) == 0 {
		return nil
	}
	// MelFrames hears every sample at its peak, so the level is the search's
	// blind spot: set it so the voice's notes carry the loudness the recording
	// has around them
	v := decode(pool.Members[0].Data)
	var want, have float64
	for _, t := range targets {
		lo, hi := min(int(t.start*float64(rate)), len(x)), min(int(t.end*float64(rate)), len(x))
		want += rms(x[lo:hi])
		have += rms(audio.RenderNoteAt(v, t.midi, cents, t.vel, t.dur, 0))
	}
	if have > 0 && want > 0 {
		v.Level = min(max(v.Level*want/have, 0.1), 4) // the level ValidateVoice allows
	}
	return &fitted{voice: v, seed: seed, to: -pool.Members[0].Score, from: -best}
}

func rms(x []float64) float64 {
	var e float64
	for _, v := range x {
		e += v * v
	}
	return math.Sqrt(e / float64(max(len(x), 1)))
}

// cmdGenes says each voice named as a label states one, a line of JSON a voice:
// its wave and its genes in VoiceDef.Genes' order
func cmdGenes(s *Session, a []string) error {
	for _, name := range a {
		v, ok := s.voices.get(name)
		if !ok {
			return fmt.Errorf("unknown voice %q (ls voice lists them)", name)
		}
		line, err := json.Marshal(struct {
			Voice string    `json:"voice"`
			Wave  string    `json:"wave"`
			Genes []float64 `json:"genes"`
		}{name, cmp.Or(v.Wave, "sine"), v.Genes()})
		if err != nil {
			return err
		}
		fmt.Fprintf(s.out, "%s\n", line)
	}
	return nil
}

// fitsManifest is a fits directory's fits.json: the sizes its fits ran at, the
// checksum of the weight files its rips heard through, or that it fitted the
// labels' own notes and ripped nothing; a directory holds one kind
type fitsManifest struct {
	Version     int    `json:"version"`
	Notes       int    `json:"notes"`
	Pool        int    `json:"pool"`
	Generations int    `json:"generations"`
	Seed        uint64 `json:"seed"`
	Weights     string `json:"weights"`
	Given       bool   `json:"given,omitempty"`
}

const fitsFile = "fits.json"

// fitRow is one line's fit read against its loop's label. A distance is the
// fit's measure of a voice on the notes the fit listened to, or held on the
// line's other notes; a voice's genes are its own less the label's, far their
// RMS. Null where the label, the front end or the line gives none.
type fitRow struct {
	Loop       string    `json:"loop"`
	Line       string    `json:"line"`
	Notes      int       `json:"notes"`
	Listened   int       `json:"listened"`
	Held       int       `json:"held"`
	Matched    int       `json:"matched"` // the line's notes a labelled note answers
	Voice      string    `json:"voice"`   // the label's voice that plays most of them
	Share      float64   `json:"share"`   // its share of the labelled notes they match
	Family     string    `json:"family"`
	Wave       string    `json:"wave"`
	Fitted     string    `json:"fitted"`
	Heard      *float64  `json:"heard"` // the voice the front end heard
	Seed       *float64  `json:"seed"`  // the best seed
	Fit        *float64  `json:"fit"`
	Own        *float64  `json:"own"` // the label's own voice
	HeardHeld  *float64  `json:"heard_held"`
	SeedHeld   *float64  `json:"seed_held"`
	FitHeld    *float64  `json:"fit_held"`
	OwnHeld    *float64  `json:"own_held"`
	HeardGenes []float64 `json:"heard_genes"`
	SeedGenes  []float64 `json:"seed_genes"`
	FitGenes   []float64 `json:"fit_genes"`
	HeardFar   *float64  `json:"heard_far"`
	SeedFar    *float64  `json:"seed_far"`
	FitFar     *float64  `json:"fit_far"`
}

func cmdFits(s *Session, a []string) error {
	usage := errors.New("want fits <dir> from <corpus dir> [given] [part k/n] [notes <n>] [pool <n>] [generations <n>] [seed <n>]")
	if a[1] != "from" {
		return usage
	}
	o, part, parts, given := s.fit, 0, 0, false
	for w := a[3:]; len(w) > 0; w = w[2:] {
		if w[0] == "given" {
			given, w = true, w[1:]
			if len(w) == 0 {
				break
			}
		}
		if len(w) < 2 {
			return usage
		}
		var err error
		if w[0] == "part" {
			part, parts, err = parsePart(w[1])
		} else {
			err = o.size(w[0], w[1])
		}
		if err != nil {
			return err
		}
	}
	return s.fits(s.path(a[0]), s.path(a[2]), o, part, parts, given)
}

// fits reads the voice fit over a corpus directory's loops that kept their
// audio: each is ripped and its lines fitted as rip does it, at o's sizes, or
// given, each voice its label states fitted on its own notes, and a row a line
// fitted (fitRows) goes to <dir>/<the loop's checksum>.json. A loop on disk is
// kept, parts taking the loops i%parts == part; the last lines count them all.
func (s *Session) fits(dir, from string, o fitOptions, part, parts int, given bool) error {
	if info, err := os.Stat(from); err != nil {
		return err
	} else if !info.IsDir() {
		return fmt.Errorf("%s: fits reads a corpus directory, whose labels state each note's voice", from)
	}
	var corpus corpusManifest
	if data, err := os.ReadFile(filepath.Join(from, "corpus.json")); err != nil {
		return err
	} else if err := json.Unmarshal(data, &corpus); err != nil {
		return fmt.Errorf("%s: %w", filepath.Join(from, "corpus.json"), err)
	}
	if corpus.VoiceGenes != audio.VoiceGeneCount {
		return fmt.Errorf("%s: its labels state %d genes a voice, and a voice has %d", from, corpus.VoiceGenes, audio.VoiceGeneCount)
	}
	loops, silent, err := truthSources(from)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	// written whole or not there, as parts started together each write a file
	whole := func(file string, v any) error {
		tmp := fmt.Sprintf("%s.%d.part", file, os.Getpid())
		if err := writeJSON(tmp, v); err != nil {
			return err
		}
		return os.Rename(tmp, file)
	}
	m := fitsManifest{Version: 1, Notes: o.notes, Pool: o.pool, Generations: o.generations, Seed: o.seed, Weights: s.earSum, Given: given}
	if given {
		m.Weights = "" // nothing is ripped
	}
	manifest := filepath.Join(dir, fitsFile)
	if data, err := os.ReadFile(manifest); err == nil {
		var was fitsManifest
		if json.Unmarshal(data, &was) != nil || was != m {
			return fmt.Errorf("%s holds %s, and these are %s: a directory holds one kind", dir, was, m)
		}
	} else if err := whole(manifest, m); err != nil {
		return err
	}
	said := s.out
	s.out = io.Discard // a search's generations are rip's and fit's to say
	defer func() { s.out = said }()
	var rows []fitRow
	var taken, kept int
	var failed []string
	for i, l := range loops {
		if parts > 0 && i%parts != part {
			continue
		}
		recording, err := os.ReadFile(l.at(l.wav))
		if err != nil {
			return err
		}
		sum := sha256.Sum256(recording)
		file := filepath.Join(dir, fmt.Sprintf("%x.json", sum[:8]))
		var loop []fitRow
		if data, err := os.ReadFile(file); err == nil {
			if err := json.Unmarshal(data, &loop); err != nil {
				return fmt.Errorf("%s: %w", file, err)
			}
			kept++
		} else {
			var lab corpusLoop
			if data, err := os.ReadFile(l.at(l.truth)); err != nil {
				return err
			} else if err := json.Unmarshal(data, &lab); err != nil {
				return fmt.Errorf("%s: %w", l.truth, err)
			}
			x, rate, err := audio.ReadWAV(bytes.NewReader(recording))
			if err != nil {
				return fmt.Errorf("%s: %w", l.wav, err)
			}
			var lines []fitLine
			cents := lab.Tuning
			if given {
				for _, v := range labelVoices(lab, corpus) {
					lines = append(lines, fitLine{word: v.name, notes: v.notes, voice: v})
				}
			} else {
				r, err := s.ripper().Samples(x, rate)
				if err != nil {
					failed = append(failed, fmt.Sprintf("%s: %v", l.wav, err))
					continue
				}
				for reg, at := range lineVoices(r) {
					lines = append(lines, fitLine{word: ripLines[reg].word, notes: lineNotes(r, reg), heard: *at})
				}
				cents = r.Tuning
			}
			loop = s.fitRows(strings.TrimSuffix(l.wav, ".wav"), x, rate, cents, lines, lab, corpus, o)
			if err := whole(file, loop); err != nil {
				return err
			}
			fmt.Fprintf(said, "%s: %d line(s) fitted\n", l.wav, len(loop))
		}
		taken, rows = taken+1, append(rows, loop...)
	}
	what := fmt.Sprintf("fits of %d loop(s) of %s in %s at %d note(s), %d generation(s) of %d, seed %d: %d line(s)", taken, from, dir, o.notes, o.generations, o.pool, o.seed, len(rows))
	if given {
		what += ", each a voice its label states, on its notes"
	}
	if kept > 0 {
		what += fmt.Sprintf("; %d loop(s) were there", kept)
	}
	if silent > 0 {
		what += fmt.Sprintf("; %d loop(s) kept no audio", silent)
	}
	if parts > 0 {
		what += fmt.Sprintf("; part %d of %d", part, parts)
	}
	fmt.Fprintln(said, what)
	if len(failed) > 0 {
		fmt.Fprintf(said, "no rip of %d, as %s\n", len(failed), failed[0])
	}
	fitCounts(said, rows, corpus.Waves)
	return nil
}

// String names the kind of fits m holds
func (m fitsManifest) String() string {
	of := fmt.Sprintf("through the weights %q", m.Weights)
	if m.Given {
		of = "on the labels' notes"
	}
	return fmt.Sprintf("fits of %d note(s), %d generation(s) of %d, seed %d, %s", m.Notes, m.Generations, m.Pool, m.Seed, of)
}

// labelVoice is a voice a label's notes state: its instrument's name, its wave
// and family by the manifest's names, its genes where stated, and its notes
type labelVoice struct {
	name, wave, family string
	genes              []float64
	notes              []audio.RipNote
}

// labelVoices are the voices a label's notes were played with, in the order
// the label first states each
func labelVoices(lab corpusLoop, m corpusManifest) []*labelVoice {
	named := func(names []string, i int) string {
		if i < 0 || i >= len(names) {
			return ""
		}
		return names[i]
	}
	var out []*labelVoice
	for _, e := range lab.Events {
		if e.Drum >= 0 || e.MIDI <= 0 {
			continue
		}
		wave, family := named(m.Waves, e.Wave), named(m.Families, e.Family)
		i := slices.IndexFunc(out, func(v *labelVoice) bool {
			return v.name == e.Instr && v.wave == wave && v.family == family && slices.Equal(v.genes, e.Voice)
		})
		if i < 0 {
			i, out = len(out), append(out, &labelVoice{name: e.Instr, wave: wave, family: family, genes: e.Voice})
		}
		out[i].notes = append(out[i].notes, audio.RipNote{Start: float64(e.Frame) * audio.RipFrameSec, End: float64(e.End) * audio.RipFrameSec, MIDI: e.MIDI, Vel: e.Vel})
	}
	return out
}

// lineOwner is whose line a rip's line is: the label's voice that plays most
// of its notes that match a labelled note on the loop's clock, that voice's
// share of the labelled notes they match, a note two voices double counted for
// each, and how many of the line's notes match any; nil where none does
func lineOwner(notes []audio.RipNote, voices []*labelVoice, given []audio.RipNote) (owner *labelVoice, share float64, matched int) {
	var most, all int
	for _, v := range voices {
		n := audio.MatchNotes(notes, v.notes, true).All.Matched
		if all += n; n > most {
			owner, most = v, n
		}
	}
	if owner == nil {
		return nil, 0, 0
	}
	return owner, float64(most) / float64(all), audio.MatchNotes(notes, given, true).All.Matched
}

// fitLine is a line fits fits: a rip's, with the voice its front end heard, or
// a label's voice and its notes
type fitLine struct {
	word  string
	notes []audio.RipNote
	heard *audio.VoiceDef
	voice *labelVoice
}

// fitRows fits each line as rip does, at o's sizes and the tuning in cents,
// and reads it against the loop's label: a row a line fitted
func (s *Session) fitRows(loop string, x []float64, rate int, cents float64, lines []fitLine, lab corpusLoop, m corpusManifest, o fitOptions) []fitRow {
	finite := func(d float64) *float64 {
		if math.IsNaN(d) || math.IsInf(d, 0) {
			return nil
		}
		return &d
	}
	// a voice's distance as the search scores a seed, through its genome
	near := func(v *audio.VoiceDef, targets []fitTarget) *float64 {
		if v == nil || len(targets) == 0 {
			return nil
		}
		if v = voiceFromGenome(voiceGenome(v)); audio.ValidateVoice(v) != nil {
			return nil
		}
		return finite(fitDistance(v, targets, cents))
	}
	apart := func(v *audio.VoiceDef, label []float64) ([]float64, *float64) {
		if v == nil || label == nil {
			return nil, nil
		}
		genes, sum := v.Genes(), 0.0
		for i, g := range label {
			genes[i] -= g
			sum += genes[i] * genes[i]
		}
		return genes, finite(math.Sqrt(sum / float64(len(genes))))
	}
	voices, given := labelVoices(lab, m), lab.notes()
	rows := []fitRow{}
	for _, l := range lines {
		f := s.fitVoice(x, rate, cents, l.notes, l.heard, o)
		if f == nil {
			continue
		}
		// the fit's own choice of notes, in its order: those it heard, then the rest
		targets := fitTargets(x, rate, l.notes, len(l.notes))
		listened, held := targets[:min(max(o.notes, 1), len(targets))], targets[min(max(o.notes, 1), len(targets)):]
		row := fitRow{Loop: loop, Line: l.word, Notes: len(l.notes), Listened: len(listened), Held: len(held), Fitted: cmp.Or(f.voice.Wave, "sine"),
			Heard: near(l.heard, listened), Seed: finite(f.from), Fit: finite(f.to),
			HeardHeld: near(l.heard, held), SeedHeld: near(f.seed, held), FitHeld: near(f.voice, held)}
		if owner, share, matched := lineOwner(l.notes, voices, given); owner != nil {
			if l.voice != nil { // a voice's own notes are its line, whatever doubles them
				owner = l.voice
			}
			row.Matched, row.Voice, row.Share, row.Family, row.Wave = matched, owner.name, share, owner.family, owner.wave
			if owner.genes != nil {
				own := audio.VoiceFromGenes(owner.wave, owner.genes)
				row.Own, row.OwnHeld = near(own, listened), near(own, held)
				row.HeardGenes, row.HeardFar = apart(l.heard, owner.genes)
				row.SeedGenes, row.SeedFar = apart(f.seed, owner.genes)
				row.FitGenes, row.FitFar = apart(f.voice, owner.genes)
			}
		}
		rows = append(rows, row)
	}
	return rows
}

// fitCounts prints the rows counted, the lines of one voice alone and the rest
// apart, each whole and by its label's wave: how often the fitted wave is the
// label's, how often the label's own voice and how often the fit is the nearer
// on the notes listened to and held out, each distance's median, and how often
// the fit ends nearer the label's genes than the voice heard and the best seed
func fitCounts(w io.Writer, rows []fitRow, waves []string) {
	of := func(k, n int) string {
		if n == 0 {
			return "-"
		}
		return fmt.Sprintf("%d/%d", k, n)
	}
	median := func(v []float64) string {
		if len(v) == 0 {
			return "-"
		}
		slices.Sort(v)
		return fmt.Sprintf("%.3f", (v[(len(v)-1)/2]+v[len(v)/2])/2)
	}
	unowned := 0
	for _, r := range rows {
		if r.Matched == 0 {
			unowned++
		}
	}
	fmt.Fprintf(w, "%d line(s) match no labelled note and are left out. right: the fitted wave is the label's; short: the label's own voice is nearer the recording than the fit, the search stopped short; prefers: the fit is nearer than the label's own voice, the measure prefers another; heard, seed, fit, own: the median distance of the voice the front end heard, the best seed, the fit and the label's own voice; genes nearer than heard, seed: the fit ends nearer the label's genes than the voice heard or the best seed began\n", unowned)
	fmt.Fprintf(w, "%-32s | %-47s | %-47s | %-19s\n", "", "on the notes the fit listened to", "on the line's notes held out", "genes nearer than")
	fmt.Fprintf(w, "%-9s %-6s %5s %9s | %9s %9s %6s %6s %6s %6s | %9s %9s %6s %6s %6s %6s | %9s %9s\n", "lines", "wave", "rows", "right", "short", "prefers", "heard", "seed", "fit", "own", "short", "prefers", "heard", "seed", "fit", "own", "heard", "seed")
	for _, alone := range [...]bool{true, false} {
		for _, wave := range append([]string{"all"}, append(slices.Clone(waves), "")...) {
			var n, right, waved int
			var short, prefers, owned, nearer, both [2]int
			var dist [2][4][]float64
			for _, r := range rows {
				if r.Matched == 0 || (r.Share == 1) != alone || (wave != "all" && r.Wave != wave) {
					continue
				}
				n++
				if r.Wave != "" {
					waved++
					if r.Fitted == r.Wave {
						right++
					}
				}
				for h, d := range [2][4]*float64{{r.Heard, r.Seed, r.Fit, r.Own}, {r.HeardHeld, r.SeedHeld, r.FitHeld, r.OwnHeld}} {
					for k, v := range d {
						if v != nil {
							dist[h][k] = append(dist[h][k], *v)
						}
					}
					if fit, own := d[2], d[3]; fit != nil && own != nil {
						owned[h]++
						switch {
						case *own < *fit:
							short[h]++
						case *fit < *own:
							prefers[h]++
						}
					}
				}
				for k, began := range [2]*float64{r.HeardFar, r.SeedFar} {
					if began != nil && r.FitFar != nil {
						if both[k]++; *r.FitFar < *began {
							nearer[k]++
						}
					}
				}
			}
			if n == 0 {
				continue
			}
			lines := "mixed"
			if alone {
				lines = "one voice"
			}
			fmt.Fprintf(w, "%-9s %-6s %5d %9s", lines, cmp.Or(wave, "none"), n, of(right, waved))
			for h := range dist {
				fmt.Fprintf(w, " | %9s %9s", of(short[h], owned[h]), of(prefers[h], owned[h]))
				for k := range dist[h] {
					fmt.Fprintf(w, " %6s", median(dist[h][k]))
				}
			}
			fmt.Fprintf(w, " | %9s %9s\n", of(nearer[0], both[0]), of(nearer[1], both[1]))
		}
	}
}
