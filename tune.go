package stapes

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/csv"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/lixenwraith/audio"
	"github.com/lixenwraith/genetic"
)

// tuneOptions sizes a search: the loops rendered and their bars, the generations
// run over a pool of candidates, the seed every draw follows, whether the loops
// are augmented, a room, a mix and random voices behind each, and whether the
// arranger writes them in place of the bank's patterns
type tuneOptions struct {
	loops, bars, generations, pool int
	seed                           uint64
	aug, gen                       bool
	glide                          bool           // a share of the arranged notes bend into the next or scoop, and of the voices sway
	meters                         bool           // the loops in 2/4 to 12/8 by corpusMeters' shares, and three in ten starting part-way into a bar
	overlaps                       float64        // the share of the loops of single notes overlapping and no beat; none at 0
	steady                         bool           // the arranged loops keep their tempo and start on a bar line, as a rip's score asks
	nowav                          bool           // an export keeps the frames and the labels only
	codec                          bool           // a third of the loops go through a lossy codec and back
	labels                         bool           // an export writes each loop's label over the one on disk and renders nothing
	tempos                         []int          // the tempos the loops cycle through; tuneTempos when nil
	part, parts                    int            // an export renders the loops i%parts == part; all when parts is 0
	skip                           func(int) bool // the loops drawn but not rendered
	samples                        *sampleSet     // recorded notes some tonal parts play on; none when nil
	guard                          float64        // the F1 a set tune truth tries may take from any condition's before it is refused
	frames                         string         // where tune truth keeps each recording's frames between runs; "" hears them every run
}

// tuneLoop is one loop, rendered unless an export writes labels alone: what
// made it, its label, the rip a perfect listener makes of one that starts on a
// bar line, and the voices its tonal parts were played with
type tuneLoop struct {
	audio    []float64
	want     *audio.Rip
	label    corpusLoop
	slots    [audio.MusicSlots]audio.PatternID // a bank loop's patterns
	texture  string                            // the parts an arranged loop plays, bank or overlap
	bpm      int
	meter    audio.Meter
	root     int
	scale    audio.ScaleID
	voices   map[string]*audio.VoiceDef
	families map[string]int // each voice's family, an index into audio.RipFamilies
	tuning   float64        // cents off A440 the loop was rendered at
	tonal    int            // its tonal parts
	sampled  int            // those played on recorded notes
}

// tempos spread the corpus over the sequencer's range
var tuneTempos = []int{96, 112, 128, 140, 156, 176}

// The level a loop is laid at: its loudest half second's RMS drawn evenly
// between loopQuiet and loopLoud dBFS, its peak held at loopPeak
const (
	loopQuiet  = -32.0
	loopLoud   = -14.0
	loopPeak   = -1.0
	loopWindow = 0.5 // seconds a loop's level is measured over, its loudest window
)

func cmdTune(s *Session, a []string) error {
	o := tuneOptions{loops: 12, bars: 4, generations: 8, pool: 16, seed: 1, guard: 0.01}
	generations := func(w string) error {
		n, err := strconv.Atoi(w)
		if err != nil || n < 1 || n > 200 {
			return fmt.Errorf("generations %q: want 1 to 200", w)
		}
		o.generations = n
		return nil
	}
	usage := errors.New("want tune [generations] [file], or tune [pool] [generations] truth <list> [guard <x>] [frames <dir>]")
	if i := slices.Index(a, "truth"); i >= 0 {
		if i > 2 || i+1 >= len(a) {
			return usage
		}
		for w := a[i+2:]; len(w) > 0; w = w[2:] {
			switch {
			case len(w) > 1 && w[0] == "guard":
				g, err := strconv.ParseFloat(w[1], 64)
				if err != nil || !(g >= 0 && g <= 1) {
					return fmt.Errorf("guard %q: want 0 to 1, the F1 a set may take from a condition", w[1])
				}
				o.guard = g
			case len(w) > 1 && w[0] == "frames":
				o.frames = s.path(w[1])
			default:
				return usage
			}
		}
		if i > 0 {
			n, err := strconv.Atoi(a[0])
			if err != nil || n < 2 || n > 256 {
				return fmt.Errorf("pool %q: want 2 to 256", a[0])
			}
			o.pool = n
		}
		if i > 1 {
			if err := generations(a[1]); err != nil {
				return err
			}
		}
		if s.frontEnd == nil {
			return fmt.Errorf("tune truth tunes the models' decoders, and no %s is loaded", frameFile)
		}
		return s.tuneTruth(o, s.path(a[i+1]), "", s.frontEnd)
	}
	if len(a) > 2 {
		return usage
	}
	if len(a) > 0 {
		if err := generations(a[0]); err != nil {
			return err
		}
	}
	file := ""
	if len(a) > 1 {
		file = s.path(a[1])
	}
	return s.tune(o, file)
}

// corpus renders the loops, each with its expected rip, all held at once: what
// a search over them wants
func (s *Session) corpus(o tuneOptions) ([]tuneLoop, error) {
	loops := make([]tuneLoop, 0, o.loops)
	err := s.eachLoop(o, func(_ int, l tuneLoop) error {
		loops = append(loops, l)
		return nil
	})
	return loops, err
}

// eachLoop makes the corpus's loops one at a time to visit, so an export of
// thousands holds one in memory: of the registered rhythm, melody and bass
// patterns, drawn by the seed over the tempos and the keys, or arranged, or
// single notes overlapping. Each word draws from a stream of its own, so a loop
// it leaves alone is the same with it and without.
func (s *Session) eachLoop(o tuneOptions, visit func(int, tuneLoop) error) error {
	var byRole [audio.MusicSlots][]audio.PatternID
	for _, p := range audio.RegisteredPatterns() {
		if strings.HasPrefix(p.Name, "corpus_") {
			continue // a part of an earlier loop, not the bank's
		}
		if p.ID == audio.PatternMelodyGen {
			continue // the sequencer writes its lead bar by bar as it plays: no label reads it
		}
		switch p.Role {
		case audio.RoleRhythm, audio.RoleMelody, audio.RoleBass:
			byRole[p.Role-1] = append(byRole[p.Role-1], p.ID)
		}
	}
	bank := len(byRole[0]) > 0 && len(byRole[1]) > 0 && len(byRole[3]) > 0
	if !o.gen && !bank {
		return errors.New("a corpus without gen needs registered rhythm, melody and bass patterns (builtin pattern, apply)")
	}
	rng := rand.New(rand.NewPCG(o.seed, 0x5bd1e995))
	for i := range o.loops {
		var slots [audio.MusicSlots]audio.PatternID
		for slot, pool := range byRole {
			if len(pool) > 0 && bank {
				slots[slot] = pool[rng.IntN(len(pool))]
			}
		}
		tempos := tuneTempos
		if len(o.tempos) > 0 {
			tempos = o.tempos
		}
		bpm := tempos[i%len(tempos)]
		root := audio.DefaultRootNote + rng.IntN(12) - 6
		scale := audio.ScaleID(rng.IntN(int(audio.ScaleCount)))
		if o.skip != nil && o.skip(i) {
			continue // drawn all the same, so the loops after it are an unbroken run's
		}
		stream := func(salt uint64) *rand.Rand { return rand.New(rand.NewPCG(o.seed^salt, uint64(i)+1)) }
		var srng *rand.Rand
		if o.samples != nil {
			srng = stream(0x3c3c3c3c)
		}
		var (
			m   *music
			err error
		)
		orng := stream(0x6f766572)
		a := newArranger(stream(0x5a5a5a5a), stream(0x3e7e3e7e), o.bars, bpm, o.meters, !o.steady)
		switch {
		case o.overlaps > 0 && orng.Float64() < o.overlaps:
			m, err = s.overlap(orng, o.samples)
		case o.gen || a.m != (audio.Meter{}) || a.bars != o.bars: // the bank is 4/4 from its bar line
			var glrng *rand.Rand
			if o.glide {
				glrng = stream(0x676c6964)
			}
			m, err = s.arrange(a, o.samples, srng, glrng, o.steady)
		default:
			m = bankMusic(slots, bpm, o.bars, root, scale)
		}
		if err != nil {
			return err
		}
		l, err := s.play(o, i, m, srng)
		if err != nil {
			return err
		}
		l.slots = slots
		if err := visit(i, l); err != nil {
			return err
		}
	}
	return nil
}

// bankMusic is a bank loop: the slots' patterns for bars bars at bpm, a part a track
func bankMusic(slots [audio.MusicSlots]audio.PatternID, bpm, bars, root int, scale audio.ScaleID) *music {
	m := &music{kind: "bank", root: root, scale: scale}
	for range bars {
		m.timeline, m.heads = append(m.timeline, slots), append(m.heads, audio.BarHead{BPM: bpm})
	}
	for slot, id := range slots {
		if p := audio.GetPattern(id); p != nil {
			for _, tr := range p.Tracks {
				m.parts[slot] = append(m.parts[slot], &part{kind: &bankKinds[slot], instr: tr.Instr})
			}
		}
	}
	starts := audio.BeatStarts(m.heads)
	m.to = starts[len(starts)-1]
	return m
}

// bankKinds are a bank loop's parts' kinds, by slot, named by the slot's role
var bankKinds = [audio.MusicSlots]partKind{{name: slotRoles[0]}, {name: slotRoles[1], slot: 1}, {name: slotRoles[2], slot: 2}, {name: slotRoles[3], slot: 3}}

// play is a loop of m as the label tells it and, unless o writes labels
// alone, as it sounds: its tonal parts on the engine's voices, under aug of
// random genes, or on the recorded notes where those have them, mixed, cut to
// the loop, through a room, a mix and a codec where the words ask, at its level
func (s *Session) play(o tuneOptions, i int, m *music, srng *rand.Rand) (tuneLoop, error) {
	stream := func(salt uint64) *rand.Rand { return rand.New(rand.NewPCG(o.seed^salt, uint64(i)+1)) }
	events := audio.ExpectedEvents(m.timeline, m.heads, m.root, m.scale, m.prog)
	slices.SortStableFunc(events, func(x, y audio.RipEvent) int { return cmp.Compare(x.At, y.At) })
	own := map[*part][]int{}
	for k, e := range events {
		p := m.parts[e.Slot][e.Track]
		own[p] = append(own[p], k)
	}
	notes := func(p *part) []audio.RipEvent {
		var out []audio.RipEvent
		for _, k := range own[p] {
			out = append(out, events[k])
		}
		return out
	}
	l := tuneLoop{texture: m.kind, bpm: m.heads[0].BPM, meter: m.heads[0].Meter, root: m.root, scale: m.scale}
	var instrs []audio.InstrumentType
	var textures []string
	for slot, ps := range m.parts {
		for _, p := range ps {
			if !slices.Contains(textures, p.kind.name) {
				textures = append(textures, p.kind.name)
			}
			if p.instr.IsDrum() || len(own[p]) == 0 {
				continue
			}
			l.tonal++
			if set := o.samples; set != nil && m.kind == "bank" && slot > 0 && srng.Float64() < set.share {
				pool := set.other
				if slot == audio.MusicSlots-1 {
					pool = set.bass
				}
				for range sampledDraws * min(len(pool), 1) {
					in := pool[srng.IntN(len(pool))]
					if ok, err := set.fits(in, notes(p)); err != nil {
						return l, err
					} else if ok {
						p.sample = in
						break
					}
				}
			} else if p.sample != nil {
				if ok, err := set.fits(p.sample, notes(p)); err != nil {
					return l, err
				} else if !ok {
					p.sample = nil // the engine's instrument drawn beside it plays it
				}
			}
			if p.sample != nil {
				l.sampled++
			} else {
				instrs = append(instrs, p.instr)
			}
		}
	}
	if m.kind == "arranged" {
		l.texture = strings.Join(textures, "+")
	}
	arng, sway := stream(0xa5a5a5a5), (*rand.Rand)(nil)
	if o.glide {
		sway = stream(0x73776179)
	}
	voices, families, restore := s.loopVoices(instrs, o.aug, arng, sway, !o.labels)
	defer restore()
	l.voices, l.families = voices, families
	if o.aug {
		l.tuning = arng.Float64()*100 - 50 // a recording's pitch standard, within a quarter tone of A440
	}
	starts := audio.BeatStarts(m.heads)
	n := starts[len(starts)-1]
	ends := map[int]int{}
	var lines [][]float64
	pedal := srng != nil && srng.Float64() < 1.0/3
	cache := map[sampleKey][]float64{}
	for _, ps := range m.parts {
		for _, p := range ps {
			if p.sample == nil {
				continue
			}
			at, x, err := o.samples.play(p.sample, notes(p), n, l.tuning, pedal && p.kind.name != "bass" && p.kind.name != slotRoles[3], cache, !o.labels)
			if err != nil {
				return l, err
			}
			for j, k := range own[p] {
				ends[k] = at[j]
			}
			if x != nil {
				g := sampledLevel * math.Pow(10, (8*srng.Float64()-4)/20)
				for j := range x {
					x[j] *= g
				}
				lines = append(lines, x)
			}
		}
	}
	l.label = m.label(events, ends, voices, families, l.tuning, o.bars)
	if o.labels { // a label holds what was drawn, and nothing of how it sounds
		return l, nil
	}
	played := m.played()
	s.eng.SetTuning(l.tuning)
	buf, err := s.eng.RenderBars(played, m.heads, m.root, m.scale, m.prog, int64(i+1), nil)
	s.eng.SetTuning(0)
	if err != nil {
		return l, err
	}
	engine := slices.ContainsFunc(buf, func(v float64) bool { return v != 0 })
	for _, x := range lines {
		for j := range buf {
			buf[j] += x[j]
		}
	}
	buf = slices.Clip(buf[m.from:m.to])
	if !engine && len(lines) > 0 {
		air(buf, stream(0x61697221))
	}
	if o.aug {
		buf = degrade(buf, arng)
	}
	if o.codec {
		if buf, err = codecRoundTrip(buf, audio.AudioSampleRate, stream(0xc0dec)); err != nil {
			return l, err
		}
	}
	level(buf, stream(0x6c65766c))
	l.audio = buf
	if m.from == 0 && m.kind != "overlap" {
		l.want = audio.ExpectedRipBars(m.timeline, m.heads, m.root, m.scale, m.prog)
	}
	return l, nil
}

// sampledLevel is the RMS a recorded note's loudest window is played at under
// full velocity, near the engine's voices' own, each part 4 dB either side
const sampledLevel = 0.25

// played is the timeline the engine renders: a pattern with a part the recorded
// notes play is cloned under corpus_played_<slot>_<n>, that part's notes taken out
func (m *music) played() [][audio.MusicSlots]audio.PatternID {
	out := slices.Clone(m.timeline)
	for slot, ps := range m.parts {
		if !slices.ContainsFunc(ps, func(p *part) bool { return p.sample != nil }) {
			continue
		}
		clones := map[audio.PatternID]audio.PatternID{}
		for b := range out {
			id := out[b][slot]
			c, ok := clones[id]
			if src := audio.GetPattern(id); !ok && src != nil {
				cp := *src
				cp.ID, cp.Name, cp.Tracks = audio.PatternSilence, fmt.Sprintf("corpus_played_%d_%d", slot, len(clones)), slices.Clone(src.Tracks)
				for t := range cp.Tracks {
					if ps[t].sample != nil {
						cp.Tracks[t].Events = nil
					}
				}
				c = audio.RegisterPattern(&cp)
				clones[id] = c
			}
			out[b][slot] = c
		}
	}
	return out
}

// label is a loop's label: each event where it sounds in the loop's frames, by
// its part and its instrument, a note with the wave and the genes it was played
// with, or the recorded instrument's family and where the recording let it go;
// its beats and bar lines where the render puts them, and an arranged loop's chords
func (m *music) label(events []audio.RipEvent, ends map[int]int, voices map[string]*audio.VoiceDef, families map[string]int, tuning float64, bars int) corpusLoop {
	frame := func(at int) int { return int(float64(at-m.from)/audio.AudioSampleRate/audio.RipFrameSec + 0.5) }
	frames := loopFrames(m.to - m.from)
	lab := corpusLoop{BPM: m.heads[0].BPM, Root: m.root, Scale: int(m.scale), Bars: bars, Meter: m.heads[0].Meter.Doc(), Frames: frames, Tuning: tuning, Kind: m.kind}
	if m.kind == "overlap" { // no beat, no bar and no key: as a recording's label with no beat track
		lab.BPM, lab.Root, lab.Scale, lab.Bars = 0, 0, 0, 0
	}
	for k, e := range events {
		if e.At < m.from || e.At >= m.to {
			continue
		}
		p := m.parts[e.Slot][e.Track]
		ev := corpusEvent{Frame: frame(e.At), Drum: -1, Vel: e.Vel, Wave: -1, Family: -1, Part: p.kind.name, Instr: audio.InstrumentName(e.Instr)}
		if e.Instr.IsDrum() {
			if e.Instr >= audio.InstrClap {
				continue
			}
			ev.Drum, ev.End, ev.Family = int(e.Instr), ev.Frame, familyIndex("drums")
		} else {
			ev.MIDI = e.MIDI
			ev.End = frame(e.At + e.For)
			if p.sample != nil {
				ev.End, ev.Family, ev.Instr = frame(ends[k]), p.sample.family, p.sample.name
			} else if v := voices[ev.Instr]; v != nil {
				ev.Wave = max(slices.Index(audio.VoiceWaves[:], cmp.Or(v.Wave, "sine")), 0)
				ev.Voice, ev.Family = v.Genes(), families[ev.Instr]
				start := float64(e.At-m.from) / audio.AudioSampleRate
				ev.Cents = noteCents(v, e, start, float64(e.For)/audio.AudioSampleRate, tuning, ev.Frame, ev.End)
			}
		}
		if ev.Frame < frames {
			lab.Events = append(lab.Events, ev)
		}
	}
	if m.kind == "overlap" {
		return lab
	}
	starts, per := audio.BeatStarts(m.heads), m.heads[0].Meter.Beats()
	for k, at := range starts[:len(starts)-1] {
		if f := frame(at); at >= m.from && at < m.to && f < frames {
			if lab.Beats = append(lab.Beats, f); k%per == 0 {
				lab.Downbeats = append(lab.Downbeats, f)
			}
		}
	}
	for b, tones := range m.chords {
		at := starts[b*per]
		if at >= m.to {
			break
		}
		f := max(frame(at), 0)
		if c := len(lab.Chords); c > 0 && slices.Equal(lab.Chords[c-1].Tones, tones) {
			continue
		} else if c > 0 && lab.Chords[c-1].Frame == f {
			lab.Chords = lab.Chords[:c-1]
		}
		lab.Chords = append(lab.Chords, corpusChord{Frame: f, Tones: tones})
	}
	return lab
}

// loopFrames is the frames BandFrames makes of n samples at the engine's rate
func loopFrames(n int) int {
	return (n*audio.RipRate/audio.AudioSampleRate + audio.RipHop - 1) / audio.RipHop
}

// loudest is the RMS of x's loudest window of loopWindow, its hop half that
func loudest(x []float64) float64 {
	w := min(int(loopWindow*audio.AudioSampleRate), len(x))
	var best float64
	for i := 0; i+w <= len(x); i += max(w/2, 1) {
		var e float64
		for _, v := range x[i : i+w] {
			e += v * v
		}
		best = max(best, math.Sqrt(e/float64(max(w, 1))))
	}
	return best
}

// level lays a loop at a level drawn whatever its parts: its loudest window's
// RMS between loopQuiet and loopLoud dBFS, lowered where its peak would pass
// loopPeak, and a peak that would hold it under loopQuiet bent down to it
func level(x []float64, rng *rand.Rand) {
	r := loudest(x)
	if r == 0 {
		return
	}
	var peak float64
	for _, v := range x {
		peak = max(peak, math.Abs(v))
	}
	top := math.Pow(10, loopPeak/20)
	g := min(math.Pow(10, (loopQuiet+(loopLoud-loopQuiet)*rng.Float64())/20)/r, top/peak)
	floor := math.Pow(10, loopQuiet/20) / r
	bend := g < floor
	g = max(g, floor)
	knee := 0.7 * top
	for i, v := range x {
		v *= g
		if a := math.Abs(v); bend && a > knee {
			v = math.Copysign(knee+(top-knee)*math.Tanh((a-knee)/(top-knee)), v)
		}
		x[i] = v
	}
}

// air lays a recording's noise floor over a loop of recorded notes alone,
// which hold nothing over their 8 kHz: white noise over 7.5 kHz, 45 to 65 dB
// under the loop's loudest window
func air(x []float64, rng *rand.Rand) {
	g := loudest(x) * math.Pow(10, -(45+20*rng.Float64())/20)
	a := math.Exp(-2 * math.Pi * sampledTop / audio.AudioSampleRate)
	var lp1, lp2 float64
	for i := range x {
		w := rng.NormFloat64()
		lp1 = a*lp1 + (1-a)*w
		lp2 = a*lp2 + (1-a)*lp1
		x[i] += g * (w - lp2) * 1.2
	}
}

// loopVoices is the voice each tonal instrument plays with and the family it
// sounds as: the registered one, or under aug one of random genes near it, a
// third of the time on another wave, a third of the loops under a piano's pedal;
// sway gives three in ten a vibrato; restore puts the registered ones back. A
// loop not rendered registers none: the mixer refuses one past its queue.
func (s *Session) loopVoices(instrs []audio.InstrumentType, aug bool, rng, sway *rand.Rand, render bool) (map[string]*audio.VoiceDef, map[string]int, func()) {
	voices, families, known := map[string]*audio.VoiceDef{}, map[string]int{}, familyVoices()
	var originals []*audio.VoiceDef
	pedal := 1.0 // the labels stay the notes as played, however long they ring
	if aug && rng.Float64() < 1.0/3 {
		pedal = 3 + rng.Float64()*7
	}
	for _, instr := range instrs {
		if instr.IsDrum() {
			continue
		}
		name := audio.InstrumentName(instr)
		v := audio.VoiceDefByName(name)
		if v == nil || voices[name] != nil {
			continue
		}
		voices[name], families[name] = v, voiceFamily(known, name, v)
		play, family := v, families[name]
		if aug {
			genes := v.Genes()
			for i := range genes {
				genes[i] = min(max(genes[i]+rng.NormFloat64()*0.15, 0), 1)
			}
			wave := cmp.Or(v.Wave, "sine")
			if rng.Float64() < 0.3 {
				wave = audio.VoiceWaves[rng.IntN(len(audio.VoiceWaves))]
			}
			r := audio.VoiceFromGenes(wave, genes)
			// measured before the pedal: its long release is the player's, not the instrument's
			heard := voiceFamily(known, name, r)
			if rng.Float64() < 0.15 { // distorted, as a guitar through an amp
				r.Drive, r.Level = 2+rng.Float64()*14, r.Level*0.6
				heard = familyIndex("unknown")
			}
			if instr != audio.InstrBass {
				r.Release = min(r.Release*pedal, 3)
			}
			r.VibratoHz, r.VibratoCents, r.VibratoDelay, r.Vowel = v.VibratoHz, v.VibratoCents, v.VibratoDelay, v.Vowel // not genes; kept from the registered voice
			r.Name = name
			if audio.ValidateVoice(r) == nil {
				play, family = r, heard
			}
		}
		// a string and a modal bank hold their pitch, and a voice's own vibrato stays
		if w := cmp.Or(play.Wave, "sine"); sway != nil && sway.Float64() < 0.3 && w != "string" && w != "modal" && play.VibratoHz == 0 {
			c := *play
			c.VibratoHz, c.VibratoCents, c.VibratoDelay = 4.5+2.5*sway.Float64(), 10+70*sway.Float64(), 0.4*sway.Float64()
			play = &c
		}
		if play == v {
			continue
		}
		if !render {
			if audio.ValidateVoice(play) != nil {
				continue
			}
		} else if _, err := s.eng.DefineVoice(play); err != nil {
			continue
		} else {
			originals = append(originals, v)
		}
		voices[name], families[name] = play, family
	}
	return voices, families, func() {
		for _, v := range originals {
			s.eng.DefineVoice(v)
		}
	}
}

// voiceFamilies is the family each of the engine's voices plays as, by name: the
// bank's and those tool/rip-train/focus adds
var voiceFamilies = map[string]string{
	"piano": "piano", "bass": "bass", "pluck": "guitar", "pad": "synth", "lead": "synth",
	"vocal": "voice", "choir": "voice", "nylon": "guitar", "steel": "guitar", "plucked_bass": "bass",
}

// programFamilies is the family each General MIDI program plays as, a row a run
// of programs up to the next row's first: a `corpus from` note's when its list
// gives the recording none
var programFamilies = []struct {
	from   int
	family string
}{
	{0, "piano"},     // pianos, chromatic percussion
	{16, "synth"},    // organs
	{24, "guitar"},   // guitars
	{32, "bass"},     // basses
	{40, "strings"},  // violin to harp
	{47, "unknown"},  // timpani
	{48, "strings"},  // string ensembles, synth strings
	{52, "voice"},    // choir, voice, synth voice
	{55, "unknown"},  // orchestra hit
	{56, "winds"},    // brass, reeds, pipes
	{80, "synth"},    // synth leads, pads, effects
	{104, "guitar"},  // sitar, banjo, shamisen, koto, kalimba
	{109, "winds"},   // bagpipe
	{110, "strings"}, // fiddle
	{111, "winds"},   // shanai
	{112, "unknown"}, // percussive, sound effects
}

// programFamily is the family a note of a program plays as, -1 for a note whose
// file set no program (audio.RipNote.Program -1)
func programFamily(program int) int {
	for _, r := range slices.Backward(programFamilies) {
		if program >= r.from {
			return familyIndex(r.family)
		}
	}
	return -1
}

// familyReach is the RMS distance of genes within which a voice sounds as a
// registered one of its wave: a redraw lands 0.12 from its own, 0.16 at the 95th
// percentile, and the bank's two voices of a wave lie 0.19 and 0.22 apart
const familyReach = 0.2

// familyIndex is a family's index in audio.RipFamilies, -1 for a name of none
func familyIndex(name string) int { return slices.Index(audio.RipFamilies[:], name) }

// familyVoices is the registered voices voiceFamilies names, read before a loop
// redraws any under its name
func familyVoices() map[string]*audio.VoiceDef {
	known := map[string]*audio.VoiceDef{}
	for name := range voiceFamilies {
		if v := audio.VoiceDefByName(name); v != nil {
			known[name] = v
		}
	}
	return known
}

// voiceFamily is the family a voice played under name sounds as: name's while it
// stays within familyReach of the registered voice of that name on its wave, else
// that of the nearest registered voice within reach on its wave, else unknown
func voiceFamily(known map[string]*audio.VoiceDef, name string, v *audio.VoiceDef) int {
	genes := v.Genes()
	far := func(r *audio.VoiceDef) float64 {
		if cmp.Or(r.Wave, "sine") != cmp.Or(v.Wave, "sine") {
			return math.Inf(1)
		}
		var d float64
		for i, g := range r.Genes() {
			d += (g - genes[i]) * (g - genes[i])
		}
		return math.Sqrt(d / float64(len(genes)))
	}
	if r := known[name]; r != nil && far(r) <= familyReach {
		return familyIndex(voiceFamilies[name])
	}
	best, near := "unknown", familyReach
	for _, n := range slices.Sorted(maps.Keys(known)) {
		if d := far(known[n]); d <= near {
			best, near = voiceFamilies[n], d
		}
	}
	return familyIndex(best)
}

// degrade puts a room and a mix between a render and the frames a model hears,
// each drawn from rng so the model does not learn the synthesizer's clean
// output: a swell of up to 6 dB, for half the loops a low-pass, two equaliser
// bands, for a third a soft clip, for half a compressor, a booth's reverb or for
// a third a hall's further back, and for half a noise floor 40 to 65 dB down
func degrade(x []float64, rng *rand.Rand) []float64 {
	sr := float64(audio.AudioSampleRate)
	out := make([]float64, len(x))
	gain := math.Pow(10, (rng.Float64()*9-6)/20)
	swell, period, phase := rng.Float64()*6, (0.5+rng.Float64()*1.5)*float64(len(x)), rng.Float64()*2*math.Pi
	a := 1.0
	if rng.Float64() < 0.5 {
		a = 1 - math.Exp(-2*math.Pi*(4000+rng.Float64()*7000)/sr)
	}
	drive := 1.0
	if rng.Float64() < 1.0/3 {
		drive = 1.5 + rng.Float64()*2
	}
	var bands [2]func(float64) float64
	for k := range bands {
		bands[k] = audio.Peaking(80*math.Pow(100, rng.Float64()), rng.Float64()*12-6, 0.5+rng.Float64()*1.5)
	}
	var lp float64
	for i, v := range x {
		g := gain * math.Pow(10, swell*math.Sin(2*math.Pi*float64(i)/period+phase)/20)
		lp += a * (v*g - lp)
		out[i] = math.Tanh(drive*bands[1](bands[0](lp))) / math.Tanh(drive)
	}
	if rng.Float64() < 0.5 {
		compress(out, 2+rng.Float64()*4, 3+rng.Float64()*9, sr)
	}
	// a Schroeder reverb: four combs in parallel into two allpasses, at a decay
	// and a mix of the draw's
	rt, wet := 0.15+rng.Float64()*0.55, 0.05+rng.Float64()*0.25
	if rng.Float64() < 1.0/3 {
		rt, wet = 0.7+rng.Float64()*1.8, 0.03+rng.Float64()*0.12
	}
	combs := [4]int{int(0.0297 * sr), int(0.0371 * sr), int(0.0411 * sr), int(0.0437 * sr)}
	var lines [4][]float64
	var feedback [4]float64
	for k, n := range combs {
		lines[k] = make([]float64, n)
		feedback[k] = math.Pow(0.001, float64(n)/sr/rt)
	}
	aps := [2]int{int(0.005 * sr), int(0.0017 * sr)}
	var apLines [2][]float64
	for k, n := range aps {
		apLines[k] = make([]float64, n)
	}
	var rms float64
	for i := range out {
		dry := out[i]
		var sum float64
		for k := range combs {
			j := i % len(lines[k])
			y := lines[k][j]
			lines[k][j] = dry + y*feedback[k]
			sum += y
		}
		sum *= 0.25
		for k := range aps {
			j := i % len(apLines[k])
			buf := apLines[k][j]
			apLines[k][j] = sum + buf*0.5
			sum = buf - sum*0.5
		}
		out[i] = dry + wet*sum
		rms += out[i] * out[i]
	}
	rms = math.Sqrt(rms / float64(max(len(out), 1)))
	// a recording's hiss sits bare over a synthesizer's empty treble: half the loops have none
	noise := 0.0
	if rng.Float64() < 0.5 {
		noise = rms * math.Pow(10, (rng.Float64()*25-65)/20)
	}
	var peak float64
	for i := range out {
		out[i] += rng.NormFloat64() * noise
		peak = max(peak, math.Abs(out[i]))
	}
	if peak > 1 { // the room on top of the clip can pass full scale: turned down, not clipped hard
		for i := range out {
			out[i] /= peak
		}
	}
	return out
}

// compress turns down what rises over the loop's peak less under dB by ratio,
// on an envelope that rises in 5 ms and falls in 120 ms
func compress(x []float64, ratio, under, sr float64) {
	var peak float64
	for _, v := range x {
		peak = max(peak, math.Abs(v))
	}
	threshold := peak * math.Pow(10, -under/20)
	rise, fall := 1-math.Exp(-1/(0.005*sr)), 1-math.Exp(-1/(0.12*sr))
	var env float64
	for i, v := range x {
		if a := math.Abs(v); a > env {
			env += rise * (a - env)
		} else {
			env += fall * (a - env)
		}
		if env > threshold {
			x[i] = v * math.Pow(env/threshold, 1/ratio-1)
		}
	}
}

// corpusEvent is one labelled event of an exported loop, in frames of the rip's
// grid: a drum's hit, or a note with the wave and voice genes it was played with
type corpusEvent struct {
	Frame  int       `json:"frame"`
	End    int       `json:"end"`
	Drum   int       `json:"drum"` // kick 0, snare 1, hihat 2; -1 for a note
	MIDI   int       `json:"midi"`
	Vel    float64   `json:"vel"`    // 0 to 1; -1 where the recording's list says its velocities say nothing (velocityWords)
	Wave   int       `json:"wave"`   // index into the waves of the manifest; -1 unknown
	Family int       `json:"family"` // index into the families of the manifest; -1 unlabelled
	Voice  []float64 `json:"voice,omitempty"`
	Cents  centsList `json:"cents,omitempty"` // a note that moves or a pitch track places: cents from MIDI's cell centre at each frame from Frame to End
	Part   string    `json:"part,omitempty"`  // the line of a rendered loop it plays in
	Instr  string    `json:"instr,omitempty"` // what it sounds on: a voice's name, a drum's, or a recorded instrument's
}

// centsList is where a note sits at each of its frames, NaN at a frame its pitch
// track does not place, which a label writes as null
type centsList []float64

func (c centsList) MarshalJSON() ([]byte, error) {
	out := make([]any, len(c))
	for i, v := range c {
		if !math.IsNaN(v) {
			out[i] = v
		}
	}
	return json.Marshal(out)
}

// corpusLoop is an exported loop's label file
type corpusLoop struct {
	BPM    int           `json:"bpm"`
	Root   int           `json:"root"`
	Scale  int           `json:"scale"`
	Bars   int           `json:"bars"`
	Meter  string        `json:"meter,omitempty"` // the bars' time signature, absent for 4/4
	Frames int           `json:"frames"`
	Tuning float64       `json:"tuning"` // cents off A440 the loop sounds at
	Events []corpusEvent `json:"events"`
	Source string        `json:"source,omitempty"` // the recording a loop was heard from
	Kind   string        `json:"kind,omitempty"`   // a rendered loop's: bank, arranged or overlap
	// its beats and bar lines in frames, where the render puts them or a
	// recording's list names them; a loop of neither, as an overlap, lists none
	Beats     []int         `json:"beats,omitempty"`
	Downbeats []int         `json:"downbeats,omitempty"`
	Chords    []corpusChord `json:"chords,omitempty"`
}

// corpusChord is the chord an arranged loop sounds from a frame on: its pitch
// classes, its root's first
type corpusChord struct {
	Frame int   `json:"frame"`
	Tones []int `json:"tones"`
}

// notes is a label's notes as a truth on the loop's own clock
func (l corpusLoop) notes() []audio.RipNote {
	var out []audio.RipNote
	for _, e := range l.Events {
		if e.Drum < 0 && e.MIDI > 0 {
			out = append(out, audio.RipNote{Start: float64(e.Frame) * audio.RipFrameSec, End: float64(e.End) * audio.RipFrameSec, MIDI: e.MIDI, Vel: e.Vel, Bass: e.MIDI < audio.RipBassTop})
		}
	}
	return out
}

// hits is a label's drum hits as a truth on the loop's own clock
func (l corpusLoop) hits() []audio.RipHit {
	var out []audio.RipHit
	for _, e := range l.Events {
		if e.Drum >= 0 {
			out = append(out, audio.RipHit{Start: float64(e.Frame) * audio.RipFrameSec, Drum: audio.InstrumentType(e.Drum), Vel: e.Vel})
		}
	}
	return out
}

// grid is a label's beats and bar lines in seconds
func (l corpusLoop) grid() (beats, downbeats []float64) {
	for _, f := range l.Beats {
		beats = append(beats, float64(f)*audio.RipFrameSec)
	}
	for _, f := range l.Downbeats {
		downbeats = append(downbeats, float64(f)*audio.RipFrameSec)
	}
	return beats, downbeats
}

// corpusManifest names the corpus's layout for the trainer: its frames are the
// band frames of audio's RipLayout, which the trainer checks before it reads one,
// and a label holds every note's own pitch
type corpusManifest struct {
	Version    int      `json:"version"`
	Loops      int      `json:"loops"`
	Rate       int      `json:"rate"`
	Hop        int      `json:"hop"`
	Layout     int      `json:"layout"`
	Drums      []string `json:"drums"`
	Waves      []string `json:"waves"`
	Families   []string `json:"families"`
	VoiceGenes int      `json:"voice_genes"`
}

// newCorpusManifest is the manifest of a corpus of loops
func newCorpusManifest(loops int) corpusManifest {
	return corpusManifest{Version: 1, Loops: loops, Rate: audio.RipRate, Hop: audio.RipHop, Layout: audio.RipLayout,
		Drums: []string{"kick", "snare", "hihat"}, Waves: audio.VoiceWaves[:], Families: audio.RipFamilies[:], VoiceGenes: audio.VoiceGeneCount}
}

func cmdCorpus(s *Session, a []string) error {
	if len(a) > 2 && a[1] == "from" {
		return s.corpusFrom(s.path(a[0]), s.path(a[2]), a[3:])
	}
	o := tuneOptions{loops: 24, bars: 4, seed: 1}
	for more := true; more && len(a) > 1; {
		n := len(a)
		switch {
		case a[n-1] == "aug":
			o.aug = true
		case a[n-1] == "gen":
			o.gen = true
		case a[n-1] == "glide":
			o.glide = true
		case a[n-1] == "meters":
			o.meters = true
		case a[n-1] == "nowav":
			o.nowav = true
		case a[n-1] == "codec":
			o.codec = true
		case a[n-1] == "labels":
			o.labels = true
		case n > 2 && a[n-2] == "seed":
			v, err := strconv.ParseUint(a[n-1], 10, 64)
			if err != nil {
				return fmt.Errorf("seed %q: want a number", a[n-1])
			}
			o.seed = v
			a = a[:n-1]
		case a[n-1] == "overlaps":
			o.overlaps = overlapShare
		case n > 2 && a[n-2] == "overlaps":
			v, err := strconv.ParseFloat(a[n-1], 64)
			if err != nil || !(v > 0 && v <= 1) {
				return fmt.Errorf("overlaps %q: want the share of the loops, above 0 to 1", a[n-1])
			}
			o.overlaps = v
			a = a[:n-1]
		case n > 2 && a[n-2] == "sampled":
			set, err := loadSampleSet(s.path(a[n-1]))
			if err != nil {
				return err
			}
			o.samples = set
			a = a[:n-1]
		case n > 2 && a[n-2] == "tempos":
			tempos, err := parseTempos(a[n-1])
			if err != nil {
				return err
			}
			o.tempos = tempos
			a = a[:n-1]
		case n > 2 && a[n-2] == "part":
			var err error
			if o.part, o.parts, err = parsePart(a[n-1]); err != nil {
				return err
			}
			a = a[:n-1]
		default:
			more = false
			continue
		}
		a = a[:len(a)-1]
	}
	if len(a) > 3 {
		return fmt.Errorf("%q: not a word of corpus", a[len(a)-1])
	}
	for i, p := range []*int{&o.loops, &o.bars} {
		limit := 65536
		if i == 1 {
			limit = 64
		}
		if len(a) > i+1 {
			n, err := strconv.Atoi(a[i+1])
			if err != nil || n < 1 || n > limit {
				return fmt.Errorf("%q: want 1 to %d", a[i+1], limit)
			}
			*p = n
		}
	}
	return s.exportCorpus(s.path(a[0]), o)
}

// parsePart reads "k/n", the share of a job a run takes: its items i%n == k
func parsePart(arg string) (part, parts int, err error) {
	k, of, _ := strings.Cut(arg, "/")
	part, err1 := strconv.Atoi(k)
	parts, err2 := strconv.Atoi(of)
	if err1 != nil || err2 != nil || part < 0 || part >= parts {
		return 0, 0, fmt.Errorf("part %q: want k/n, k from 0 below n", arg)
	}
	return part, parts, nil
}

// velocityWords is what a list's velocity column may say of a recording's MIDI
// velocities, a word a cell, each the velocity its labels give a note or a hit of
// the MIDI's: an empty cell keeps the MIDI's, and none says they tell nothing of
// how the notes were played, as a lab's one velocity for every note does, so the
// labels give -1, which asks the velocity head nothing. A word is a row here.
var velocityWords = map[string]func(vel float64) float64{
	"":     func(vel float64) float64 { return vel },
	"none": func(float64) float64 { return -1 },
}

// corpusFrom writes a corpus of recordings and the MIDI files they were played
// from, one loop each, listed by a CSV with audio_filename and midi_filename
// columns, and split, duration, family, velocity, pitch and beats when present: the
// frames as heard, the notes as the MIDI plays them, placed by the pitch track, the
// voices unknown. A loop on disk is kept, known by its source; labels writes it over.
func (s *Session) corpusFrom(dir, list string, a []string) error {
	split, hours, codec, labels := "", math.Inf(1), false, false
	for len(a) > 0 {
		if a[0] == "codec" || a[0] == "labels" {
			codec, labels, a = codec || a[0] == "codec", labels || a[0] == "labels", a[1:]
			continue
		}
		if len(a) < 2 || a[0] != "split" && a[0] != "hours" {
			return fmt.Errorf("%q: want split <name>, hours <h>, codec or labels", a[0])
		}
		if a[0] == "split" {
			split = a[1]
		} else if h, err := strconv.ParseFloat(a[1], 64); err != nil || h <= 0 {
			return fmt.Errorf("hours %q: want a positive number", a[1])
		} else {
			hours = h
		}
		a = a[2:]
	}
	f, err := os.Open(list)
	if err != nil {
		return err
	}
	rows, err := csv.NewReader(f).ReadAll()
	f.Close()
	if err != nil {
		return err
	}
	if len(rows) == 0 {
		return fmt.Errorf("%s: no header", list)
	}
	col := map[string]int{}
	for i, name := range rows[0] {
		col[name] = i
	}
	wav, ok1 := col["audio_filename"]
	mid, ok2 := col["midi_filename"]
	if !ok1 || !ok2 {
		return fmt.Errorf("%s: want audio_filename and midi_filename columns", list)
	}
	rows, at := rows[1:], func(name string) string { return filepath.Join(filepath.Dir(list), name) }
	if c, ok := col["split"]; ok && split != "" {
		rows = slices.DeleteFunc(rows, func(r []string) bool { return r[c] != split })
	}
	family := func([]string) int { return -1 }
	if c, ok := col["family"]; ok {
		for _, r := range rows {
			if r[c] != "" && familyIndex(r[c]) < 0 {
				return fmt.Errorf("%s: family %q: want one of %s, or none", list, r[c], strings.Join(audio.RipFamilies[:], ", "))
			}
		}
		family = func(r []string) int { return familyIndex(r[c]) }
	}
	vel := func([]string) func(float64) float64 { return velocityWords[""] }
	if c, ok := col["velocity"]; ok {
		for _, r := range rows {
			if velocityWords[r[c]] == nil {
				words := slices.Sorted(maps.Keys(velocityWords))
				return fmt.Errorf("%s: velocity %q: want %s, or an empty cell for the MIDI's own", list, r[c], strings.Join(words[1:], ", "))
			}
		}
		vel = func(r []string) func(float64) float64 { return velocityWords[r[c]] }
	}
	// a share of a large set may be all that is on disk, and a recording heard may be gone
	listed := len(rows)
	rows = slices.DeleteFunc(rows, func(r []string) bool {
		_, err1 := os.Stat(at(r[wav]))
		_, err2 := os.Stat(at(r[mid]))
		return err1 != nil && !labels || err2 != nil
	})
	missing := listed - len(rows)
	// a column of files a recording may name, an empty cell none
	named := func(name string) (func([]string) string, error) {
		c, ok := col[name]
		if !ok {
			return func([]string) string { return "" }, nil
		}
		for _, r := range rows {
			if _, err := os.Stat(at(r[c])); r[c] != "" && err != nil {
				return nil, fmt.Errorf("%s: %s: %w", list, name, err)
			}
		}
		return func(r []string) string { return r[c] }, nil
	}
	pitch, err := named("pitch")
	if err != nil {
		return err
	}
	beats, err := named("beats")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	loopFile := func(i int) string { return filepath.Join(dir, fmt.Sprintf("loop_%03d", i)) }
	taken, held, frames := map[int]bool{}, map[string]int{}, 0
	there, _ := filepath.Glob(filepath.Join(dir, "loop_*.json"))
	for _, name := range there {
		i, err := strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(filepath.Base(name), "loop_"), ".json"))
		data, rerr := os.ReadFile(name)
		var l corpusLoop
		if err != nil || rerr != nil || json.Unmarshal(data, &l) != nil {
			if labels {
				return fmt.Errorf("%s does not read, so its recording is not known: remove it and its frames, and run the line without labels", name)
			}
			continue // a label cut short is written again
		}
		taken[i], held[l.Source], frames = true, i, frames+l.Frames
	}
	// a run hears the recordings no loop holds, and labels writes those some loop does
	rows = slices.DeleteFunc(rows, func(r []string) bool { _, ok := held[r[wav]]; return ok != labels })
	// the same draw every run, so a share of the list spans its composers and years
	rng := rand.New(rand.NewPCG(1, 2))
	rng.Shuffle(len(rows), func(i, j int) { rows[i], rows[j] = rows[j], rows[i] })
	type job struct {
		at    int
		row   []string
		score *audio.MIDIScore
	}
	var (
		jobs    []job
		skipped []string
		next    int
		sum     = float64(frames) * audio.RipFrameSec
	)
	for _, r := range rows {
		if !labels && sum >= hours*3600 {
			break
		}
		g, err := os.Open(at(r[mid]))
		if err != nil {
			return err
		}
		score, err := audio.ReadMIDI(g)
		g.Close()
		if errors.Is(err, audio.ErrMIDIUnlined) {
			skipped = append(skipped, r[mid])
			continue
		} else if err != nil {
			return fmt.Errorf("%s: %w", at(r[mid]), err)
		}
		if i, ok := held[r[wav]]; ok {
			jobs = append(jobs, job{i, r, score})
			continue
		}
		for taken[next] {
			next++
		}
		jobs = append(jobs, job{next, r, score})
		taken[next] = true
		if c, ok := col["duration"]; ok {
			d, _ := strconv.ParseFloat(r[c], 64)
			sum += d
		}
	}
	loops := len(taken)
	for i := range taken {
		if i >= loops {
			return fmt.Errorf("%s: %s.json past %d loops: a loop below it was removed", dir, loopFile(i), loops)
		}
	}
	if loops == 0 {
		return fmt.Errorf("%s: no recordings listed for split %q, %d skipped", list, split, len(skipped))
	}
	var (
		mu            sync.Mutex
		wg            sync.WaitGroup
		first         error
		work          = make(chan job)
		placed, notes int // of the recordings a pitch track is named for
	)
	for range min(4, runtime.NumCPU()) { // a recording's samples are hundreds of megabytes
		wg.Go(func() {
			for j := range work {
				var coder *rand.Rand
				if codec {
					coder = rand.New(rand.NewPCG(0xc0dec, uint64(j.at)+1))
				}
				var tracks []pitchTrack
				err := error(nil)
				if p := pitch(j.row); p != "" {
					tracks, err = readPitch(at(p))
				}
				var grid [2][]float64
				if b := beats(j.row); b != "" && err == nil {
					grid[0], grid[1], err = readBeats(at(b))
				}
				n, sat := 0, 0
				if err == nil {
					n, sat, err = corpusRecording(loopFile(j.at), at(j.row[wav]), j.row[wav], family(j.row), vel(j.row), j.score, tracks, grid, coder, labels)
				}
				mu.Lock()
				if !labels {
					frames += n
				}
				if tracks != nil {
					placed, notes = placed+sat, notes+len(j.score.Notes)
				}
				first = cmp.Or(first, err)
				mu.Unlock()
			}
		})
	}
	for _, j := range jobs {
		work <- j
	}
	close(work)
	wg.Wait()
	if first != nil {
		return first
	}
	if err := writeJSON(filepath.Join(dir, "corpus.json"), newCorpusManifest(loops)); err != nil {
		return err
	}
	what := fmt.Sprintf("wrote %d recording(s) to %s from %s", len(jobs), dir, list)
	if labels {
		what = fmt.Sprintf("wrote the labels of %d recording(s) in %s again from %s and heard none", len(jobs), dir, list)
	} else if codec {
		what += ", a third through a lossy codec"
	}
	if notes > 0 {
		what += fmt.Sprintf(", %d of %d note(s) placed by a pitch track", placed, notes)
	}
	fmt.Fprintf(s.out, "%s, %d loop(s) and %.1f hours in all; %d listed were not on disk\n", what, loops, float64(frames)*audio.RipFrameSec/3600, missing)
	if len(skipped) > 0 {
		fmt.Fprintf(s.out, "skipped %d whose MIDI has no notes and no kick, snare or hihat, as %s\n", len(skipped), skipped[0])
	}
	return nil
}

// corpusRecording writes a recording's band frames, through a codec when coder is
// set, or under labels reads how many are on disk, and its score as one loop, its
// frames and the notes its pitch tracks place returned: a note by family, or its
// program's (programFamily) when family is -1, a hit and a strike of no line (a tom, a
// crash: drum -1, midi 0) by the drums, each at vel of its velocity, with grid's beats.
func corpusRecording(base, wav, source string, family int, vel func(float64) float64, score *audio.MIDIScore, tracks []pitchTrack, grid [2][]float64, coder *rand.Rand, labels bool) (int, int, error) {
	frames, whole, placed := 0, false, 0
	if labels {
		if frames, whole = bandsLength(base + ".bands"); !whole {
			return 0, 0, fmt.Errorf("%s.bands: no band frames of layout %d whole on disk to label", base, audio.RipLayout)
		}
	} else {
		f, err := os.Open(wav)
		if err != nil {
			return 0, 0, err
		}
		x, rate, err := audio.ReadWAV(f)
		f.Close()
		if err != nil {
			return 0, 0, fmt.Errorf("%s: %w", wav, err)
		}
		if coder != nil {
			if x, err = codecRoundTrip(x, rate, coder); err != nil {
				return 0, 0, err
			}
		}
		bands := audio.BandFrames(x, rate)
		if err := writeRows(base+".bands", bands); err != nil {
			return 0, 0, err
		}
		frames = len(bands)
	}
	frame := func(sec float64) int { return int(sec/audio.RipFrameSec + 0.5) }
	lab := corpusLoop{Frames: frames, Source: source}
	for _, n := range score.Notes {
		f := family
		if f < 0 {
			f = programFamily(n.Program)
		}
		ev := corpusEvent{Frame: frame(n.Start), End: frame(n.End), Drum: -1, MIDI: n.MIDI, Vel: vel(n.Vel), Wave: -1, Family: f}
		if tr := trackOf(tracks, n); tr != nil {
			ev.Cents = tr.cents(n.MIDI, ev.Frame, ev.End)
			placed += min(len(ev.Cents), 1)
		}
		lab.Events = append(lab.Events, ev)
	}
	for _, h := range append(slices.Clip(score.Hits), score.Struck...) {
		lab.Events = append(lab.Events, corpusEvent{Frame: frame(h.Start), End: frame(h.Start), Drum: int(h.Drum), Vel: vel(h.Vel), Wave: -1, Family: familyIndex("drums")})
	}
	lab.Events = slices.DeleteFunc(lab.Events, func(e corpusEvent) bool { return e.Frame >= frames })
	for i, to := range []*[]int{&lab.Beats, &lab.Downbeats} {
		for _, sec := range grid[i] {
			if f := frame(sec); sec >= 0 && f < frames {
				*to = append(*to, f)
			}
		}
	}
	return frames, placed, writeJSON(base+".json", lab)
}

// readBeats reads a recording's beats and bar lines, in its own seconds, by what
// their file is: a MIDI file's under its time signatures and tempo map, or a JAMS
// file's beat_position annotation
func readBeats(file string) (beats, downbeats []float64, err error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, nil, err
	}
	if isMIDI(file) {
		score, err := audio.ReadMIDI(bytes.NewReader(data))
		if err != nil {
			return nil, nil, fmt.Errorf("%s: %w", file, err)
		}
		beats, downbeats = score.Beats(func(sec float64) float64 { return sec })
		return beats, downbeats, nil
	}
	if beats, downbeats, err = audio.ReadJAMSBeats(bytes.NewReader(data)); err != nil {
		return nil, nil, fmt.Errorf("%s: %w", file, err)
	}
	return beats, downbeats, nil
}

// pitchTrack is a recording's pitch as its collection annotates it, one voice's
// or one string's: the seconds of its samples, ascending, the Hz there, 0 where
// nothing is voiced, and the usual time between two, which its reach is told in
type pitchTrack struct {
	sec, hz []float64
	step    float64
	notes   [][3]float64 // a string's notes: start, end, pitch; nil for a voice, whose track is every note's
}

// readPitch reads a pitch track by what its file is: a JAMS file's strings, each
// pitch_contour annotation beside the note_midi annotation of its data_source or
// its place, or a CSV's one voice, seconds and Hz a row under any header
func readPitch(file string) ([]pitchTrack, error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}
	var tracks []pitchTrack
	if !bytes.HasPrefix(bytes.TrimSpace(data), []byte("{")) {
		rows, err := csv.NewReader(bytes.NewReader(data)).ReadAll()
		if err != nil {
			return nil, fmt.Errorf("%s: %w", file, err)
		}
		tracks = []pitchTrack{{}}
		for i, r := range rows {
			sec, err1 := strconv.ParseFloat(strings.TrimSpace(r[0]), 64)
			hz, err2 := strconv.ParseFloat(strings.TrimSpace(r[len(r)-1]), 64)
			if len(r) != 2 || (err1 != nil || err2 != nil) && i > 0 {
				return nil, fmt.Errorf("%s: row %d: want seconds and Hz", file, i+1)
			} else if err1 == nil && err2 == nil {
				tracks[0].sec, tracks[0].hz = append(tracks[0].sec, sec), append(tracks[0].hz, hz)
			}
		}
	} else {
		var jam struct {
			Annotations []struct {
				Namespace string
				Data      json.RawMessage
				Meta      struct {
					Source string `json:"data_source"`
				} `json:"annotation_metadata"`
			}
		}
		if err := json.Unmarshal(data, &jam); err != nil {
			return nil, fmt.Errorf("%s: %w", file, err)
		}
		of, seen := map[string]int{}, map[string]int{} // a string's track, and the annotations seen of a namespace
		for _, a := range jam.Annotations {
			if a.Namespace != "pitch_contour" && a.Namespace != "note_midi" {
				continue
			}
			key := cmp.Or(a.Meta.Source, strconv.Itoa(seen[a.Namespace]))
			seen[a.Namespace]++
			k, ok := of[key]
			if !ok {
				k, of[key] = len(tracks), len(tracks)
				tracks = append(tracks, pitchTrack{notes: [][3]float64{}})
			}
			obs, err := jamsData(a.Data)
			if err != nil {
				return nil, fmt.Errorf("%s: %s: %w", file, a.Namespace, err)
			}
			for _, o := range obs {
				var note float64
				sample := struct {
					Frequency float64
					Voiced    *bool
				}{}
				if a.Namespace == "note_midi" && json.Unmarshal(o.Value, &note) == nil {
					tracks[k].notes = append(tracks[k].notes, [3]float64{o.Time, o.Time + o.Duration, note})
				} else if a.Namespace == "pitch_contour" && json.Unmarshal(o.Value, &sample) == nil {
					if sample.Voiced != nil && !*sample.Voiced {
						sample.Frequency = 0
					}
					tracks[k].sec, tracks[k].hz = append(tracks[k].sec, o.Time), append(tracks[k].hz, sample.Frequency)
				}
			}
		}
	}
	for k := range tracks {
		tr := &tracks[k]
		if !slices.IsSorted(tr.sec) {
			return nil, fmt.Errorf("%s: a pitch track's seconds do not ascend", file)
		}
		var gaps []float64
		for i := 1; i < len(tr.sec); i++ {
			gaps = append(gaps, tr.sec[i]-tr.sec[i-1])
		}
		if slices.Sort(gaps); len(gaps) > 0 {
			tr.step = gaps[len(gaps)/2]
		}
	}
	if !slices.ContainsFunc(tracks, func(t pitchTrack) bool { return len(t.sec) > 0 }) {
		return nil, fmt.Errorf("%s: no pitch track: want a CSV of seconds and Hz, or a JAMS file's pitch_contour annotations", file)
	}
	return tracks, nil
}

// jamsObservation is one of a JAMS annotation's observations
type jamsObservation struct {
	Time, Duration float64
	Value          json.RawMessage
}

// jamsData reads an annotation's data, written as a list of observations or as a
// list a field
func jamsData(raw json.RawMessage) ([]jamsObservation, error) {
	var obs []jamsObservation
	if json.Unmarshal(raw, &obs) == nil {
		return obs, nil
	}
	var cols struct {
		Time, Duration []float64
		Value          []json.RawMessage
	}
	if err := json.Unmarshal(raw, &cols); err != nil {
		return nil, err
	}
	if len(cols.Duration) != len(cols.Time) || len(cols.Value) != len(cols.Time) {
		return nil, errors.New("its time, duration and value lists differ in length")
	}
	for i, t := range cols.Time {
		obs = append(obs, jamsObservation{t, cols.Duration[i], cols.Value[i]})
	}
	return obs, nil
}

// trackOf is the track that places a note: a voice's own, or that of the string
// whose notes hold one of its pitch sounding at its onset, a frame's slack before
// it, the nearest onset first; nil where none does
func trackOf(tracks []pitchTrack, n audio.RipNote) *pitchTrack {
	var best *pitchTrack
	off := math.Inf(1)
	for k := range tracks {
		tr := &tracks[k]
		if tr.notes == nil {
			return tr
		}
		for _, a := range tr.notes {
			if d := math.Abs(a[0] - n.Start); math.Abs(a[2]-float64(n.MIDI)) <= 0.5 && n.Start >= a[0]-audio.RipFrameSec && n.Start <= a[1] && d < off {
				best, off = tr, d
			}
		}
	}
	return best
}

// pitchTrackReach is how far from a sample a pitch track still speaks, in its
// steps: past the half a whole track leaves, short of the step a missing sample does
const pitchTrackReach = 0.75

// at is the track's pitch at sec as a MIDI number, NaN where it voices nothing: its
// nearest sample's when that is voiced and in reach, and on the line to the
// sample across sec when that one is voiced and no sample is missing between
func (p *pitchTrack) at(sec float64) float64 {
	near := sort.SearchFloat64s(p.sec, sec)
	far := near - 1
	if near == len(p.sec) || near > 0 && sec-p.sec[far] < p.sec[near]-sec {
		near, far = far, near
	}
	voiced := func(k int, from, reach float64) bool {
		return k >= 0 && k < len(p.sec) && p.hz[k] > 0 && math.Abs(p.sec[k]-from) <= reach*p.step
	}
	midi := func(k int) float64 { return 69 + 12*math.Log2(p.hz[k]/440) }
	switch {
	case !voiced(near, sec, pitchTrackReach):
		return math.NaN()
	case !voiced(far, p.sec[near], 2*pitchTrackReach) || p.sec[far] == p.sec[near]:
		return midi(near)
	}
	return midi(near) + (sec-p.sec[near])/(p.sec[far]-p.sec[near])*(midi(far)-midi(near))
}

// cents is where the track places a note at each frame from from to to, as
// noteCents an engine's: NaN at a frame it does not voice, or voices a semitone
// or more from the note, and nil for a note it places at no frame
func (p *pitchTrack) cents(midi, from, to int) centsList {
	out, placed := make(centsList, max(to-from, 1)), false
	for i := range out {
		c := math.Round((p.at(float64(from+i)*audio.RipFrameSec)-float64(midi))*1000) / 10
		if out[i] = math.NaN(); math.Abs(c) < 100 {
			out[i], placed = c, true
		}
	}
	if !placed {
		return nil
	}
	return out
}

// parseTempos reads a range lo-hi, six tempos spread over it, or a list a,b,c,
// each within the sequencer's range: the tempos a corpus cycles through.
func parseTempos(arg string) ([]int, error) {
	var tempos []int
	if lo, hi, ok := strings.Cut(arg, "-"); ok {
		a, err1 := strconv.Atoi(lo)
		b, err2 := strconv.Atoi(hi)
		if err1 != nil || err2 != nil || a > b {
			return nil, fmt.Errorf("tempos %q: want lo-hi or a,b,c", arg)
		}
		for i := range 6 {
			tempos = append(tempos, a+(b-a)*i/5)
		}
	} else {
		for _, f := range strings.Split(arg, ",") {
			n, err := strconv.Atoi(f)
			if err != nil {
				return nil, fmt.Errorf("tempos %q: want lo-hi or a,b,c", arg)
			}
			tempos = append(tempos, n)
		}
	}
	for _, t := range tempos {
		if t < audio.MinBPM || t > audio.MaxBPM {
			return nil, fmt.Errorf("tempos %q: %d outside %d to %d", arg, t, audio.MinBPM, audio.MaxBPM)
		}
	}
	return tempos, nil
}

// exportCorpus writes loops into dir: loop_NNN.wav as rendered, loop_NNN.bands as
// the band frames a model hears (frames, values a frame, then float32 rows,
// little-endian), loop_NNN.json as its label, and corpus.json naming the layout. A
// loop whose label and frames are on disk is kept, so an export run again goes on;
// a part writes no layout. Under labels it writes every loop's label over and
// renders nothing, and writes the layout only where every loop's frames are on disk.
func (s *Session) exportCorpus(dir string, o tuneOptions) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	loopFile := func(i int) string { return filepath.Join(dir, fmt.Sprintf("loop_%03d", i)) }
	written, kept, unheard := 0, 0, 0
	o.skip = func(i int) bool {
		if o.parts > 0 && i%o.parts != o.part {
			return true
		}
		_, whole := bandsLength(loopFile(i) + ".bands")
		if o.labels {
			if !whole {
				unheard++
			}
			return false
		}
		_, err := os.Stat(loopFile(i) + ".json") // written last, so its loop is whole
		if err == nil && whole {
			kept++
		}
		return err == nil && whole
	}
	var sampled, tonal, metered, starts, overlaps, arranged int
	err := s.eachLoop(o, func(i int, l tuneLoop) error {
		written++
		sampled, tonal = sampled+l.sampled, tonal+l.tonal
		switch {
		case l.texture == "overlap":
			overlaps++
		case l.texture != "bank":
			arranged++
		}
		if l.label.Meter != "" {
			metered++
		}
		if len(l.label.Downbeats) > 0 && l.label.Downbeats[0] > 0 {
			starts++
		}
		base := loopFile(i)
		if !o.labels {
			if !o.nowav {
				f, err := os.Create(base + ".wav")
				if err != nil {
					return err
				}
				werr := audio.WriteWAV(f, l.audio)
				if cerr := f.Close(); werr == nil {
					werr = cerr
				}
				if werr != nil {
					return werr
				}
			}
			bands := audio.BandFrames(l.audio, audio.AudioSampleRate)
			if len(bands) != l.label.Frames {
				return fmt.Errorf("loop %d: %d frames heard, its label counted %d", i, len(bands), l.label.Frames)
			}
			if err := writeRows(base+".bands", bands); err != nil {
				return err
			}
		}
		return writeJSON(base+".json", l.label)
	})
	if err != nil {
		return err
	}
	if o.parts == 0 && unheard == 0 {
		if err := writeJSON(filepath.Join(dir, "corpus.json"), newCorpusManifest(o.loops)); err != nil {
			return err
		}
	}
	how := "as rendered"
	if o.aug {
		how = "through a room, a mix, voices of random genes and a tuning off A440"
	}
	if arranged > 0 {
		how += fmt.Sprintf(", %d arranged over the keyboard", arranged)
	}
	if o.glide {
		how += ", a share of the arranged notes bending into the next or scooping and of the voices swaying"
	}
	if o.meters {
		how += fmt.Sprintf(", %d in other meters than 4/4 and %d starting part-way into a bar", metered, starts)
	}
	if overlaps > 0 {
		how += fmt.Sprintf(", %d of single notes overlapping and no beat", overlaps)
	}
	if o.codec {
		how += ", a third through a lossy codec"
	}
	if o.samples != nil {
		how += fmt.Sprintf(", %d of %d tonal part(s) on the recorded notes in %s", sampled, tonal, o.samples.dir)
	}
	if kept > 0 {
		how += fmt.Sprintf("; %d were there", kept)
	}
	if o.labels {
		how += "; their labels alone, over those there, and nothing rendered"
	}
	if unheard > 0 {
		how += fmt.Sprintf("; %d with no frames whole on disk, so no corpus.json", unheard)
	}
	if o.parts > 0 {
		how += fmt.Sprintf("; part %d of %d, so no corpus.json", o.part, o.parts)
	}
	fmt.Fprintf(s.out, "wrote %d loop(s) of %d bar(s) to %s, %s\n", written, o.bars, dir, how)
	return nil
}

// noteCents is where a note the voice played sits at each of its frames, in cents
// from its pitch's cell centre, the loop's tuning in it, to a tenth: nil for a
// note that holds its pitch, which sits at the tuning throughout
func noteCents(v *audio.VoiceDef, e audio.RipEvent, start, dur, tuning float64, from, to int) []float64 {
	out, moves := make([]float64, max(to-from, 1)), false
	for i := range out {
		c := v.NoteCents(e.Bend, e.BendAt, dur, min(max(float64(from+i)*audio.RipFrameSec-start, 0), dur))
		out[i], moves = math.Round((tuning+c)*10)/10, moves || c != 0
	}
	if !moves {
		return nil
	}
	return out
}

// corpusCodecs are what recordings are stored in. A model that has only heard
// lossless renders flips notes on the noise and smeared attacks they leave.
var corpusCodecs = []struct {
	name, ext string
	kbps      []int
}{{"libopus", "opus", []int{64, 96, 128, 160}}, {"libmp3lame", "mp3", []int{96, 128, 192}}, {"aac", "m4a", []int{96, 128, 192}}}

// codecRoundTrip sends a third of the loops through a codec at a drawn bit rate and back
func codecRoundTrip(x []float64, rate int, rng *rand.Rand) ([]float64, error) {
	if rng.IntN(3) != 0 {
		return x, nil
	}
	c := corpusCodecs[rng.IntN(len(corpusCodecs))]
	return codecPass(x, rate, c.name, c.ext, c.kbps[rng.IntN(len(c.kbps))])
}

// codecPass codes x and decodes it, the same length and in time, through a file
// since a container read from a pipe loses the codec's delay. Only a corpus needs ffmpeg.
func codecPass(x []float64, sampleRate int, name, ext string, kbps int) ([]float64, error) {
	tmp, err := os.CreateTemp("", "corpus-*."+ext)
	if err != nil {
		return nil, err
	}
	tmp.Close()
	defer os.Remove(tmp.Name())
	rate := strconv.Itoa(sampleRate)
	raw := make([]byte, 8*len(x))
	for i, v := range x {
		binary.LittleEndian.PutUint64(raw[8*i:], math.Float64bits(v))
	}
	enc := exec.Command("ffmpeg", "-loglevel", "error", "-y", "-f", "f64le", "-ar", rate, "-ac", "1", "-i", "pipe:0", "-c:a", name, "-b:a", fmt.Sprintf("%dk", kbps), tmp.Name())
	enc.Stdin = bytes.NewReader(raw)
	if out, err := enc.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("codec %s needs ffmpeg: %w: %s", name, err, out)
	}
	out, err := exec.Command("ffmpeg", "-loglevel", "error", "-i", tmp.Name(), "-f", "f64le", "-ar", rate, "-ac", "1", "pipe:1").Output()
	if err != nil {
		return nil, fmt.Errorf("codec %s: %w", name, err)
	}
	y := make([]float64, len(x))
	for i := range min(len(y), len(out)/8) {
		y[i] = math.Float64frombits(binary.LittleEndian.Uint64(out[8*i:]))
	}
	return y, nil
}

// bandsLength is the frames a loop's .bands holds by its header, false for a file
// that is not there, not whole or not of RipLayout's frames, its fine plane's
// bands then its sharp plane's: what a labels pass reads in place of a render
func bandsLength(file string) (int, bool) {
	f, err := os.Open(file)
	if err != nil {
		return 0, false
	}
	defer f.Close()
	var head [2]uint32
	st, err := f.Stat()
	if err != nil || binary.Read(f, binary.LittleEndian, &head) != nil || head[1] != 2*audio.RipBands || st.Size() != 8+4*int64(head[0])*int64(head[1]) {
		return 0, false
	}
	return int(head[0]), true
}

// writeRows writes rows of float32s as a corpus's frames and a judge's rows are:
// the rows and the values a row as two uint32s, then the rows, little-endian
func writeRows(file string, rows [][]float32) error {
	f, err := os.Create(file)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	le := binary.LittleEndian
	values := 0
	if len(rows) > 0 {
		values = len(rows[0])
	}
	binary.Write(w, le, uint32(len(rows)))
	binary.Write(w, le, uint32(values))
	for _, row := range rows {
		binary.Write(w, le, row)
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func writeJSON(file string, v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return os.WriteFile(file, data, 0o644)
}

func cmdBench(s *Session, a []string) error {
	o := tuneOptions{loops: 8, bars: 4, seed: 7, gen: true, steady: true}
	verbose := false
words:
	for n := len(a); n > 0; n = len(a) {
		switch {
		case n > 1 && a[n-2] == "sampled":
			set, err := loadSampleSet(s.path(a[n-1]))
			if err != nil {
				return err
			}
			o.samples, a = set, a[:n-2]
		case a[n-1] == "verbose":
			verbose, a = true, a[:n-1]
		case a[n-1] == "meters":
			o.meters, a = true, a[:n-1]
		default:
			break words
		}
	}
	if len(a) > 2 {
		return fmt.Errorf("%q: want verbose, meters or sampled <dir>", a[2])
	}
	for i, p := range []*int{&o.loops, &o.bars} {
		if len(a) > i {
			n, err := strconv.Atoi(a[i])
			if err != nil || n < 1 || n > 256 {
				return fmt.Errorf("%q: want 1 to 256", a[i])
			}
			*p = n
		}
	}
	return s.bench(o, verbose)
}

// benchRow is one loop as one front end ripped it, for bench's verbose listing
type benchRow struct {
	what  string // index, tempo, meter, key, kind, parts and front end, in columns
	sc    audio.RipScore
	err   error
	heard string // the meter the rip read, when not the loop's
}

// benchTally sums the scores of the loops one front end ripped, and counts the
// rips that read the loop's meter, and those that read a bar of its steps (3/4
// for 6/8), the grid right under another label
type benchTally struct {
	sum           audio.RipScore
	n, heard, bar int
}

func (t *benchTally) add(sc audio.RipScore, want, read audio.Meter) {
	t.sum.Tempo, t.sum.Bars, t.sum.Drums = t.sum.Tempo+sc.Tempo, t.sum.Bars+sc.Bars, t.sum.Drums+sc.Drums
	t.sum.Melody, t.sum.Bass, t.sum.Total = t.sum.Melody+sc.Melody, t.sum.Bass+sc.Bass, t.sum.Total+sc.Total
	t.n++
	if read.String() == want.String() {
		t.heard++
	}
	if read.Steps() == want.Steps() {
		t.bar++
	}
}

func (t *benchTally) mean() audio.RipScore {
	d := float64(t.n)
	return audio.RipScore{Tempo: t.sum.Tempo / d, Bars: t.sum.Bars / d, Drums: t.sum.Drums / d, Melody: t.sum.Melody / d, Bass: t.sum.Bass / d, Total: t.sum.Total / d}
}

// bench grades the session's front ends, the analysis and the model when one
// loaded, on fresh arranged loops at a steady tempo from a bar line, clean and
// augmented, by the mean of each rip's score against the loop's expected rip,
// so a trained model is measured before it is trusted with a recording; under
// meters each meter's mean with the rips that read it, verbose every loop
func (s *Session) bench(o tuneOptions, verbose bool) error {
	type front struct {
		name string
		fe   audio.RipFrontEnd
	}
	fronts := []front{{"analysis", nil}}
	if s.frontEnd != nil {
		fronts = append(fronts, front{"model", s.frontEnd})
	}
	var rows []benchRow
	for _, aug := range []bool{false, true} {
		o.aug = aug
		loops, err := s.corpus(o)
		if err != nil {
			return err
		}
		kind := "clean"
		if aug {
			kind = "augmented"
		}
		for _, f := range fronts {
			var all benchTally
			byMeter := map[audio.Meter]*benchTally{}
			for i, l := range loops {
				r, err := audio.Ripper{Params: s.rip, FrontEnd: f.fe}.Samples(l.audio, audio.AudioSampleRate)
				var sc audio.RipScore
				heard := ""
				if err == nil {
					sc = audio.CompareRips(r, l.want)
					all.add(sc, l.meter, r.Meter)
					if byMeter[l.meter] == nil {
						byMeter[l.meter] = &benchTally{}
					}
					byMeter[l.meter].add(sc, l.meter, r.Meter)
					if r.Meter.String() != l.meter.String() {
						heard = r.Meter.String()
					}
				}
				if verbose {
					what := fmt.Sprintf("%3d %3d bpm %-4s %-3s %-14s %-9s %-34s %-8s", i, l.bpm, l.meter, noteName(l.root), scaleName(l.scale), kind, l.texture, f.name)
					rows = append(rows, benchRow{what: what, sc: sc, err: err, heard: heard})
				}
			}
			if all.n == 0 {
				fmt.Fprintf(s.out, "%s on %s loops: no rip\n", f.name, kind)
				continue
			}
			fmt.Fprintf(s.out, "%s on %d %s loop(s): %s\n", f.name, all.n, kind, benchScore(all.mean()))
			if !o.meters {
				continue
			}
			for _, row := range corpusMeters {
				if m := row.m; byMeter[m] != nil {
					t := byMeter[m]
					fmt.Fprintf(s.out, "  in %s, %d loop(s): %s, the meter read in %d, its bar in %d\n", m, t.n, benchScore(t.mean()), t.heard, t.bar)
				}
			}
		}
	}
	if !verbose {
		return nil
	}
	var ripped []benchRow
	for _, r := range rows {
		if r.err != nil {
			fmt.Fprintf(s.out, "loop  %s no rip (%v)\n", r.what, r.err)
			continue
		}
		fmt.Fprintf(s.out, "loop  %s %s%s\n", r.what, benchScore(r.sc), r.read())
		ripped = append(ripped, r)
	}
	slices.SortStableFunc(ripped, func(a, b benchRow) int { return cmp.Compare(a.sc.Total, b.sc.Total) })
	for _, r := range ripped[:min(3, len(ripped))] {
		fmt.Fprintf(s.out, "worst %s %s%s\n", r.what, benchScore(r.sc), r.read())
	}
	return nil
}

// read is the meter a row's rip read when not the loop's, as a listing ends it
func (r benchRow) read() string {
	if r.heard == "" {
		return ""
	}
	return ", read in " + r.heard
}

func benchScore(sc audio.RipScore) string {
	return fmt.Sprintf("%.3f (tempo %.2f bars %.2f drums %.2f melody %.2f bass %.2f)", sc.Total, sc.Tempo, sc.Bars, sc.Drums, sc.Melody, sc.Bass)
}

// tune searches the rip's parameters over the loops the corpus renders, each
// candidate scored by how its rip of every loop, through the package's own
// analysis, compares with the loop's expected rip
func (s *Session) tune(o tuneOptions, file string) error {
	loops, err := s.corpus(o)
	if err != nil {
		return err
	}
	ranges := audio.RipGeneRanges()
	bounds := make([]genetic.ParameterBounds, len(ranges))
	for i, r := range ranges {
		bounds[i] = genetic.ParameterBounds{Min: r[0], Max: r[1]}
	}
	over := fmt.Sprintf("%d loop(s) of %d bar(s)", len(loops), o.bars)
	best, base, err := s.search(o, over, bounds, s.rip.Genes(), false, func(genes []float64) float64 {
		p := audio.RipParamsFromGenes(genes)
		var sum float64
		for _, l := range loops {
			if r, err := audio.RipSamples(l.audio, audio.AudioSampleRate, p); err == nil {
				sum += audio.CompareRips(r, l.want).Total
			}
		}
		return sum / float64(len(loops))
	})
	if err != nil {
		return err
	}
	return s.keepRip(file, audio.RipParamsFromGenes(best.Data), best.Score, base)
}

// search runs the generational engine over genes within bounds from start, a
// genome scored once and candidates in parallel: near, its first pool is start
// and sets a step from it, the step widening while the best betters and
// narrowing while it stalls, as the fit's; else draws across the ranges. A
// score under zero is a set refused. It returns the best, settled, and the start's score.
func (s *Session) search(o tuneOptions, over string, bounds []genetic.ParameterBounds, start []float64, near bool, eval func([]float64) float64) (genetic.Candidate[[]float64, float64], float64, error) {
	eval = genetic.Memoize(func(g []float64) string { return fmt.Sprint(g) }, eval)
	base := eval(start)
	fmt.Fprintf(s.out, "tuning over %s: the current parameters score %.3f\n", over, base)
	cfg := genetic.DefaultConfig()
	cfg.PoolSize, cfg.EliteCount, cfg.MaxIterations = o.pool, max(o.pool/8, 1), o.generations
	cfg.Parallelism, cfg.Seed = runtime.NumCPU(), o.seed
	cfg.PerturbationRate, cfg.PerturbationStrength = 0.25, 0.15
	step := &genetic.BoundedPerturbator{Bounds: bounds, Boundary: genetic.BoundaryReflect}
	draw := func(rng *rand.Rand) []float64 {
		g := make([]float64, len(bounds))
		for i, b := range bounds {
			g[i] = b.Min + rng.Float64()*(b.Max-b.Min)
		}
		return g
	}
	if near {
		cfg.PerturbationStrength = nearStep
		draw = func(rng *rand.Rand) []float64 {
			g := slices.Clone(start)
			for try := 0; try < 8 && slices.Equal(g, start); try++ { // a draw that moved no gene is the start again
				step.Perturb(&g, cfg.PerturbationRate, nearStep, rng)
			}
			return g
		}
	}
	e := genetic.NewEngine[[]float64, float64](eval, draw, &genetic.TournamentSelector[[]float64, float64]{TournamentSize: 3},
		&genetic.UniformCombiner[[]float64, float64, float64]{MixProbability: 0.5}, step, cfg)
	e.SetSeeds([][]float64{slices.Clone(start)})
	if near {
		e.SetSchedule(genetic.AdaptiveStrength[float64](cfg.PerturbationRate, nearStep, 0.01, 0.4, 1.5))
	}
	say := func(p *genetic.Pool[[]float64, float64], gen int) bool {
		var sum float64
		passed := 0
		for _, m := range p.Members {
			if m.Score >= 0 {
				sum, passed = sum+m.Score, passed+1
			}
		}
		fmt.Fprintf(s.out, "generation %d: best %.3f mean %.3f", gen, p.Stats.BestScore, sum/float64(max(passed, 1)))
		if refused := len(p.Members) - passed; refused > 0 {
			fmt.Fprintf(s.out, " of %d, %d refused", passed, refused)
		}
		fmt.Fprintln(s.out)
		return false
	}
	e.SetTerminator(say)
	pool, err := e.Run(context.Background())
	if err != nil {
		return genetic.Candidate[[]float64, float64]{}, 0, err
	}
	say(pool, pool.Generation) // the engine's loop ends before the last one's turn
	return settled(pool.Members[0], start, base, eval), base, nil
}

// settled is a search's best with each gene whose move the score does not see
// back at the start's: a pool moves genes the truths cannot read, the heads'
// weights under a model without them, and they are not to be written by chance.
// Each is put back alone, side by side; those that cost nothing go back
// together, or one at a time where together they cost.
func settled(best genetic.Candidate[[]float64, float64], start []float64, base float64, eval func([]float64) float64) genetic.Candidate[[]float64, float64] {
	back := func(data []float64, genes ...int) []float64 {
		out := slices.Clone(data)
		for _, k := range genes {
			out[k] = start[k]
		}
		return out
	}
	var moved []int
	for k := range best.Data {
		if best.Score > base && best.Data[k] != start[k] {
			moved = append(moved, k)
		}
	}
	alone := make([]float64, len(moved))
	var wg sync.WaitGroup
	free := make(chan struct{}, runtime.NumCPU())
	for i, k := range moved {
		wg.Go(func() {
			free <- struct{}{}
			alone[i] = eval(back(best.Data, k))
			<-free
		})
	}
	wg.Wait()
	var unseen []int
	for i, k := range moved {
		if alone[i] >= best.Score {
			unseen = append(unseen, k)
		}
	}
	if all := back(best.Data, unseen...); len(unseen) > 0 {
		if score := eval(all); score >= best.Score {
			best.Data, best.Score = all, score
			return best
		}
	}
	for _, k := range unseen {
		one := back(best.Data, k)
		if score := eval(one); score >= best.Score {
			best.Data, best.Score = one, score
		}
	}
	return best
}

// nearStep is the step a search near its start opens with, a share of each
// gene's range, as the voice fit's
const nearStep = 0.1

// keepRip writes p and listens with it from now on when its score beats the
// current parameters' base
func (s *Session) keepRip(file string, p audio.RipParams, score, base float64) error {
	if score <= base {
		fmt.Fprintf(s.out, "nothing beat the current parameters (%.3f); they stay\n", base)
		return nil
	}
	file, err := saveFile(file, s.ripSrc, s.files.ripTarget)
	if err != nil {
		return err
	}
	data, err := audio.MarshalRipParams(p)
	if err != nil {
		return err
	}
	if err := os.WriteFile(file, data, 0o644); err != nil {
		return err
	}
	s.rip, s.ripSrc = p, file
	fmt.Fprintf(s.out, "wrote %s: score %.3f, was %.3f; rip listens with it from now on\n", file, score, base)
	return nil
}

// modelGenes are the parameters a front end's strikes are decoded by, the bass
// line's own path among them, which tune truth searches; second_voice and cross
// part two lines of salience, which such a front end's register never reaches
var modelGenes = []func(*audio.RipParams) *float64{
	func(p *audio.RipParams) *float64 { return &p.Strike },
	func(p *audio.RipParams) *float64 { return &p.StrikeLow },
	func(p *audio.RipParams) *float64 { return &p.StrikeHold },
	func(p *audio.RipParams) *float64 { return &p.StrikeGap },
	func(p *audio.RipParams) *float64 { return &p.StrikeAgain },
	func(p *audio.RipParams) *float64 { return &p.Release },
	func(p *audio.RipParams) *float64 { return &p.Voicing },
	func(p *audio.RipParams) *float64 { return &p.MinNote },
	func(p *audio.RipParams) *float64 { return &p.Jump },
	func(p *audio.RipParams) *float64 { return &p.Switch },
	func(p *audio.RipParams) *float64 { return &p.Retrigger },
}

// drumGenes is what tune truth searches beside them when a truth gives drum
// hits, the bar a hit rises to against its drum's loudest; barGenes when one
// gives bars: the tempo's prior and combs, the beats' tightness, the bar lines'
// weights and the meter's margins
var drumGenes = []func(*audio.RipParams) *float64{
	func(p *audio.RipParams) *float64 { return &p.OnsetMin },
}

var barGenes = []func(*audio.RipParams) *float64{
	func(p *audio.RipParams) *float64 { return &p.TempoPrior },
	func(p *audio.RipParams) *float64 { return &p.TempoWidth },
	func(p *audio.RipParams) *float64 { return &p.Tightness },
	func(p *audio.RipParams) *float64 { return &p.TempoComb },
	func(p *audio.RipParams) *float64 { return &p.BeatComb2 },
	func(p *audio.RipParams) *float64 { return &p.BeatComb3 },
	func(p *audio.RipParams) *float64 { return &p.BeatComb4 },
	func(p *audio.RipParams) *float64 { return &p.BarLine },
	func(p *audio.RipParams) *float64 { return &p.BarBeat },
	func(p *audio.RipParams) *float64 { return &p.BarAccent },
	func(p *audio.RipParams) *float64 { return &p.BarHead },
	func(p *audio.RipParams) *float64 { return &p.BarMove },
	func(p *audio.RipParams) *float64 { return &p.MeterMargin },
	func(p *audio.RipParams) *float64 { return &p.MeterHead },
	func(p *audio.RipParams) *float64 { return &p.RepeatMargin },
	func(p *audio.RipParams) *float64 { return &p.NearMargin },
	func(p *audio.RipParams) *float64 { return &p.NearRepeat },
}

// geneOf is where a parameter stands in Genes' order, which names none
func geneOf(at func(*audio.RipParams) *float64) int {
	var p audio.RipParams
	*at(&p) = 1
	return slices.Index(p.Genes(), 1)
}

// tuneTruth searches the decoders' parameters against the recordings a list
// names, each heard once through fe or read back from o.frames, the rest held at
// the session's: modelGenes, drumGenes where a truth gives hits and barGenes
// where one gives bars. A set scores the mean of every score its conditions
// carry (scores), refused past o.guard; it says each before and after.
func (s *Session) tuneTruth(o tuneOptions, list, file string, fe audio.RipFrontEnd) error {
	takes, conds, err := readTruthList(list, fe, o.frames, s.earSum)
	if err != nil {
		return err
	}
	// the current parameters' rips hear every recording and read every transcript
	cur, sums, read := s.rip, make([]truthCount, len(conds)), 0
	for _, t := range takes {
		t.judge = s.judge
		c, err := t.count(cur, s.lead)
		if err != nil {
			return err
		}
		if c.notes.given+c.drums.given+c.lines == 0 {
			return fmt.Errorf("%s: no notes, hits or bars given", t.truthFile)
		}
		t.given, sums[t.cond] = c, sums[t.cond].plus(c)
		if t.ear.read {
			read++
		}
	}
	var all truthCount
	for _, c := range sums {
		all = all.plus(c)
	}
	genes := slices.Clone(modelGenes)
	if all.drums.given > 0 {
		genes = append(genes, drumGenes...)
	}
	if all.lines > 0 {
		genes = append(genes, barGenes...)
	}
	ranges := audio.RipGeneRanges()
	bounds, start := make([]genetic.ParameterBounds, len(genes)), make([]float64, len(genes))
	for k, at := range genes {
		r := ranges[geneOf(at)]
		bounds[k], start[k] = genetic.ParameterBounds{Min: r[0], Max: r[1]}, *at(&cur)
	}
	params := func(data []float64) audio.RipParams {
		p := cur
		for k, at := range genes {
			*at(&p) = data[k]
		}
		return p
	}
	if o.frames != "" {
		fmt.Fprintf(s.out, "%d recording(s) heard and their frames kept in %s, %d read back from it\n", len(takes)-read, o.frames, read)
	}
	before := scores(sums, sums)
	score := func(data []float64) float64 {
		return guarded(scores(conditionCounts(takes, len(conds), params(data), s.lead), sums), before, o.guard)
	}
	over := fmt.Sprintf("%d recording(s) of %d condition(s) against %d note(s), %d hit(s) and %d bar line(s) given, a set refused that takes over %g from a score",
		len(takes), len(conds), all.notes.given, all.drums.given, all.lines, o.guard)
	best, base, err := s.search(o, over, bounds, start, true, score)
	if err != nil {
		return err
	}
	after := sums
	if best.Score > base {
		after = conditionCounts(takes, len(conds), params(best.Data), s.lead)
	}
	rate := all.hitRate()
	for c, name := range conds {
		if sums[c].notes.given > 0 {
			fmt.Fprintf(s.out, "%s: F1 %.3f before, %.3f after\n", name, sums[c].notes.f1(), after[c].notes.f1())
		}
		if was, ok := sums[c].drumScore(sums[c], rate); ok {
			now, _ := after[c].drumScore(sums[c], rate)
			if sums[c].drums.given > 0 {
				fmt.Fprintf(s.out, "%s: drums F1 %.3f before, %.3f after\n", name, was, now)
			} else {
				fmt.Fprintf(s.out, "%s: drums %.3f before, %.3f after: %.1f and %.1f hit(s) a minute heard where none plays, against %.0f given elsewhere\n",
					name, was, now, float64(sums[c].drums.heard)/sums[c].drumless, float64(after[c].drums.heard)/sums[c].drumless, rate)
			}
		}
		if sums[c].lines > 0 {
			fmt.Fprintf(s.out, "%s: bars %.3f before, %.3f after\n", name, sums[c].bars(), after[c].bars())
		}
	}
	return s.keepRip(file, params(best.Data), best.Score, base)
}

// guarded is a set's score: the mean of its scores, or for a set that takes
// more than guard from any of them at the start, the most it takes under zero,
// so every set that passes scores over every set refused
func guarded(got, start []float64, guard float64) float64 {
	var sum, cost float64
	for c := range got {
		sum, cost = sum+got[c], max(cost, start[c]-got[c])
	}
	if cost > guard {
		return -cost
	}
	return sum / float64(len(got))
}

// heardOnce hears one recording once, or reads the frames a run before kept in
// file: valid while the front end it wraps ignores the parameters, as the model
// does, so a search over the decoders pays them alone
type heardOnce struct {
	fe   audio.RipFrontEnd
	file string // where the frames are kept between runs, "" for nowhere
	once sync.Once
	f    *audio.RipFrames
	read bool // the frames came from file
	err  error
}

func (h *heardOnce) Hear(x []float64, rate int, p audio.RipParams) (*audio.RipFrames, error) {
	h.once.Do(func() {
		if h.f = readFrames(h.file); h.f != nil {
			h.read = true
		} else if h.f, h.err = h.fe.Hear(x, rate, p); h.err == nil && h.file != "" {
			h.err = writeFrames(h.file, h.f)
		}
	})
	return h.f, h.err
}

// readFrames is the frames a file keeps, nil when it holds none whole
func readFrames(file string) *audio.RipFrames {
	f, err := os.Open(file)
	if err != nil {
		return nil
	}
	defer f.Close()
	var fr audio.RipFrames
	if gob.NewDecoder(bufio.NewReader(f)).Decode(&fr) != nil {
		return nil
	}
	return &fr
}

// writeFrames keeps a model's frames in file, whole or not at all: every field
// a model fills is exported, which is all gob writes
func writeFrames(file string, fr *audio.RipFrames) error {
	tmp, err := os.CreateTemp(filepath.Dir(file), "*.part")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	w := bufio.NewWriter(tmp)
	err = gob.NewEncoder(w).Encode(fr)
	if err == nil {
		err = w.Flush()
	}
	if err == nil {
		err = tmp.Chmod(0o644)
	}
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return err
	}
	return os.Rename(tmp.Name(), file)
}

// truthTake is a recording a truth list names, kept at the rate the rip reads it
// at, so no candidate resamples it, its condition, and its truth: a MIDI file's
// notes, or a transcript read against each rip as truth reads it
type truthTake struct {
	wav, truthFile string
	x              []float64
	ear            *heardOnce
	truth          func(*audio.Rip) ([]audio.RipNote, error)
	hits           []audio.RipHit  // the drum hits its truth gives, nil for none
	drums          bool            // its truth knows the drums, so a hit heard that it does not give is false
	beats, lines   []float64       // the beats and bar lines a label gives, in the recording's seconds, nil for none
	given          truthCount      // what the truth gives, all missed by a rip that fails
	cond           int             // its condition, an index into the list's
	timed          bool            // the truth's seconds are the recording's
	judge          *audio.RipJudge // what rules its rip's strikes, nil for the installed scores
}

// truthLine is a truth list's line: a recording and its truth, relative paths
// read from dir, its condition, whether the truth's seconds are the recording's
// and whether no drum plays in it
type truthLine struct {
	dir, wav, truth, cond string
	timed, drumless       bool
}

func (l truthLine) at(p string) string {
	if filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(l.dir, p)
}

// readTruthLines reads a line "wav<TAB>truth[<TAB>condition[<TAB>[timed] [drumless]]]"
// a recording, a relative path from the list's own folder, a line that names no
// condition one of its own under its recording's name
func readTruthLines(list string) ([]truthLine, error) {
	data, err := os.ReadFile(list)
	if err != nil {
		return nil, err
	}
	var lines []truthLine
	for i, line := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		f := strings.Split(line, "\t")
		for k := range f {
			f[k] = strings.TrimSpace(f[k])
		}
		var words []string
		if len(f) == 4 {
			words = strings.Fields(f[3])
		}
		other := func(w string) bool { return w != "timed" && w != "drumless" }
		if len(f) < 2 || len(f) > 4 || f[0] == "" || f[1] == "" || slices.ContainsFunc(words, other) {
			return nil, fmt.Errorf("%s line %d: want wav<TAB>truth[<TAB>condition[<TAB>[timed] [drumless]]]", list, i+1)
		}
		l := truthLine{dir: filepath.Dir(list), wav: f[0], truth: f[1], cond: f[0], timed: slices.Contains(words, "timed"), drumless: slices.Contains(words, "drumless")}
		if len(f) > 2 && f[2] != "" {
			l.cond = f[2]
		}
		lines = append(lines, l)
	}
	if len(lines) == 0 {
		return nil, fmt.Errorf("%s: no recordings listed", list)
	}
	return lines, nil
}

// take is a line's recording, its bytes and their checksum given, at the rate
// the rip reads it at, and its truth; with frames, its frames are kept there
// under the checksums of its file and of the weight files
func (l truthLine) take(recording []byte, sum [sha256.Size]byte, fe audio.RipFrontEnd, frames, weights string) (*truthTake, error) {
	x, rate, err := audio.ReadWAV(bytes.NewReader(recording))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", l.wav, err)
	}
	given, err := os.ReadFile(l.at(l.truth))
	if err != nil {
		return nil, err
	}
	t := &truthTake{wav: l.wav, truthFile: l.truth, x: audio.Resample(x, rate, audio.RipRate), ear: &heardOnce{fe: fe}, timed: l.timed, drums: l.drumless}
	if frames != "" {
		t.ear.file = filepath.Join(frames, fmt.Sprintf("%x-%.16s.frames", sum[:8], weights))
	}
	switch {
	case strings.EqualFold(filepath.Ext(l.truth), ".json"): // a corpus loop's label
		var lab corpusLoop
		if err := json.Unmarshal(given, &lab); err != nil {
			return nil, fmt.Errorf("%s: %w", l.truth, err)
		}
		beats, downbeats := lab.grid()
		notes := lab.notes()
		t.truth, t.hits, t.beats, t.lines = func(*audio.Rip) ([]audio.RipNote, error) { return notes, nil }, lab.hits(), beats, downbeats
		t.drums = true // a label holds every event of its loop, so no hit there is none played
	case isMIDI(l.truth):
		score, err := audio.ReadMIDI(bytes.NewReader(given))
		if err != nil {
			return nil, fmt.Errorf("%s: %w", l.truth, err)
		}
		// its drum channel's hits, and no bars: a transcription's are its writer's default signature and tempo
		t.truth, t.hits = func(*audio.Rip) ([]audio.RipNote, error) { return score.Notes, nil }, score.Hits
	default:
		text := string(given)
		t.truth = func(r *audio.Rip) ([]audio.RipNote, error) { return transcript(text, r) }
	}
	if l.drumless && len(t.hits) > 0 {
		return nil, fmt.Errorf("%s: drumless, and its truth gives %d drum hit(s)", l.truth, len(t.hits))
	}
	t.drums = t.drums || len(t.hits) > 0
	return t, nil
}

// readTruthList reads a truth list's recordings and the conditions in the list's
// order; with frames, each recording's are kept there
func readTruthList(list string, fe audio.RipFrontEnd, frames, weights string) ([]*truthTake, []string, error) {
	lines, err := readTruthLines(list)
	if err != nil {
		return nil, nil, err
	}
	if frames != "" {
		if err := os.MkdirAll(frames, 0o755); err != nil {
			return nil, nil, err
		}
	}
	var takes []*truthTake
	var conds []string
	for _, l := range lines {
		recording, err := os.ReadFile(l.at(l.wav))
		if err != nil {
			return nil, nil, err
		}
		t, err := l.take(recording, sha256.Sum256(recording), fe, frames, weights)
		if err != nil {
			return nil, nil, err
		}
		if t.cond = slices.Index(conds, l.cond); t.cond < 0 {
			t.cond, conds = len(conds), append(conds, l.cond)
		}
		takes = append(takes, t)
	}
	return takes, conds, nil
}

// count rips the take under p and counts it against its truth: its notes by
// pitch and onset, its hits within half a step as truth counts them, every hit
// heard false where the truth knows the drums and gives none, and the downbeats
// given that fall on a bar line of a bar the given's length (MatchBars)
func (t *truthTake) count(p audio.RipParams, lead audio.LeadRule) (truthCount, error) {
	missed := truthCount{notes: noteCount{given: t.given.notes.given}, drums: noteCount{given: t.given.drums.given}, lines: t.given.lines}
	if minutes := float64(len(t.x)) / audio.RipRate / 60; len(t.hits) > 0 {
		missed.drummed = minutes
	} else if t.drums {
		missed.drumless = minutes
	}
	r, err := audio.Ripper{Params: p, FrontEnd: t.ear, Lead: lead, Judge: t.judge}.Samples(t.x, audio.RipRate)
	if err != nil {
		return missed, fmt.Errorf("%s: %w", t.wav, err)
	}
	given, err := t.truth(r)
	if err != nil {
		return missed, fmt.Errorf("%s: %w", t.truthFile, err)
	}
	ag, _ := matchHeard(r, given, t.timed)
	c := truthCount{notes: everyNote(ag), lines: len(t.lines), drummed: missed.drummed, drumless: missed.drumless}
	if t.drums {
		for _, l := range audio.MatchHits(r.Hits(), t.hits, 30/float64(r.BPM)/audio.StepsPerBeat).Drums {
			c.drums = c.drums.plus(noteCount{l.Matched, l.B, l.A})
		}
	}
	if c.lines > 0 {
		c.onLine = audio.MatchBars(r, t.beats, t.lines).Downbeats * float64(c.lines)
	}
	return c, nil
}

// truthCount is a rip against a truth: its notes, its drum hits, and of the
// downbeats the truth gives those on a bar line of the rip's; with the minutes
// of the recordings whose truth gives hits and of those it knows play no drum
type truthCount struct {
	notes, drums      noteCount
	lines             int
	onLine            float64
	drummed, drumless float64
}

func (c truthCount) plus(o truthCount) truthCount {
	return truthCount{c.notes.plus(o.notes), c.drums.plus(o.drums), c.lines + o.lines, c.onLine + o.onLine, c.drummed + o.drummed, c.drumless + o.drumless}
}

// bars is the share of the downbeats given that fall on a bar line
func (c truthCount) bars() float64 { return c.onLine / float64(max(c.lines, 1)) }

// hitRate is the hits a minute the truths give over the recordings with drums,
// what a hit heard where none plays is weighed against: 0 when none gives any
func (c truthCount) hitRate() float64 {
	if c.drummed == 0 {
		return 0
	}
	return float64(c.drums.given) / c.drummed
}

// drumScore is c's drums where given carries a score of them: the F1 where it
// gives hits; where it knows that none plays, the F1 of a rip that heard c's
// hits beside every hit such minutes hold at rate a minute, so one by none
// heard and less by degrees. With no rate nothing weighs a hit: no score.
func (c truthCount) drumScore(given truthCount, rate float64) (float64, bool) {
	if given.drums.given > 0 {
		return c.drums.f1(), true
	}
	even := rate * given.drumless
	if even == 0 {
		return 0, false
	}
	return 2 * even / (2*even + float64(c.drums.heard)), true
}

// scores is every score the conditions carry, in their order: each one's notes'
// F1, its drums' (drumScore) and its bars where given gives it any to count,
// each a share of one capability, so a set is weighed by their mean and held by each
func scores(counts, given []truthCount) []float64 {
	var all truthCount
	for _, g := range given {
		all = all.plus(g)
	}
	var out []float64
	for i, c := range counts {
		if given[i].notes.given > 0 {
			out = append(out, c.notes.f1())
		}
		if d, ok := c.drumScore(given[i], all.hitRate()); ok {
			out = append(out, d)
		}
		if given[i].lines > 0 {
			out = append(out, c.bars())
		}
	}
	return out
}

// noteCount is a rip's notes against a truth's: those that hit, those given and
// those heard
type noteCount struct{ hits, given, heard int }

// everyNote counts an agreement's notes by pitch and onset, whatever line holds
// each: the count a search or a round is picked by
func everyNote(ag audio.Agreement) noteCount {
	return noteCount{ag.All.Matched, ag.All.B, ag.All.A}
}

// agreed counts an agreement's notes within their lines, every note once: the
// melody register's hits as one line or as its two lines apart, whichever is
// more, and the bass's
func agreed(ag audio.Agreement) noteCount {
	reg, bass := ag.Register, ag.Lines[2]
	return noteCount{max(reg.Matched, ag.Lines[0].Matched+ag.Lines[1].Matched) + bass.Matched, reg.B + bass.B, reg.A + bass.A}
}

func (c noteCount) plus(o noteCount) noteCount {
	return noteCount{c.hits + o.hits, c.given + o.given, c.heard + o.heard}
}

// f1 is twice the hits over the notes given and heard
func (c noteCount) f1() float64 { return 2 * float64(c.hits) / float64(max(c.given+c.heard, 1)) }

// conditionCounts is each of n conditions' count under p, its recordings' as
// one; a rip that fails hears none of what its truth gives
func conditionCounts(takes []*truthTake, n int, p audio.RipParams, lead audio.LeadRule) []truthCount {
	sums := make([]truthCount, n)
	for _, t := range takes {
		c, _ := t.count(p, lead)
		sums[t.cond] = sums[t.cond].plus(c)
	}
	return sums
}
