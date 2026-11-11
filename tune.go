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
	"io/fs"
	"maps"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/lixenwraith/audio"
	"github.com/lixenwraith/genetic"
)

// tuneOptions sizes a search: the loops rendered and their bars, the generations
// run over a pool of candidates, the seed every draw follows, whether the loops
// are augmented, a room, a mix and random voices behind each, whether parts are
// generated, beats, melodies and bass lines the bank never had, and whether a
// share of the loops is in another meter than the bank's 4/4
type tuneOptions struct {
	loops, bars, generations, pool int
	seed                           uint64
	aug, gen                       bool
	meters                         bool           // a third of the loops in 3/4 or 6/8, every part generated in it
	nowav                          bool           // an export keeps the frames and the labels only
	codec                          bool           // an export sends a third of its loops through a lossy codec and back
	tempos                         []int          // the tempos the loops cycle through; tuneTempos when nil
	part, parts                    int            // an export renders the loops i%parts == part; all when parts is 0
	skip                           func(int) bool // the loops drawn but not rendered
	samples                        *sampleSet     // recorded notes some tonal slots play on; none when nil
	midis                          []string       // arrangements most loops play bars of, in place of drawn patterns
	dense                          bool           // those arrangements play every part, the inner ones softer, on instruments of their own
	guard                          float64        // the F1 a set tune truth tries may take from any condition's before it is refused
	frames                         string         // where tune truth keeps each recording's frames between runs; "" hears them every run
}

// tuneLoop is one rendered loop, what made it, the rip a perfect listener makes
// of it, and the voices its tonal tracks were played with
type tuneLoop struct {
	audio    []float64
	want     *audio.Rip
	slots    [audio.MusicSlots]audio.PatternID
	bpm      int
	meter    audio.Meter
	root     int
	scale    audio.ScaleID
	voices   map[string]*audio.VoiceDef
	families map[string]int                 // each voice's family, an index into audio.RipFamilies
	tuning   float64                        // cents off A440 the loop was rendered at
	sampled  [audio.MusicSlots]*sampledLine // the slots played on recorded notes, nil for the engine's voice
}

// tempos spread the corpus over the sequencer's range
var tuneTempos = []int{96, 112, 128, 140, 156, 176}

// tuneMeters are the meters other than 4/4 a corpus of meters draws a third of
// its loops in: three quarters, and two dotted quarters of three eighths
var tuneMeters = []audio.Meter{{Num: 3, Den: 4}, {Num: 6, Den: 8}}

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
			return fmt.Errorf("tune truth tunes the model's decoders, and no %s is loaded", modelFile)
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

// corpus renders loops of the registered rhythm, melody and bass patterns, drawn
// by the seed over the tempos and the keys, each with its expected rip, all held
// at once: what a search over them wants
func (s *Session) corpus(o tuneOptions) ([]tuneLoop, error) {
	loops := make([]tuneLoop, 0, o.loops)
	err := s.eachLoop(o, func(_ int, l tuneLoop) error {
		loops = append(loops, l)
		return nil
	})
	return loops, err
}

// eachLoop renders the corpus's loops one at a time to visit, so an export of
// thousands holds one in memory
func (s *Session) eachLoop(o tuneOptions, visit func(int, tuneLoop) error) error {
	var byRole [audio.MusicSlots][]audio.PatternID
	for _, p := range audio.RegisteredPatterns() {
		if strings.HasPrefix(p.Name, "corpus_") {
			continue // a generated part of an earlier loop, not the bank's
		}
		if p.ID == audio.PatternMelodyGen {
			continue // the sequencer writes its lead bar by bar as it plays: no label reads it
		}
		switch p.Role {
		case audio.RoleRhythm, audio.RoleMelody, audio.RoleBass:
			byRole[p.Role-1] = append(byRole[p.Role-1], p.ID)
		}
	}
	if len(byRole[0]) == 0 || len(byRole[1]) == 0 || len(byRole[3]) == 0 {
		return errors.New("tune needs registered rhythm, melody and bass patterns (builtin pattern, apply)")
	}
	rng := rand.New(rand.NewPCG(o.seed, 0x5bd1e995))
	for i := range o.loops {
		var slots [audio.MusicSlots]audio.PatternID
		for slot, pool := range byRole {
			if len(pool) > 0 {
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
		// the meter, the augmentation and the generated parts draw from streams
		// of their own, so a loop's drawn patterns and key are the same with and
		// without, and a loop in 4/4 the same with meters and without
		var meter audio.Meter
		if o.meters {
			mrng := rand.New(rand.NewPCG(o.seed^0x3e7e3e7e, uint64(i)+1))
			if mrng.Float64() < 1.0/3 {
				meter = tuneMeters[mrng.IntN(len(tuneMeters))]
			}
		}
		other := meter != audio.Meter{} // the bank's patterns are 4/4: every part is generated
		if o.gen || other {
			grng := rand.New(rand.NewPCG(o.seed^0x5a5a5a5a, uint64(i)+1))
			degrees := len(audio.ScaleIntervals(scale))
			for _, g := range []struct {
				slot int
				make func(*rand.Rand, int, audio.Meter) *audio.Pattern
			}{{0, genBeat}, {1, genMelody}, {2, genAccompaniment}, {3, genBass}} {
				if grng.Float64() >= 0.6 && (!other || g.slot == 2) {
					continue
				}
				p := g.make(grng, degrees, meter)
				p.ID = audio.PatternIDByName(p.Name) // the last loop's, replaced in place
				if id, err := s.eng.DefinePattern(p); err == nil {
					slots[g.slot] = id
				} else if other {
					slots[g.slot] = audio.PatternSilence
				}
			}
			// much of a recording's music has no drums or no bass, a solo piano
			// neither, where the bank's loops always had both under the melody
			if grng.Float64() < 0.3 {
				slots[0] = audio.PatternSilence
			}
			if grng.Float64() < 0.15 {
				slots[3] = audio.PatternSilence
			}
		}
		if len(o.midis) > 0 {
			mrng := rand.New(rand.NewPCG(o.seed^0x6d696469, uint64(i)+1))
			if mrng.Float64() < 0.75 {
				if got, ok := s.midiLoop(o, mrng); ok {
					slots, bpm, root, scale, meter = got.slots, got.bpm, got.root, got.scale, got.meter
				}
			}
		}
		arng := rand.New(rand.NewPCG(o.seed^0xa5a5a5a5, uint64(i)+1))
		voices, families, restore := s.loopVoices(slots, o.aug, arng)
		cents := 0.0
		if o.aug {
			cents = arng.Float64()*100 - 50 // a recording's pitch standard, within a quarter tone of A440
		}
		played, srng := slots, (*rand.Rand)(nil)
		var sampled [audio.MusicSlots]*sampledLine
		if o.samples != nil {
			srng = rand.New(rand.NewPCG(o.seed^0x3c3c3c3c, uint64(i)+1))
			var err error
			if played, sampled, err = o.samples.draw(slots, meter, o.bars, bpm, root, scale, cents, srng); err != nil {
				restore()
				return err
			}
		}
		s.eng.SetTuning(cents)
		buf, err := s.eng.RenderMusic(played, bpm, meter, o.bars, root, scale, nil, int64(i+1))
		s.eng.SetTuning(0)
		restore()
		if err != nil {
			return err
		}
		if o.samples != nil {
			buf = mixSampled(buf, sampled, srng)
		}
		if o.aug {
			buf = degrade(buf, arng)
		}
		want := audio.ExpectedRip(slots, bpm, meter, o.bars, root, scale, nil)
		if err := visit(i, tuneLoop{audio: buf, want: want, slots: slots, bpm: bpm, meter: meter, root: root, scale: scale, voices: voices, families: families, tuning: cents, sampled: sampled}); err != nil {
			return err
		}
	}
	return nil
}

// midiPlay is a stretch of an arrangement as a loop's parts
type midiPlay struct {
	slots [audio.MusicSlots]audio.PatternID
	bpm   int
	root  int
	scale audio.ScaleID
	meter audio.Meter
}

// inner spreads an arrangement's inner notes over up to three tracks by octave,
// each up to 24 dB under the file's own level, so a soft line sounds under a ring
func inner(tracks []audio.Track, rng *rand.Rand) []audio.Track {
	n := 1 + rng.IntN(3)
	out := make([]audio.Track, n)
	gain := make([]float64, n)
	for t := range out {
		out[t] = audio.Track{Humanize: tracks[0].Humanize}
		gain[t] = math.Pow(10, -24*rng.Float64()/20)
	}
	for _, tr := range tracks {
		for _, ev := range tr.Events {
			t := ((ev.Oct % n) + n) % n
			ev.Vel = max(ev.Vel*gain[t], 0.08)
			out[t].Events = append(out[t].Events, ev)
		}
	}
	return slices.DeleteFunc(out, func(t audio.Track) bool { return len(t.Events) == 0 })
}

// midiLoop is a random stretch of a random file's beat, melody, second line and
// bass, moved up to half an octave so the file's key says nothing; a file that can't
// be read or holds no such stretch passes the draw to another.
func (s *Session) midiLoop(o tuneOptions, rng *rand.Rand) (midiPlay, bool) {
	for try := 0; try < 8; try++ {
		f, err := os.Open(o.midis[rng.IntN(len(o.midis))])
		if err != nil {
			continue
		}
		score, err := audio.ReadMIDI(f)
		f.Close()
		if err != nil {
			continue
		}
		if o.dense { // the further parts play as the under line, spread by octave (inner)
			for i := range score.Notes {
				if n := &score.Notes[i]; n.Further {
					n.Further, n.Bass, n.Second = false, false, true
				}
			}
		}
		r, _ := score.Rip()
		meter, known := r.Meter, r.Meter.Steps() == 16
		if known {
			meter = audio.Meter{}
		} else if _, known = beatsByMeter[meter]; !known {
			continue
		}
		if len(r.Bars) < o.bars {
			continue
		}
		lo, sounding := 0, false
		for pick := 0; pick < 8 && !sounding; pick++ {
			lo = rng.IntN(len(r.Bars) - o.bars + 1)
			for b := lo; b < lo+o.bars && !sounding; b++ {
				sounding = len(r.Bars[b].Melody)+len(r.Bars[b].Under) > 0
			}
		}
		if !sounding {
			continue
		}
		define := func(p *audio.Pattern) audio.PatternID {
			p.ID = audio.PatternIDByName(p.Name)
			id, err := s.eng.DefinePattern(p)
			if err != nil {
				return audio.PatternSilence
			}
			return id
		}
		var play midiPlay
		for slot := range play.slots {
			play.slots[slot] = audio.PatternSilence
		}
		for _, part := range []audio.RipPart{audio.RipBeat, audio.RipMelody, audio.RipBass} {
			d := r.Pattern("corpus_midi_"+audio.RipPartNames[part], "corpus", part, lo, lo+o.bars)
			if d == nil {
				continue
			}
			p, err := d.Pattern()
			if err != nil {
				continue
			}
			if part == audio.RipMelody && len(p.Tracks) > 1 {
				tonal := audio.TonalInstruments()
				under := *p
				under.Name, under.Tracks = "corpus_midi_under", slices.Clone(p.Tracks[1:])
				if o.dense {
					under.Tracks = inner(under.Tracks, rng)
				}
				for t := range under.Tracks {
					under.Tracks[t].Instr = tonal[rng.IntN(len(tonal))]
					if under.Tracks[t].Instr == audio.InstrBass {
						under.Tracks[t].Instr = audio.InstrLead
					}
				}
				p.Tracks = p.Tracks[:1]
				play.slots[2] = define(&under)
			}
			play.slots[audio.RipPartSlots[part]] = define(p)
		}
		play.bpm, play.root, play.scale, play.meter = r.BPM, r.Root+rng.IntN(12)-6, r.Scale, meter
		return play, true
	}
	return midiPlay{}, false
}

// beatPlaces is where genBeat strikes in a meter's bar: the kicks always played,
// the kicks played now and then, the snares nearly always, the ghost snares, and
// the hats' spacings, one drawn
type beatPlaces struct{ kicks, extras, snares, ghosts, every []int }

// beatsByMeter strike 4/4 on one and three against two and four, 3/4 on one
// against two and three, and 6/8 on its first dotted beat against its second,
// its hats on the eighths or the dotted beats
var beatsByMeter = map[audio.Meter]beatPlaces{
	{}:               {kicks: []int{0, 8}, extras: []int{3, 6, 10, 11, 14}, snares: []int{4, 12}, ghosts: []int{7, 15}, every: []int{2, 4}},
	{Num: 3, Den: 4}: {kicks: []int{0}, extras: []int{6, 10, 11}, snares: []int{4, 8}, ghosts: []int{7, 11}, every: []int{2, 4}},
	{Num: 6, Den: 8}: {kicks: []int{0}, extras: []int{4, 8, 10}, snares: []int{6}, ghosts: []int{2, 11}, every: []int{2, 6}},
}

// genBeat is a beat the bank never had, a bar of meter m: in 4/4 kicks on one
// and three with extras, the snare on two and four with a ghost now and then,
// hats on the eighths or the quarters (beatsByMeter), velocities drawn;
// registered under one name a loop at a time
func genBeat(rng *rand.Rand, _ int, m audio.Meter) *audio.Pattern {
	at, bar := beatsByMeter[m], m.Steps()
	var kick, snare, hat []audio.Step
	for _, pos := range at.kicks {
		kick = append(kick, audio.Step{Pos: pos, Vel: 0.9 + rng.Float64()*0.1})
	}
	for _, pos := range at.extras {
		if rng.Float64() < 0.25 {
			kick = append(kick, audio.Step{Pos: pos, Vel: 0.6 + rng.Float64()*0.3})
		}
	}
	for _, pos := range at.snares {
		if rng.Float64() < 0.9 {
			snare = append(snare, audio.Step{Pos: pos, Vel: 0.8 + rng.Float64()*0.2})
		}
	}
	for _, pos := range at.ghosts {
		if rng.Float64() < 0.2 {
			snare = append(snare, audio.Step{Pos: pos, Vel: 0.4})
		}
	}
	every := at.every[rng.IntN(len(at.every))]
	for pos := 0; pos < bar; pos += every {
		if rng.Float64() < 0.85 {
			hat = append(hat, audio.Step{Pos: pos, Vel: 0.5 + rng.Float64()*0.3})
		}
	}
	p := &audio.Pattern{Name: "corpus_beat", Steps: bar, Meter: m, Role: audio.RoleRhythm}
	for _, tr := range []audio.Track{{Instr: audio.InstrKick, Events: kick}, {Instr: audio.InstrSnare, Events: snare}, {Instr: audio.InstrHihat, Events: hat}} {
		if len(tr.Events) > 0 {
			p.Tracks = append(p.Tracks, tr)
		}
	}
	return p
}

// genMelody is a melody the bank never had, one or two bars of meter m: a walk
// over the key's degrees within two octaves, lengths of one to four steps with
// a rest now and then, or in half the loops a slow line of two to eight, on a
// tonal instrument drawn from the registered ones. A compound meter's lengths
// are its eighths, quarters and dotted quarters, so the line keeps to its eighths.
func genMelody(rng *rand.Rand, degrees int, m audio.Meter) *audio.Pattern {
	steps := m.Steps() * (1 + rng.IntN(2))
	tonal := audio.TonalInstruments()
	instr := tonal[rng.IntN(len(tonal))]
	if instr == audio.InstrBass {
		instr = audio.InstrLead
	}
	fast, slow := []int{1, 1, 2, 2, 3, 4}, []int{2, 2, 4, 4, 6, 8}
	if m.Compound() {
		fast, slow = []int{2, 2, 2, 4, 4, 6}, []int{2, 4, 4, 6, 6, 12}
	}
	lengths := fast
	if rng.Float64() < 0.5 {
		lengths = slow
	}
	deg, oct := rng.IntN(degrees), 1+rng.IntN(2)
	var evs []audio.Step
	for pos := 0; pos < steps; {
		dur := lengths[rng.IntN(len(lengths))]
		if rng.Float64() < 0.2 {
			pos += dur
			continue
		}
		deg += rng.IntN(5) - 2
		for deg < 0 {
			deg, oct = deg+degrees, oct-1
		}
		for deg >= degrees {
			deg, oct = deg-degrees, oct+1
		}
		oct = min(max(oct, 1), 2)
		evs = append(evs, audio.Step{Pos: pos, Vel: 0.5 + rng.Float64()*0.5, Deg: deg, Oct: oct, Dur: min(dur, steps-pos)})
		pos += dur
	}
	if len(evs) == 0 {
		evs = []audio.Step{{Pos: 0, Vel: 0.8, Deg: 0, Oct: 1, Dur: 4}}
	}
	soften(rng, evs, steps)
	return &audio.Pattern{Name: "corpus_melody", Steps: steps, Meter: m, Role: audio.RoleMelody, Tracks: []audio.Track{{Instr: instr, Events: evs}}}
}

// soften plays half of a third of the generated parts 6 to 18 dB softer, as a
// pianist plays a phrase, so the model hears soft strikes beside loud ones
func soften(rng *rand.Rand, evs []audio.Step, steps int) {
	if rng.Float64() >= 1.0/3 {
		return
	}
	gain, from := math.Pow(10, -(6+12*rng.Float64())/20), rng.IntN(steps/2+1)
	for i := range evs {
		if evs[i].Pos >= from && evs[i].Pos < from+steps/2 {
			evs[i].Vel *= gain
		}
	}
}

// genAccompaniment is an accompaniment the bank never had, for the free slot: a
// broken chord, the key's first, third and fifth with the seventh now and then,
// in eighths or sixteenths that change pitch from note to note, up, down or
// alternating, softer than a lead, on a tonal instrument drawn from the
// registered ones, one or two bars of meter m: the line that runs under a slow
// melody in much music, which the rip has to part from it
func genAccompaniment(rng *rand.Rand, degrees int, m audio.Meter) *audio.Pattern {
	steps := m.Steps() * (1 + rng.IntN(2))
	tonal := audio.TonalInstruments()
	instr := tonal[rng.IntN(len(tonal))]
	if instr == audio.InstrBass {
		instr = audio.InstrLead
	}
	chord := []int{0, 2, 4}
	if rng.Float64() < 0.3 && degrees > 6 {
		chord = append(chord, 6)
	}
	dur := 1 + rng.IntN(2) // sixteenths or eighths
	base := 1 + rng.IntN(2)
	order := rng.IntN(3) // up, down, alternating
	var evs []audio.Step
	for i, pos := 0, 0; pos < steps; i, pos = i+1, pos+dur {
		k := i % len(chord)
		switch order {
		case 1:
			k = len(chord) - 1 - k
		case 2:
			if i%2 == 1 {
				k = len(chord) - 1 - i/2%len(chord)
			} else {
				k = i / 2 % len(chord)
			}
		}
		oct := base + i/len(chord)%2 // the chord in one octave, then the next
		deg := chord[k]
		if deg >= degrees {
			deg = 0
		}
		evs = append(evs, audio.Step{Pos: pos, Vel: 0.4 + rng.Float64()*0.3, Deg: deg, Oct: min(oct, 2), Dur: dur})
	}
	soften(rng, evs, steps)
	return &audio.Pattern{Name: "corpus_accompaniment", Steps: steps, Meter: m, Role: audio.RoleFill, Tracks: []audio.Track{{Instr: instr, Events: evs}}}
}

// genBass is a bass line the bank never had, a bar of meter m: the root and its
// neighbours below the fifth, in the root's octave, in halves, quarters and
// eighths, or in a compound meter dotted halves, dotted quarters, quarters and
// eighths
func genBass(rng *rand.Rand, degrees int, m audio.Meter) *audio.Pattern {
	steps, lengths := m.Steps(), []int{2, 4, 4, 8}
	if m.Compound() {
		lengths = []int{2, 4, 6, 12}
	}
	var evs []audio.Step
	for pos := 0; pos < steps; {
		dur := lengths[rng.IntN(len(lengths))]
		deg := []int{0, 0, 0, 4, 3, 2}[rng.IntN(6)]
		if deg >= degrees {
			deg = 0
		}
		evs = append(evs, audio.Step{Pos: pos, Vel: 0.7 + rng.Float64()*0.3, Deg: deg, Oct: 0, Dur: min(dur, steps-pos)})
		pos += dur
	}
	return &audio.Pattern{Name: "corpus_bass", Steps: steps, Meter: m, Role: audio.RoleBass, Tracks: []audio.Track{{Instr: audio.InstrBass, Events: evs}}}
}

// loopVoices is the voice each tonal instrument the slotted patterns name plays
// with and the family it sounds as: the registered one, or under aug one of random
// genes near it, a third of the time on another wave, and in a third of the loops
// every release but the bass's three to ten times as long, as under a piano's
// pedal; registered for the render and put back by restore
func (s *Session) loopVoices(slots [audio.MusicSlots]audio.PatternID, aug bool, rng *rand.Rand) (map[string]*audio.VoiceDef, map[string]int, func()) {
	voices, families, known := map[string]*audio.VoiceDef{}, map[string]int{}, familyVoices()
	var originals []*audio.VoiceDef
	pedal := 1.0 // the labels stay the notes as played, however long they ring
	if aug && rng.Float64() < 1.0/3 {
		pedal = 3 + rng.Float64()*7
	}
	for _, id := range slots {
		p := audio.GetPattern(id)
		if p == nil {
			continue
		}
		for _, tr := range p.Tracks {
			if tr.Instr.IsDrum() {
				continue
			}
			name := audio.InstrumentName(tr.Instr)
			v := audio.VoiceDefByName(name)
			if v == nil || voices[name] != nil {
				continue
			}
			voices[name], families[name] = v, voiceFamily(known, name, v)
			if !aug {
				continue
			}
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
			family := voiceFamily(known, name, r)
			if rng.Float64() < 0.15 { // distorted, as a guitar through an amp
				r.Drive, r.Level = 2+rng.Float64()*14, r.Level*0.6
				family = familyIndex("unknown")
			}
			if tr.Instr != audio.InstrBass {
				r.Release = min(r.Release*pedal, 3)
			}
			r.VibratoHz, r.VibratoCents, r.VibratoDelay, r.Vowel = v.VibratoHz, v.VibratoCents, v.VibratoDelay, v.Vowel // not genes; kept from the registered voice
			r.Name = name
			if audio.ValidateVoice(r) != nil {
				continue
			}
			if _, err := s.eng.DefineVoice(r); err != nil {
				continue
			}
			originals = append(originals, v)
			voices[name], families[name] = r, family
		}
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
	// a recording's hiss, which a studio's Billie Jean carries 39 dB under its
	// median frame, sits bare over a synthesizer's empty treble: half the loops have none
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
	Vel    float64   `json:"vel"`
	Wave   int       `json:"wave"`   // index into the waves of the manifest; -1 unknown
	Family int       `json:"family"` // index into the families of the manifest; -1 unlabelled
	Voice  []float64 `json:"voice,omitempty"`
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
}

// corpusManifest names the corpus's layout for the trainer
type corpusManifest struct {
	Version    int      `json:"version"`
	Loops      int      `json:"loops"`
	Rate       int      `json:"rate"`
	Hop        int      `json:"hop"`
	MelBands   int      `json:"mel_bands"`
	MinMIDI    int      `json:"min_midi"`
	Pitches    int      `json:"pitches"`
	Drums      []string `json:"drums"`
	Waves      []string `json:"waves"`
	Families   []string `json:"families"`
	VoiceGenes int      `json:"voice_genes"`
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
		case a[n-1] == "meters":
			o.meters = true
		case a[n-1] == "nowav":
			o.nowav = true
		case a[n-1] == "codec":
			o.codec = true
		case a[n-1] == "dense":
			o.dense = true
		case n > 2 && a[n-2] == "seed":
			v, err := strconv.ParseUint(a[n-1], 10, 64)
			if err != nil {
				return fmt.Errorf("seed %q: want a number", a[n-1])
			}
			o.seed = v
			a = a[:n-1]
		case n > 2 && a[n-2] == "midi":
			err := filepath.WalkDir(s.path(a[n-1]), func(path string, d fs.DirEntry, err error) error {
				if err == nil && !d.IsDir() && strings.EqualFold(filepath.Ext(path), ".mid") {
					o.midis = append(o.midis, path)
				}
				return err
			})
			if err != nil {
				return err
			}
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
			k, of, _ := strings.Cut(a[n-1], "/")
			part, err1 := strconv.Atoi(k)
			parts, err2 := strconv.Atoi(of)
			if err1 != nil || err2 != nil || part < 0 || part >= parts {
				return fmt.Errorf("part %q: want k/n, k from 0 below n", a[n-1])
			}
			o.part, o.parts = part, parts
			a = a[:n-1]
		default:
			more = false
			continue
		}
		a = a[:len(a)-1]
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

// corpusFrom writes a corpus of recordings and the MIDI files they were played
// from, one loop each, listed by a CSV with audio_filename and midi_filename
// columns, and split, duration and family when present: the frames as heard, the
// notes as the MIDI plays them, the voices unknown. A loop on disk is kept, known
// by its source, so a run again goes on where it stopped and a batch after it.
func (s *Session) corpusFrom(dir, list string, a []string) error {
	split, hours, codec := "", math.Inf(1), false
	for len(a) > 0 {
		if a[0] == "codec" {
			codec, a = true, a[1:]
			continue
		}
		if len(a) < 2 || a[0] != "split" && a[0] != "hours" {
			return fmt.Errorf("%q: want split <name>, hours <h> or codec", a[0])
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
	// a share of a large set may be all that is on disk
	listed := len(rows)
	rows = slices.DeleteFunc(rows, func(r []string) bool {
		_, err1 := os.Stat(at(r[wav]))
		_, err2 := os.Stat(at(r[mid]))
		return err1 != nil || err2 != nil
	})
	missing := listed - len(rows)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	loopFile := func(i int) string { return filepath.Join(dir, fmt.Sprintf("loop_%03d", i)) }
	taken, held, frames := map[int]bool{}, map[string]bool{}, 0
	labels, _ := filepath.Glob(filepath.Join(dir, "loop_*.json"))
	for _, name := range labels {
		i, err := strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(filepath.Base(name), "loop_"), ".json"))
		data, rerr := os.ReadFile(name)
		var l corpusLoop
		if err != nil || rerr != nil || json.Unmarshal(data, &l) != nil {
			continue // a label cut short is written again
		}
		taken[i], held[l.Source], frames = true, true, frames+l.Frames
	}
	rows = slices.DeleteFunc(rows, func(r []string) bool { return held[r[wav]] })
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
		if sum >= hours*3600 {
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
		mu    sync.Mutex
		wg    sync.WaitGroup
		first error
		work  = make(chan job)
	)
	for range min(4, runtime.NumCPU()) { // a recording's samples are hundreds of megabytes
		wg.Go(func() {
			for j := range work {
				var coder *rand.Rand
				if codec {
					coder = rand.New(rand.NewPCG(0xc0dec, uint64(j.at)+1))
				}
				n, err := corpusRecording(loopFile(j.at), at(j.row[wav]), j.row[wav], family(j.row), j.score, coder)
				mu.Lock()
				frames += n
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
	m := corpusManifest{Version: 1, Loops: loops, Rate: audio.RipRate, Hop: audio.RipHop, MelBands: audio.RipMelBands,
		MinMIDI: audio.RipMinMIDI, Pitches: audio.RipPitches, Drums: []string{"kick", "snare", "hihat"}, Waves: audio.VoiceWaves[:], Families: audio.RipFamilies[:], VoiceGenes: audio.VoiceGeneCount}
	if err := writeJSON(filepath.Join(dir, "corpus.json"), m); err != nil {
		return err
	}
	how := ""
	if codec {
		how = ", a third through a lossy codec"
	}
	fmt.Fprintf(s.out, "wrote %d recording(s) to %s from %s%s, %d loop(s) and %.1f hours in all; %d listed were not on disk\n",
		len(jobs), dir, list, how, loops, float64(frames)*audio.RipFrameSec/3600, missing)
	if len(skipped) > 0 {
		fmt.Fprintf(s.out, "skipped %d whose MIDI has no notes and no kick, snare or hihat, as %s\n", len(skipped), skipped[0])
	}
	return nil
}

// corpusRecording writes a recording's frames, through a codec when coder is
// set, and its score as one loop, the frames it holds returned: its notes played
// by family, or each by its program's (programFamily) when family is -1, its hits
// and other strikes by the drums, a strike of no line (a tom, a crash) an onset,
// drum -1 and midi 0.
func corpusRecording(base, wav, source string, family int, score *audio.MIDIScore, coder *rand.Rand) (int, error) {
	f, err := os.Open(wav)
	if err != nil {
		return 0, err
	}
	x, rate, err := audio.ReadWAV(f)
	f.Close()
	if err != nil {
		return 0, fmt.Errorf("%s: %w", wav, err)
	}
	if coder != nil {
		if x, err = codecRoundTrip(x, rate, coder); err != nil {
			return 0, err
		}
	}
	mel := audio.MelFrames(x, rate)
	if err := writeMel(base+".mel", mel); err != nil {
		return 0, err
	}
	frame := func(sec float64) int { return int(sec/audio.RipFrameSec + 0.5) }
	lab := corpusLoop{Frames: len(mel), Source: source}
	for _, n := range score.Notes {
		f := family
		if f < 0 {
			f = programFamily(n.Program)
		}
		lab.Events = append(lab.Events, corpusEvent{Frame: frame(n.Start), End: frame(n.End), Drum: -1, MIDI: n.MIDI, Vel: n.Vel, Wave: -1, Family: f})
	}
	for _, h := range append(slices.Clip(score.Hits), score.Struck...) {
		lab.Events = append(lab.Events, corpusEvent{Frame: frame(h.Start), End: frame(h.Start), Drum: int(h.Drum), Vel: h.Vel, Wave: -1, Family: familyIndex("drums")})
	}
	lab.Events = slices.DeleteFunc(lab.Events, func(e corpusEvent) bool { return e.Frame >= len(mel) })
	return len(mel), writeJSON(base+".json", lab)
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

// exportCorpus writes loops into dir: loop_NNN.wav as rendered, loop_NNN.mel as
// the log-mel frames a front-end model hears (frames, bands, then float32 rows,
// little-endian), loop_NNN.json as the events that sound in it with their frames
// and voices, and corpus.json naming the layout. A loop whose label is on disk is
// kept, so an export run again goes on where it stopped; a part writes no layout.
func (s *Session) exportCorpus(dir string, o tuneOptions) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	loopFile := func(i int) string { return filepath.Join(dir, fmt.Sprintf("loop_%03d", i)) }
	written, kept := 0, 0
	o.skip = func(i int) bool {
		if o.parts > 0 && i%o.parts != o.part {
			return true
		}
		_, err := os.Stat(loopFile(i) + ".json") // written last, so its loop is whole
		if err == nil {
			kept++
		}
		return err == nil
	}
	sampledLines, tonalLines, metered := 0, 0, 0
	err := s.eachLoop(o, func(i int, l tuneLoop) error {
		written++
		if l.meter != (audio.Meter{}) {
			metered++
		}
		base := loopFile(i)
		for k, line := range l.sampled[1:] {
			tonalLines += min(len(slotEvents(l.slots, k+1, l.meter, o.bars, l.root, l.scale)), 1)
			if line != nil {
				sampledLines++
			}
		}
		if o.codec {
			coded, err := codecRoundTrip(l.audio, audio.AudioSampleRate, rand.New(rand.NewPCG(o.seed^0xc0dec, uint64(i)+1)))
			if err != nil {
				return err
			}
			l.audio = coded
		}
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
		mel := audio.MelFrames(l.audio, audio.AudioSampleRate)
		if err := writeMel(base+".mel", mel); err != nil {
			return err
		}
		lab := corpusLoop{BPM: l.bpm, Root: l.root, Scale: int(l.scale), Bars: o.bars, Meter: l.meter.Doc(), Frames: len(mel), Tuning: l.tuning}
		stepSec := float64(audio.SamplesPerStep(l.bpm)) / audio.AudioSampleRate
		for _, e := range loopEvents(l.slots, l.meter, o.bars, l.root, l.scale) {
			start := float64(e.Bar*l.meter.Steps()+e.Step) * stepSec
			ev := corpusEvent{Frame: int(start/audio.RipFrameSec + 0.5), Drum: -1, Vel: e.Vel, Wave: -1, Family: -1}
			if e.Instr.IsDrum() {
				if e.Instr >= audio.InstrClap {
					continue
				}
				ev.Drum, ev.End, ev.Family = int(e.Instr), ev.Frame, familyIndex("drums")
			} else {
				ev.MIDI = e.MIDI
				ev.End = int((start+float64(e.Dur)*stepSec)/audio.RipFrameSec + 0.5)
				name := audio.InstrumentName(e.Instr)
				if line := l.sampled[e.slot]; line != nil {
					ev.End = int(line.ends[e.n]/audio.RipFrameSec + 0.5) // where the recording let it go
					ev.Family = line.family
				} else if v := l.voices[name]; v != nil {
					ev.Wave = max(slices.Index(audio.VoiceWaves[:], cmp.Or(v.Wave, "sine")), 0)
					ev.Voice, ev.Family = v.Genes(), l.families[name]
				}
			}
			if ev.Frame < len(mel) {
				lab.Events = append(lab.Events, ev)
			}
		}
		return writeJSON(base+".json", lab)
	})
	if err != nil {
		return err
	}
	if o.parts == 0 {
		m := corpusManifest{Version: 1, Loops: o.loops, Rate: audio.RipRate, Hop: audio.RipHop, MelBands: audio.RipMelBands,
			MinMIDI: audio.RipMinMIDI, Pitches: audio.RipPitches, Drums: []string{"kick", "snare", "hihat"}, Waves: audio.VoiceWaves[:], Families: audio.RipFamilies[:], VoiceGenes: audio.VoiceGeneCount}
		if err := writeJSON(filepath.Join(dir, "corpus.json"), m); err != nil {
			return err
		}
	}
	how := "as rendered"
	if o.aug {
		how = "through a room, a mix, voices of random genes and a tuning off A440"
	}
	if o.gen {
		how += ", with generated beats, melodies and bass lines"
	}
	if o.meters {
		how += fmt.Sprintf(", %d in 3/4 or 6/8 with every part generated", metered)
	}
	if o.codec {
		how += ", a third through a lossy codec"
	}
	if o.samples != nil {
		how += fmt.Sprintf(", %d of %d tonal line(s) on the recorded notes in %s", sampledLines, tonalLines, o.samples.dir)
	}
	if kept > 0 {
		how += fmt.Sprintf("; %d were there", kept)
	}
	if o.parts > 0 {
		how += fmt.Sprintf("; part %d of %d, so no corpus.json", o.part, o.parts)
	}
	fmt.Fprintf(s.out, "wrote %d loop(s) of %d bar(s) to %s, %s\n", written, o.bars, dir, how)
	return nil
}

// slotEvent is an expected event with the slot it plays in and its place among that slot's
type slotEvent struct {
	audio.RipEvent
	slot, n int
}

// loopEvents is ExpectedEvents in its order, each event knowing its slot
func loopEvents(slots [audio.MusicSlots]audio.PatternID, m audio.Meter, bars, root int, scale audio.ScaleID) []slotEvent {
	var per [audio.MusicSlots][]audio.RipEvent
	for k := range slots {
		per[k] = slotEvents(slots, k, m, bars, root, scale)
	}
	var out []slotEvent
	var at [audio.MusicSlots]int
	for bar := range bars {
		for k, evs := range per {
			for ; at[k] < len(evs) && evs[at[k]].Bar == bar; at[k]++ {
				out = append(out, slotEvent{evs[at[k]], k, at[k]})
			}
		}
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

func writeMel(file string, mel [][]float32) error {
	f, err := os.Create(file)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	le := binary.LittleEndian
	bands := 0
	if len(mel) > 0 {
		bands = len(mel[0])
	}
	binary.Write(w, le, uint32(len(mel)))
	binary.Write(w, le, uint32(bands))
	for _, row := range mel {
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
	o := tuneOptions{loops: 8, bars: 4, seed: 7, gen: true}
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
	what  string // index, tempo, meter, key, kind, generated parts and front end, in columns
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

// bench grades the front ends the session has, the analysis and the model when
// one loaded, on fresh loops the engine renders with generated parts among the
// bank's, clean and augmented: the mean of each rip's score against the loop's
// expected rip, so a trained model is measured against the analysis before it
// is trusted with a recording, and under meters the mean of each meter's loops
// with the rips that read it; verbose then lists every loop and the worst three
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
					what := fmt.Sprintf("%3d %3d bpm %-4s %-3s %-14s %-9s %-23s %-8s", i, l.bpm, l.meter, noteName(l.root), scaleName(l.scale), kind, generatedParts(l.slots), f.name)
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
			for _, m := range append([]audio.Meter{{}}, tuneMeters...) {
				if t := byMeter[m]; t != nil {
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

// generatedParts names the slots a loop's generated patterns fill, bank for none
func generatedParts(slots [audio.MusicSlots]audio.PatternID) string {
	var parts []string
	for k, id := range slots {
		if p := audio.GetPattern(id); p != nil && strings.HasPrefix(p.Name, "corpus_") {
			parts = append(parts, slotRoles[k])
		}
	}
	if len(parts) == 0 {
		return "bank"
	}
	return strings.Join(parts, "+")
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
// score under zero is a set refused. It returns the best and the start's score.
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
	return pool.Members[0], base, nil
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

// geneOf is where a parameter stands in Genes' order, which names none
func geneOf(at func(*audio.RipParams) *float64) int {
	var p audio.RipParams
	*at(&p) = 1
	return slices.Index(p.Genes(), 1)
}

// tuneTruth searches modelGenes against the recordings a list names, each heard
// once through fe or read back from o.frames, the other parameters held at the
// session's: a set scores the mean of its conditions' F1 by pitch and onset, and
// one that takes more than o.guard from a condition's F1 at the start is refused;
// it says each condition's F1 before and after
func (s *Session) tuneTruth(o tuneOptions, list, file string, fe audio.RipFrontEnd) error {
	takes, conds, err := readTruthList(list, fe, o.frames, s.earSum)
	if err != nil {
		return err
	}
	cur, ranges := s.rip, audio.RipGeneRanges()
	bounds, start := make([]genetic.ParameterBounds, len(modelGenes)), make([]float64, len(modelGenes))
	for k, at := range modelGenes {
		r := ranges[geneOf(at)]
		bounds[k], start[k] = genetic.ParameterBounds{Min: r[0], Max: r[1]}, *at(&cur)
	}
	params := func(genes []float64) audio.RipParams {
		p := cur
		for k, at := range modelGenes {
			*at(&p) = genes[k]
		}
		return p
	}
	// the current parameters' rips hear every recording and read every transcript
	sums, given, read := make([]noteCount, len(conds)), 0, 0
	for _, t := range takes {
		c, err := t.count(cur, s.lead)
		if err != nil {
			return err
		}
		if c.given == 0 {
			return fmt.Errorf("%s: no notes given", t.truthFile)
		}
		t.given, given, sums[t.cond] = c.given, given+c.given, sums[t.cond].plus(c)
		if t.ear.read {
			read++
		}
	}
	if o.frames != "" {
		fmt.Fprintf(s.out, "%d recording(s) heard and their frames kept in %s, %d read back from it\n", len(takes)-read, o.frames, read)
	}
	before := f1s(sums)
	over := fmt.Sprintf("%d recording(s) of %d condition(s) against %d note(s) given, a set refused that takes over %g from a condition", len(takes), len(conds), given, o.guard)
	best, base, err := s.search(o, over, bounds, start, true, func(genes []float64) float64 {
		return guarded(conditionF1(takes, len(conds), params(genes), s.lead), before, o.guard)
	})
	if err != nil {
		return err
	}
	after := before
	if best.Score > base {
		after = conditionF1(takes, len(conds), params(best.Data), s.lead)
	}
	for c, name := range conds {
		fmt.Fprintf(s.out, "%s: F1 %.3f before, %.3f after\n", name, before[c], after[c])
	}
	return s.keepRip(file, params(best.Data), best.Score, base)
}

// guarded is a set's score: the mean of its conditions' F1, or for a set that
// takes more than guard from any condition's F1 at the start, the most it takes
// under zero, so every set that passes scores over every set refused
func guarded(f1, start []float64, guard float64) float64 {
	var sum, cost float64
	for c := range f1 {
		sum, cost = sum+f1[c], max(cost, start[c]-f1[c])
	}
	if cost > guard {
		return -cost
	}
	return sum / float64(len(f1))
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
	given          int  // the notes the truth gives, all missed by a rip that fails
	cond           int  // its condition, an index into the list's
	timed          bool // the truth's seconds are the recording's
}

// readTruthList reads a line "wav<TAB>truth[<TAB>condition[<TAB>timed]]" a
// recording, a relative path from the list's own folder, and the conditions in
// the list's order, a line that names none a condition of its own under its
// recording's name; with frames, each recording's are kept there under the
// checksums of its file and of the weight files
func readTruthList(list string, fe audio.RipFrontEnd, frames, weights string) ([]*truthTake, []string, error) {
	data, err := os.ReadFile(list)
	if err != nil {
		return nil, nil, err
	}
	at := func(p string) string {
		if filepath.IsAbs(p) {
			return p
		}
		return filepath.Join(filepath.Dir(list), p)
	}
	if frames != "" {
		if err := os.MkdirAll(frames, 0o755); err != nil {
			return nil, nil, err
		}
	}
	var takes []*truthTake
	var conds []string
	for i, line := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		f := strings.Split(line, "\t")
		for k := range f {
			f[k] = strings.TrimSpace(f[k])
		}
		if len(f) < 2 || len(f) > 4 || f[0] == "" || f[1] == "" || len(f) == 4 && f[3] != "timed" && f[3] != "" {
			return nil, nil, fmt.Errorf("%s line %d: want wav<TAB>truth[<TAB>condition[<TAB>timed]]", list, i+1)
		}
		wav, truth, cond := f[0], f[1], f[0]
		if len(f) > 2 && f[2] != "" {
			cond = f[2]
		}
		recording, err := os.ReadFile(at(wav))
		if err != nil {
			return nil, nil, err
		}
		x, rate, err := audio.ReadWAV(bytes.NewReader(recording))
		if err != nil {
			return nil, nil, fmt.Errorf("%s: %w", wav, err)
		}
		given, err := os.ReadFile(at(truth))
		if err != nil {
			return nil, nil, err
		}
		t := &truthTake{wav: wav, truthFile: truth, x: audio.Resample(x, rate, audio.RipRate), ear: &heardOnce{fe: fe}, timed: len(f) == 4 && f[3] == "timed"}
		if t.cond = slices.Index(conds, cond); t.cond < 0 {
			t.cond, conds = len(conds), append(conds, cond)
		}
		if frames != "" {
			sum := sha256.Sum256(recording)
			t.ear.file = filepath.Join(frames, fmt.Sprintf("%x-%.16s.frames", sum[:8], weights))
		}
		if isMIDI(truth) {
			score, err := audio.ReadMIDI(bytes.NewReader(given))
			if err != nil {
				return nil, nil, fmt.Errorf("%s: %w", truth, err)
			}
			t.truth = func(*audio.Rip) ([]audio.RipNote, error) { return score.Notes, nil }
		} else {
			text := string(given)
			t.truth = func(r *audio.Rip) ([]audio.RipNote, error) { return transcript(text, r) }
		}
		takes = append(takes, t)
	}
	if len(takes) == 0 {
		return nil, nil, fmt.Errorf("%s: no recordings listed", list)
	}
	return takes, conds, nil
}

// count rips the take under p and counts its notes against its truth's
func (t *truthTake) count(p audio.RipParams, lead audio.LeadRule) (noteCount, error) {
	r, err := audio.Ripper{Params: p, FrontEnd: t.ear, Lead: lead}.Samples(t.x, audio.RipRate)
	if err != nil {
		return noteCount{given: t.given}, fmt.Errorf("%s: %w", t.wav, err)
	}
	given, err := t.truth(r)
	if err != nil {
		return noteCount{given: t.given}, fmt.Errorf("%s: %w", t.truthFile, err)
	}
	return everyNote(audio.MatchNotes(r.Notes, given, t.timed)), nil
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

// conditionF1 is each of n conditions' F1 under p, by pitch and onset, its
// recordings' counts as one; a rip that fails hears none of its notes
func conditionF1(takes []*truthTake, n int, p audio.RipParams, lead audio.LeadRule) []float64 {
	sums := make([]noteCount, n)
	for _, t := range takes {
		c, _ := t.count(p, lead)
		sums[t.cond] = sums[t.cond].plus(c)
	}
	return f1s(sums)
}

// f1s is each count's F1
func f1s(counts []noteCount) []float64 {
	out := make([]float64, len(counts))
	for i, c := range counts {
		out[i] = c.f1()
	}
	return out
}
