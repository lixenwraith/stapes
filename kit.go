package stapes

import (
	"cmp"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sync"

	"github.com/lixenwraith/audio"
	"github.com/lixenwraith/audio/bank"
)

// A loop's kit under the corpus's word kits: two to ten pieces, drawn from a
// stream of its own, a pass of the render playing as many as the engine has drums
const (
	kitFewest = 2
	kitMost   = 10
	kitSalt   = 0x6b697473 // "kits"
	kitPass   = int(audio.InstrClap) + 1
)

// kitPiece is one struck sound of a loop's kit: its number in the kit, the
// genes it was drawn as, the band its hit rises most in, and its sound
type kitPiece struct {
	n     int
	genes []float64
	band  int
	sound *audio.SoundDef
}

// loopKit is a loop's kit, its pieces as drawn, and the piece or pieces each
// role takes: the low pulse, the backbeat, the timekeeper, the fill's from the
// highest down and the accents
type loopKit struct {
	pieces          []*kitPiece
	low, back, time *kitPiece
	fills, accents  []*kitPiece
}

// kitSound is the name piece n of a loop's kit is registered under, written
// over by the next loop's piece n
func kitSound(n int) string { return fmt.Sprintf("corpus_piece_%d", n) }

// newKit draws a loop's kit from rng: kitFewest to kitMost pieces evenly, each
// its genes evenly over [0, 1] in StruckGeneNames' order, then its level evenly
// in dB over kitLevels; each piece is heard for its band, then cast
func newKit(rng *rand.Rand) (*loopKit, error) {
	levels, err := kitLevels()
	if err != nil {
		return nil, err
	}
	k := &loopKit{}
	for n := range kitFewest + rng.IntN(kitMost-kitFewest+1) {
		genes := make([]float64, audio.StruckGeneCount)
		for i := range genes {
			genes[i] = rng.Float64()
		}
		d := audio.StruckFromGenes(genes)
		d.Name, d.Norm = kitSound(n), levels[0]*math.Pow(levels[1]/levels[0], rng.Float64())
		band, err := strikeBand(d)
		if err != nil {
			return nil, err
		}
		k.pieces = append(k.pieces, &kitPiece{n: n, genes: genes, band: band, sound: d})
	}
	k.cast()
	return k, nil
}

// kitLevels is the span a piece's level is drawn over: the norms of the bank's
// own drums, its quietest's and its loudest's
var kitLevels = sync.OnceValues(func() ([2]float64, error) {
	shipped, err := bank.Sounds()
	if err != nil {
		return [2]float64{}, err
	}
	span := [2]float64{math.Inf(1), 0} // a drum of no norm is left out
	for i := range kitPass {
		name := audio.InstrumentName(audio.InstrumentType(i))
		for _, d := range shipped.Sound {
			if d.Name == name && d.Norm > 0 {
				span = [2]float64{min(span[0], d.Norm), max(span[1], d.Norm)}
			}
		}
	}
	if span[1] == 0 {
		return span, errors.New("kit: the bank has none of the engine's drums")
	}
	return span, nil
})

// strikeAt is the frame a piece's hit is laid on to read its band, after as
// many of silence, as audio's HitFrames lays a hit four frames in: enough that
// the band is the same laid later (TestAPiecesBandIsWhereItsStrikeRisesMost)
const strikeAt = 4

// strikeBand is the band a piece's hit rises most in where it is struck: its
// hit at full velocity, take 0, which the loop's drum plays at that velocity
// (riseBand)
func strikeBand(d *audio.SoundDef) (int, error) {
	x, err := audio.RenderHit(d, audio.SoundShape{}, 1, 0)
	if err != nil {
		return 0, err
	}
	return riseBand(x, strikeAt), nil
}

// riseBand is the band a hit x at the engine's rate rises most in laid on frame
// at: through BandFrames over the samples the sharp plane reads up to the frame
// after at, the band whose sharp plane rises most over the frame before on the
// frames onsetRise reads a strike on frame at by, the one before to the one
// after; the lowest of equals
func riseBand(x []float64, at int) int {
	lead := at * audio.RipHop * audio.AudioSampleRate / audio.RipRate
	y := make([]float64, ((at+1)*audio.RipHop+audio.RipN/2)*audio.AudioSampleRate/audio.RipRate)
	copy(y[lead:], x)
	rows := audio.BandFrames(y, audio.AudioSampleRate)
	band, top := 0, math.Inf(-1)
	for t := at - 1; t <= at+1 && t < len(rows); t++ {
		for b := range audio.RipBands {
			if up := float64(rows[t][audio.RipBands+b] - rows[t-1][audio.RipBands+b]); up > top {
				band, top = b, up
			}
		}
	}
	return band
}

// cast gives each piece its role by its place in the order of the bands, of
// equals the first drawn: the lowest the low pulse, the highest the time, the
// one half the kit up the backbeat, those under it the fill from the highest
// down and those over it the accents; of two, the higher is backbeat and time
func (k *loopKit) cast() {
	by := slices.Clone(k.pieces)
	slices.SortStableFunc(by, func(x, y *kitPiece) int { return cmp.Compare(x.band, y.band) })
	n, mid := len(by), len(by)/2
	k.low, k.back, k.time = by[0], by[mid], by[n-1]
	k.fills = slices.Clone(by[1:mid])
	slices.Reverse(k.fills)
	k.accents = by[min(mid+1, n-1) : n-1]
}

// line is the drums' line an instrument of the kit plays, nil for none
func (a *arranger) line(i audio.InstrumentType) *part {
	for _, p := range a.parts {
		if p.kind.name == "drums" && p.instr == i && !p.kit {
			return p
		}
	}
	return nil
}

// strike writes what a kit adds to the drums once enter and the window have
// cut them: its accents on the kick's bar lines, all on the first and one in
// turn after, at the kick's velocity; and its fill into the bar line after the
// last whole bar the kick strikes, the snare's own or one written as it, dealt
// from the highest fill piece down in runs as even as they fall
func (a *arranger) strike() {
	k, kick, snare := a.kit, a.line(audio.InstrKick), a.line(audio.InstrSnare)
	if kick == nil || snare == nil {
		return
	}
	steps, own := float64(a.steps), &part{kind: kick.kind, instr: audio.InstrClap, kit: true}
	last, j := math.Inf(-1), 0
	for _, n := range kick.notes {
		if math.Mod(n.at, steps) != 0 {
			continue
		}
		if n.at+steps <= a.to {
			last = n.at
		}
		for i, pc := range k.accents {
			if j == 0 || i == (j-1)%len(k.accents) {
				own.notes = append(own.notes, note{at: n.at, dur: 1, vel: n.vel, piece: pc})
			}
		}
		j++
	}
	if len(k.fills) > 0 && !math.IsInf(last, -1) {
		starts := a.pulseStarts()
		from, to := last+float64(starts[len(starts)-1]), last+steps
		var run []*note
		for i := range snare.notes {
			if n := &snare.notes[i]; n.fill && n.at >= from && n.at < to {
				run = append(run, n)
			}
		}
		if len(run) == 0 {
			at := len(own.notes)
			writeFill(own, from, to)
			for i := at; i < len(own.notes); i++ {
				run = append(run, &own.notes[i])
			}
		}
		for i, n := range run {
			n.piece = k.fills[i*len(k.fills)/len(run)]
		}
	}
	if len(own.notes) > 0 {
		a.parts = append(a.parts, own)
	}
}

// deal gives each piece a part of its own where the drums' lines were: the
// kick's strikes to the low pulse, the snare's to the backbeat, the hats' to
// the time and a strike that names a piece to it. A piece left with none
// strikes with the drums' first; a loop whose drums strike nothing keeps no kit.
func (a *arranger) deal() {
	k := a.kit
	of := map[audio.InstrumentType]*kitPiece{audio.InstrKick: k.low, audio.InstrSnare: k.back, audio.InstrHihat: k.time}
	pieces := make([]*part, len(k.pieces))
	for i, pc := range k.pieces {
		pieces[i] = &part{kind: &partKinds[0], instr: audio.InstrumentType(pc.n % kitPass), piece: pc}
	}
	var rest []*part
	at, first := -1, note{at: math.Inf(1)}
	for _, p := range a.parts {
		if p.kind.name != "drums" {
			rest = append(rest, p)
			continue
		}
		if at < 0 {
			at = len(rest)
		}
		for _, n := range p.notes {
			pc := cmp.Or(n.piece, of[p.instr])
			n.piece, n.fill = nil, false
			pieces[pc.n].notes = append(pieces[pc.n].notes, n)
			if n.at < first.at {
				first = n
			}
		}
	}
	if at < 0 || math.IsInf(first.at, 1) {
		a.kit = nil
		return
	}
	for _, p := range pieces {
		if len(p.notes) == 0 {
			p.notes = append(p.notes, first)
		}
		slices.SortStableFunc(p.notes, func(x, y note) int { return cmp.Compare(x.at, y.at) })
	}
	a.parts = slices.Insert(rest, at, pieces...)
}

// renderKit renders a loop's timeline in passes of size pieces, piece n on
// drum n mod kitPass: the first pass with the rest of the loop and each other
// with the drums alone, from one timeline and seed, summed for the loop's mix
// to take once. It registers the pieces and leaves the drums bound to the last
// pass's.
func (s *Session) renderKit(m *music, tl [][audio.MusicSlots]audio.PatternID, seed int64, size int) ([]float64, error) {
	if m.kit == nil {
		return s.eng.RenderBars(tl, m.heads, m.root, m.scale, m.prog, seed, nil)
	}
	for _, pc := range m.kit.pieces {
		var err error
		if audio.SoundDefByName(pc.sound.Name) == nil {
			_, err = s.eng.DefineSound(pc.sound) // a name new to the registry is sent to the mixer
		} else {
			_, err = audio.RegisterSound(pc.sound) // in place: the offline render reads the registry
		}
		if err != nil {
			return nil, err
		}
	}
	size = min(max(size, 1), kitPass)
	drums := make([]audio.InstrumentType, kitPass)
	for i := range drums {
		drums[i] = audio.InstrumentType(i)
	}
	var sum []float64
	for from := 0; from < len(m.kit.pieces); from += size {
		to := min(from+size, len(m.kit.pieces))
		bound, only, pass := map[string]string{}, drums, tl
		for _, pc := range m.kit.pieces[from:to] {
			bound[audio.InstrumentName(audio.InstrumentType(pc.n%kitPass))] = pc.sound.Name
		}
		if from == 0 {
			only = nil
		}
		if to-from < len(m.kit.pieces) {
			pass = m.pass(tl, from, to)
		}
		if err := s.eng.SetKit(bound); err != nil {
			return nil, err
		}
		buf, err := s.eng.RenderBars(pass, m.heads, m.root, m.scale, m.prog, seed, only)
		if err != nil {
			return nil, err
		}
		if sum == nil {
			sum = buf
			continue
		}
		for j := range sum {
			sum[j] += buf[j]
		}
	}
	return sum, nil
}

// pass is tl as a render pass plays it: each bar's pattern in the kit's slot
// cloned under corpus_pass_<bar>, the tracks of every piece outside from to to
// emptied
func (m *music) pass(tl [][audio.MusicSlots]audio.PatternID, from, to int) [][audio.MusicSlots]audio.PatternID {
	out := slices.Clone(tl)
	for b := range out {
		src := audio.GetPattern(out[b][0])
		if src == nil {
			continue
		}
		cp := *src
		cp.ID, cp.Name, cp.Tracks = audio.PatternSilence, fmt.Sprintf("corpus_pass_%d", b), slices.Clone(src.Tracks)
		for t := range cp.Tracks {
			if n := m.parts[0][t].piece.n; n < from || n >= to {
				cp.Tracks[t].Events = nil
			}
		}
		out[b][0] = audio.RegisterPattern(&cp)
	}
	return out
}
