package stapes

import (
	"cmp"
	"errors"
	"fmt"
	"iter"
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

// kitPlaces is how many places the bands are cut into, even runs from the
// lowest: the count that leaves the fewest strikes unheard at their place
// (TestTheBandsArePlacedInTheRunsThatKeepTheMostStrikesApart)
const kitPlaces = 3

// kitPiece is one struck sound of a loop's kit: its number in the kit, the
// genes it was drawn as, the band its hit rises most in, its sound, and once
// its loop strikes it its hit's rise as struck there and the place of that
type kitPiece struct {
	n     int
	genes []float64
	band  int
	sound *audio.SoundDef
	rose  []float64
	place int
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
	band, _ := riseBand(x, strikeAt)
	return band, nil
}

// placed gives each piece its place: where its hit, take 0 at the median of the
// velocities its loop strikes it at, rises most (risePlace), the take most of
// its strikes sound like, a softer one rising lower as a struck sound's layers do
func (k *loopKit) placed(parts []*part) error {
	for _, p := range parts {
		if p.piece == nil || len(p.notes) == 0 {
			continue
		}
		vels := make([]float64, len(p.notes))
		for i, n := range p.notes {
			vels[i] = n.vel
		}
		x, err := audio.RenderHit(p.piece.sound, audio.SoundShape{}, median(vels), 0)
		if err != nil {
			return err
		}
		_, p.piece.rose = riseBand(x, strikeAt)
		p.piece.place = risePlace(p.piece.rose, kitPlaces)
	}
	return nil
}

// riseBand is the band a hit x at the engine's rate rises most in laid on frame
// at: through BandFrames over the samples the sharp plane reads up to the frame
// after at, the band whose sharp plane rises most over the frame before on the
// frames onsetRise reads a strike on frame at by, the one before to the one
// after, the lowest of equals; and the energy it adds to each band (adds)
func riseBand(x []float64, at int) (band int, rose []float64) {
	lead := at * audio.RipHop * audio.AudioSampleRate / audio.RipRate
	y := make([]float64, ((at+1)*audio.RipHop+audio.RipN/2)*audio.AudioSampleRate/audio.RipRate)
	copy(y[lead:], x)
	rows := audio.BandFrames(y, audio.AudioSampleRate)
	top, rose := math.Inf(-1), make([]float64, audio.RipBands)
	for t := at - 1; t <= at+1 && t < len(rows); t++ {
		for b := range audio.RipBands {
			if up := float64(rows[t][audio.RipBands+b] - rows[t-1][audio.RipBands+b]); up > top {
				band, top = b, up
			}
		}
		adds(rose, rows[t][audio.RipBands:], rows[t-1][audio.RipBands:])
	}
	return band, rose
}

// adds keeps in rose each band's most energy added from the sharp plane before
// to now, taken back from the frames' log1p(100 e) and over sharpCounts, so a
// strike's shape is the same at any level and a low band no louder for its bins
func adds(rose []float64, now, before []float32) {
	counts := sharpCounts()
	for b := range rose {
		rose[b] = max(rose[b], (math.Expm1(float64(now[b]))-math.Expm1(float64(before[b])))/100/counts[b])
	}
}

// sharpCounts is how many times the sharp plane's bands, summed, count a steady
// cosine at each band's centre: about twenty at A0, where their triangles share
// bins, near one at the top, as the trainer's sharp_counts reads them
var sharpCounts = sync.OnceValue(func() []float64 {
	out := make([]float64, audio.RipBands)
	x := make([]float64, 4*audio.RipN*audio.AudioSampleRate/audio.RipRate)
	for b := range out {
		hz := audio.BandHz(float64(b))
		for i := range x {
			x[i] = math.Cos(2 * math.Pi * hz * float64(i) / audio.AudioSampleRate)
		}
		row := audio.BandFrames(x, audio.AudioSampleRate)[2*audio.RipN/audio.RipHop][audio.RipBands:]
		var sum float64
		for _, v := range row {
			sum += math.Expm1(float64(v))
		}
		out[b] = sum / math.Expm1(float64(row[b]))
	}
	return out
})

// risePlace is the place of places whose bands take the most energy a strike
// adds, the lowest of equals: where its sound is new, which one band's peak,
// moved a median of 11 bands by a piece's takes and velocities, says less
// steadily
func risePlace(rose []float64, places int) int {
	sum := make([]float64, places)
	for b, up := range rose {
		sum[b*places/audio.RipBands] += max(up, 0)
	}
	place := 0
	for p, v := range sum {
		if v > sum[place] {
			place = p
		}
	}
	return place
}

// pieceAlone is how far in frames from every other strike a recording's strike
// lies for its rise to be its piece's alone: a rise spreads two frames either
// way under the sharp plane's window and is read a frame either way, so a
// strike four frames from another reads none of the other's
const pieceAlone = 4

// pieceRises is a recording's pieces, a key each in order, each strike's piece,
// and each piece's energy added to each band (adds), summed over its strikes
// alone frames or more from every other, else over those beside only pieces
// read so, outside those pieces' places; nil for a piece of neither. frames
// and keys are the strikes', rows the band frames.
func pieceRises(rows iter.Seq[[]float32], frames, keys []int, alone, places int) (order, piece []int, rose [][]float64) {
	order = slices.Compact(slices.Sorted(slices.Values(keys)))
	piece = make([]int, len(keys))
	for i, k := range keys {
		piece[i], _ = slices.BinarySearch(order, k)
	}
	by := make([]int, len(frames)) // the strikes in the order of their frames
	for i := range by {
		by[i] = i
	}
	slices.SortStableFunc(by, func(i, j int) int { return cmp.Compare(frames[i], frames[j]) })
	with := func(i int) []int { // the other strikes nearer than alone
		var out []int
		k, _ := slices.BinarySearchFunc(by, frames[i]-alone+1, func(j, f int) int { return cmp.Compare(frames[j], f) })
		for ; k < len(by) && frames[by[k]] < frames[i]+alone; k++ {
			if by[k] != i {
				out = append(out, by[k])
			}
		}
		return out
	}
	sharp := map[int][]float32{} // the sharp plane of every frame a strike's rise reads
	for _, f := range frames {
		for t := max(f-2, 0); t <= f+1; t++ {
			sharp[t] = nil
		}
	}
	t := 0
	for row := range rows {
		if _, ok := sharp[t]; ok {
			sharp[t] = slices.Clone(row[audio.RipBands:])
		}
		t++
	}
	added := func(f int) []float64 { // what strike's frame f adds, nil where unread
		up, read := make([]float64, audio.RipBands), false
		for u := max(f-1, 1); u <= f+1; u++ {
			if now, before := sharp[u], sharp[u-1]; now != nil && before != nil {
				adds(up, now, before)
				read = true
			}
		}
		if !read {
			return nil
		}
		return up
	}
	sum := func(into [][]float64, p int, up []float64) {
		if into[p] == nil {
			into[p] = make([]float64, audio.RipBands)
		}
		for b, v := range up {
			into[p][b] += v
		}
	}
	rose = make([][]float64, len(order))
	for i, f := range frames {
		if up := added(f); up != nil && len(with(i)) == 0 {
			sum(rose, piece[i], up)
		}
	}
	read := make([]int, len(order))
	for p := range read {
		read[p] = -1
		if rose[p] != nil {
			read[p] = risePlace(rose[p], places)
		}
	}
	beside := make([][]float64, len(order))
	for i, f := range frames {
		up, others := added(f), with(i)
		if read[piece[i]] >= 0 || up == nil {
			continue
		}
		taken := map[int]bool{}
		for _, j := range others {
			if piece[j] != piece[i] && read[piece[j]] < 0 {
				taken = nil
				break
			}
			taken[read[piece[j]]] = piece[j] != piece[i]
		}
		if taken == nil {
			continue
		}
		for b := range up {
			if taken[b*places/audio.RipBands] {
				up[b] = 0
			}
		}
		sum(beside, piece[i], up)
	}
	for p := range rose {
		if rose[p] == nil {
			rose[p] = beside[p]
		}
	}
	return order, piece, rose
}

// recordedKit is a recording's kit by piece (pieceRises), each piece at the
// place where its strikes' rise is most (risePlace) and that rise's highest
// band, -1 for both where none of its strikes is read, and each strike's piece
func recordedKit(rows iter.Seq[[]float32], frames, keys []int) ([]corpusPiece, []int) {
	order, piece, rose := pieceRises(rows, frames, keys, pieceAlone, kitPlaces)
	kit := make([]corpusPiece, len(order))
	for p, k := range order {
		kit[p] = corpusPiece{Key: new(k), Band: -1, Place: -1}
		if rose[p] == nil {
			continue
		}
		band := 0
		for b, v := range rose[p] {
			if v > rose[p][band] {
				band = b
			}
		}
		kit[p].Band, kit[p].Place = band, risePlace(rose[p], kitPlaces)
	}
	return kit, piece
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

// pieceKit gives a recording's strikes, each event naming its key until then,
// the pieces of the kit its keys make (recordedKit), and its label that kit
func pieceKit(lab *corpusLoop, rows iter.Seq[[]float32]) {
	var frames, keys, at []int
	for i, e := range lab.Events {
		if e.Piece != nil {
			frames, keys, at = append(frames, e.Frame), append(keys, *e.Piece), append(at, i)
		}
	}
	if len(at) == 0 {
		return
	}
	kit, piece := recordedKit(rows, frames, keys)
	lab.Kit = kit
	for j, i := range at {
		lab.Events[i].Piece = new(piece[j])
	}
}
