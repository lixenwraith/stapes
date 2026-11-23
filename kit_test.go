package stapes

import (
	"bytes"
	"cmp"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/lixenwraith/audio"
)

// TestAKitsPiecesTakeRolesByTheOrderOfTheirBands: whatever its size, the
// lowest band's piece is the low pulse, the highest's the time and the
// middle's the backbeat, those under it the fill from the highest down and
// those over it the accents; every piece plays a role, of two the higher both,
// and equal bands keep the order they were drawn in
func TestAKitsPiecesTakeRolesByTheOrderOfTheirBands(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for size := kitFewest; size <= kitMost; size++ {
		for _, tied := range []bool{false, true} {
			k := &loopKit{}
			for n, b := range rng.Perm(size) {
				if tied {
					b = 0
				}
				k.pieces = append(k.pieces, &kitPiece{n: n, band: 10 * b})
			}
			k.cast()
			by := slices.Clone(k.pieces)
			slices.SortStableFunc(by, func(x, y *kitPiece) int { return x.band - y.band })
			roles := map[*kitPiece]int{k.low: 1, k.time: 1}
			roles[k.back]++
			for _, p := range append(slices.Clone(k.fills), k.accents...) {
				roles[p]++
			}
			mid := size / 2
			fills := slices.Clone(by[1:mid])
			slices.Reverse(fills)
			switch {
			case k.low != by[0] || k.time != by[size-1] || k.back != by[mid]:
				t.Errorf("%d pieces, tied %v: low %d, back %d, time %d of the order %v", size, tied, k.low.n, k.back.n, k.time.n, by)
			case !slices.Equal(k.fills, fills) || !slices.Equal(k.accents, by[min(mid+1, size-1):size-1]):
				t.Errorf("%d pieces, tied %v: fills %v, accents %v of the order %v", size, tied, k.fills, k.accents, by)
			case len(roles) != size:
				t.Errorf("%d pieces, tied %v: %d of them play a role", size, tied, len(roles))
			}
			for p, n := range roles {
				if n != 1 && (size != 2 || p != k.back) {
					t.Errorf("%d pieces: piece %d plays %d roles", size, p.n, n)
				}
			}
		}
	}
}

// TestAPiecesBandIsWhereItsStrikeRisesMost: a piece that is a tone alone, its
// body unbent with no noise, click or drive, reads the band of the pitch its
// take sounds within a band, higher for a higher tone; and every drawn piece's
// hit reads the same band and place laid on a later frame, so strikeAt's
// silence before it is enough
func TestAPiecesBandIsWhereItsStrikeRisesMost(t *testing.T) {
	gene := func(v, lo, hi float64) float64 { return math.Log(v/lo) / math.Log(hi/lo) }
	last := -1
	for _, hz := range []float64{220, 440, 880, 1760} {
		genes := make([]float64, audio.StruckGeneCount)
		genes[0], genes[1], genes[2], genes[3] = gene(hz, 30, 2800), 1, 0.5, 0.5
		d := audio.StruckFromGenes(genes)
		d.Name = kitSound(0)
		band, err := strikeBand(d)
		if err != nil {
			t.Fatal(err)
		}
		take := hz * (1 - d.PitchWalk/2) // take 0, its walk's lowest
		if want := 3 * (12*math.Log2(take/440) + 69 - audio.RipBandsLowest); math.Abs(float64(band)-want) > 1 || band <= last {
			t.Errorf("a tone of %.0f Hz reads band %d (%.0f Hz), its take's %.1f, after %d", hz, band, audio.BandHz(float64(band)), want, last)
		}
		last = band
	}
	rng := rand.New(rand.NewPCG(3, 4))
	for i := range 24 {
		genes := make([]float64, audio.StruckGeneCount)
		for g := range genes {
			genes[g] = rng.Float64()
		}
		d := audio.StruckFromGenes(genes)
		d.Name = kitSound(i % kitMost)
		x, err := audio.RenderHit(d, audio.SoundShape{}, 1, 0)
		if err != nil {
			t.Fatal(err)
		}
		band, rose := riseBand(x, strikeAt)
		later, there := riseBand(x, 2*strikeAt)
		if place := risePlace(rose, kitPlaces); band != later || place != risePlace(there, kitPlaces) {
			t.Errorf("piece %d reads band %d and place %d laid on frame %d, %d and %d on frame %d", i, band, place, strikeAt, later, risePlace(there, kitPlaces), 2*strikeAt)
		}
	}
}

// TestEveryPieceOfADrawnKitPlays: arranged loops in every meter, some starting
// part-way into a bar, draw kits of every size from two to ten, each piece
// numbered as drawn, its genes in [0, 1) and its level within the bank drums'
// norms, every drum line played on a piece, and every piece strikes in the loop
func TestEveryPieceOfADrawnKitPlays(t *testing.T) {
	levels, err := kitLevels()
	if err != nil {
		t.Fatal(err)
	}
	sizes := map[int]int{}
	for i := range 90 {
		stream := func(salt uint64) *rand.Rand { return rand.New(rand.NewPCG(7^salt, uint64(i)+1)) }
		a := newArranger(stream(0x5a5a5a5a), stream(0x3e7e3e7e), 4, tuneTempos[i%len(tuneTempos)], true, true)
		if err := a.compose(nil, nil, nil, stream(kitSalt)); err != nil {
			t.Fatal(err)
		}
		k := a.kit
		if k == nil {
			continue
		}
		sizes[len(k.pieces)]++
		strikes := map[*kitPiece][]note{}
		for _, p := range a.parts {
			if p.instr.IsDrum() && p.piece == nil {
				t.Fatalf("loop %d: a drum line of %d notes plays on no piece", i, len(p.notes))
			}
			if p.piece != nil {
				strikes[p.piece] = p.notes
			}
		}
		for n, pc := range k.pieces {
			d := pc.sound
			switch {
			case pc.n != n || d.Name != kitSound(n) || len(pc.genes) != audio.StruckGeneCount:
				t.Errorf("loop %d: piece %d numbered %d, named %q, of %d genes", i, n, pc.n, d.Name, len(pc.genes))
			case slices.Min(pc.genes) < 0 || slices.Max(pc.genes) >= 1 || d.Norm < levels[0] || d.Norm > levels[1]:
				t.Errorf("loop %d: piece %d genes %v, level %.3f outside %v", i, n, pc.genes, d.Norm, levels)
			case len(strikes[pc]) == 0:
				t.Errorf("loop %d: piece %d of %d strikes nothing", i, n, len(k.pieces))
			}
			for _, s := range strikes[pc] {
				if s.at < a.from || s.at >= a.to {
					t.Errorf("loop %d: piece %d strikes at %.2f outside the loop's %.0f to %.0f", i, n, s.at, a.from, a.to)
				}
			}
		}
	}
	for size := kitFewest; size <= kitMost; size++ {
		if sizes[size] == 0 {
			t.Errorf("no kit of %d pieces drawn: %v", size, sizes)
		}
	}
	t.Logf("kits by size: %v", sizes)
}

// TestAKitOfFourOrFewerSoundsTheSameInOnePassAsInSeveral: a loop whose kit has
// four pieces or fewer rendered in one pass, and in a pass a piece, is the
// same loop through the same room and mix, to rounding; a kit of more renders
// the same in passes of four and of one
func TestAKitOfFourOrFewerSoundsTheSameInOnePassAsInSeveral(t *testing.T) {
	var out bytes.Buffer
	s := newTestSession(t, &out)
	defer teardown(t, s)
	o := tuneOptions{loops: 3, bars: 1, seed: 14, gen: true, aug: true, kits: true} // a kit of four and one of five
	render := func(pass int) []tuneLoop {
		o.pass = pass
		loops, err := s.corpus(o)
		if err != nil {
			t.Fatal(err)
		}
		return loops
	}
	whole, apart := render(kitPass), render(1)
	sizes := map[bool]int{}
	for i, l := range whole {
		size := len(l.label.Kit)
		if size == 0 {
			continue
		}
		sizes[size <= kitPass]++
		var diff, peak float64
		for j, v := range l.audio {
			diff, peak = max(diff, math.Abs(v-apart[i].audio[j])), max(peak, math.Abs(v))
		}
		t.Logf("loop %d: %d pieces in %d pass(es) against %d: %.1e of its peak apart", i, size, (size+kitPass-1)/kitPass, size, diff/peak)
		if len(l.audio) != len(apart[i].audio) || diff > 1e-9*peak {
			t.Errorf("loop %d of %d pieces: %d samples against %d, %.1e of its peak apart", i, size, len(l.audio), len(apart[i].audio), diff/peak)
		}
	}
	if sizes[true] == 0 || sizes[false] == 0 {
		t.Errorf("kits of four or fewer, and of more: %v", sizes)
	}
}

// TestAKitsLabelSaysWhatWasStruck: a strike is labelled with its frame, its
// velocity and its piece's number in the loop's kit, as a strike of no line,
// and nothing of the drum it sounded on; the label lists each piece's genes once
// with its band, and the manifest names the genes; without the word a corpus
// writes none of it
func TestAKitsLabelSaysWhatWasStruck(t *testing.T) {
	var out bytes.Buffer
	s := newTestSession(t, &out)
	defer teardown(t, s)
	loops, err := s.corpus(tuneOptions{loops: 8, bars: 1, seed: 7, gen: true, kits: true, labels: true})
	if err != nil {
		t.Fatal(err)
	}
	strikes := 0
	for i, l := range loops {
		lab := l.label
		for _, pc := range lab.Kit {
			if len(pc.Genes) != audio.StruckGeneCount || pc.Band < 0 || pc.Band >= audio.RipBands {
				t.Errorf("loop %d: a piece of %d genes in band %d", i, len(pc.Genes), pc.Band)
			}
		}
		for _, e := range lab.Events {
			switch {
			case e.Part == "drums" && e.Piece == nil:
				t.Errorf("loop %d: a drum's hit %+v in a corpus of kits", i, e)
			case e.Piece == nil:
			case *e.Piece < 0 || *e.Piece >= len(lab.Kit) || e.Drum != -1 || e.MIDI != 0 || e.Instr != "" || e.Frame != e.End || e.Vel <= 0 || e.Vel > 1:
				t.Errorf("loop %d: a strike labelled %+v of a kit of %d", i, e, len(lab.Kit))
			default:
				strikes++
			}
		}
	}
	data, err := json.Marshal(loops[0].label)
	if err != nil {
		t.Fatal(err)
	}
	if strikes == 0 || !bytes.Contains(data, []byte(`"piece":0}`)) || !bytes.Contains(data, []byte(`"kit":[{"genes":[`)) {
		t.Errorf("%d strike(s), the first loop's label %.300s", strikes, data)
	}
	for _, kits := range []bool{true, false} {
		dir := t.TempDir() // seed 14's first loop plays no drums: its manifest alone is read
		if err := s.exportCorpus(dir, tuneOptions{loops: 1, bars: 1, seed: 14, gen: true, kits: kits, nowav: true}); err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(filepath.Join(dir, "corpus.json"))
		if err != nil {
			t.Fatal(err)
		}
		var man corpusManifest
		if err := json.Unmarshal(data, &man); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(man.StruckGenes, audio.StruckGeneNames()[:len(man.StruckGenes)]) || (len(man.StruckGenes) == audio.StruckGeneCount) != kits {
			t.Errorf("kits %v: the manifest's struck genes %v", kits, man.StruckGenes)
		}
		if strings.Contains(out.String(), "struck sounds") != kits {
			t.Errorf("kits %v: %s", kits, out.String())
		}
		out.Reset()
	}
}

// kitStrike is a strike of a drawn kit as its recording holds it: its frame,
// its piece's number and its velocity
type kitStrike struct {
	frame, piece int
	vel          float64
}

// kitDrums is draw i at seed: bars of the arranger's drums alone on a kit drawn
// for them, struck, played and dealt as a corpus loop's, as its samples, its
// strikes and its kit, each piece two strokes a velocity layer to keep a draw's
// cost down; nil where the drums strike nothing
func kitDrums(t *testing.T, s *Session, seed uint64, i, bars int) ([]float64, []kitStrike, *loopKit) {
	t.Helper()
	stream := func(salt uint64) *rand.Rand { return rand.New(rand.NewPCG(seed^salt, uint64(i)+1)) }
	a := newArranger(stream(0x5a5a5a5a), stream(0x3e7e3e7e), bars, tuneTempos[i%len(tuneTempos)], true, false)
	krng := stream(kitSalt)
	k, err := newKit(krng)
	if err != nil {
		t.Fatal(err)
	}
	for _, pc := range k.pieces {
		pc.sound.Variants = 2
	}
	a.kit, a.krng = k, krng
	for _, d := range []audio.InstrumentType{audio.InstrKick, audio.InstrSnare, audio.InstrHihat} {
		a.parts = append(a.parts, &part{kind: &partKinds[0], instr: d})
	}
	writeDrums(a, a.parts)
	keep := func() {
		for _, p := range a.parts {
			p.notes = slices.DeleteFunc(p.notes, func(n note) bool { return n.at < a.from || n.at >= a.to })
			slices.SortStableFunc(p.notes, func(x, y note) int { return cmp.Compare(x.at, y.at) })
		}
	}
	keep()
	a.strike()
	a.perform()
	keep()
	if a.deal(); a.kit == nil {
		return nil, nil, nil
	}
	if err := a.kit.placed(a.parts); err != nil {
		t.Fatal(err)
	}
	m, err := a.music(false)
	if err != nil {
		t.Fatal(err)
	}
	x, err := s.renderKit(m, m.timeline, int64(i+1), kitPass)
	if err != nil {
		t.Fatal(err)
	}
	var strikes []kitStrike
	for _, e := range audio.ExpectedEvents(m.timeline, m.heads, m.root, m.scale, m.prog) {
		if e.At >= m.from && e.At < m.to {
			f := int(float64(e.At-m.from)/audio.AudioSampleRate/audio.RipFrameSec + 0.5)
			strikes = append(strikes, kitStrike{f, m.parts[e.Slot][e.Track].piece.n, e.Vel})
		}
	}
	return slices.Clip(x[m.from:m.to]), strikes, m.kit
}

// placeTake is a recording of drawn drums as the place's count reads it: its
// strikes, the rise of each piece's hit alone as its loop strikes it (placed)
// and its band frames
type placeTake struct {
	strikes []kitStrike
	alone   [][]float64
	rows    [][]float32
}

// placeTakes is a draw's n recordings of two bars at seed, each at a level drawn
// as a loop's, with no room and through degrade, as the lag's fit lays its takes
func placeTakes(t *testing.T, s *Session, seed uint64, n int) []placeTake {
	t.Helper()
	var out []placeTake
	for i := range n {
		x, strikes, k := kitDrums(t, s, seed, i, 2)
		if k == nil {
			continue
		}
		alone := make([][]float64, len(k.pieces))
		for p, pc := range k.pieces {
			alone[p] = pc.rose
		}
		level(x, rand.New(rand.NewPCG(seed^0x6c65766c, uint64(i)+1)))
		for _, room := range []*rand.Rand{nil, roomOf(true, seed, uint64(i))} {
			y := x
			if room != nil {
				y = degrade(slices.Clone(x), room)
			}
			out = append(out, placeTake{strikes, alone, audio.BandFrames(y, audio.AudioSampleRate)})
		}
	}
	return out
}

// placeCount is what a count of places leaves a draw's strikes: the share of
// them another strike within a frame shares its piece's place with, the share
// of those alone (pieceAlone) whose own rise is most at another place, and the
// share either leaves unheard at their place, the second standing for every
// strike's sound
type placeCount struct{ shared, elsewhere, unheard float64 }

// countPlaces counts a draw's strikes at places of even runs
func countPlaces(takes []placeTake, places int) placeCount {
	var all, shared, lone, elsewhere int
	for _, k := range takes {
		for i, st := range k.strikes {
			own := risePlace(k.alone[st.piece], places)
			with, near := false, false
			for j, o := range k.strikes {
				with = with || j != i && abs(o.frame-st.frame) <= 1 && risePlace(k.alone[o.piece], places) == own
				near = near || j != i && abs(o.frame-st.frame) < pieceAlone
			}
			all, shared = all+1, shared+b2i(with)
			if near || st.frame < 2 || st.frame+1 >= len(k.rows) {
				continue
			}
			up := make([]float64, audio.RipBands)
			for u := st.frame - 1; u <= st.frame+1; u++ {
				adds(up, k.rows[u][audio.RipBands:], k.rows[u-1][audio.RipBands:])
			}
			lone, elsewhere = lone+1, elsewhere+b2i(risePlace(up, places) != own)
		}
	}
	c := placeCount{shared: float64(shared) / float64(all), elsewhere: float64(elsewhere) / float64(max(lone, 1))}
	c.unheard = c.shared + (1-c.shared)*c.elsewhere
	return c
}

// TestTheBandsArePlacedInTheRunsThatKeepTheMostStrikesApart: drawn kits' drums
// alone, no room and through degrade. A strike is unheard where another within
// a frame shares its piece's place or its sound rises most elsewhere: over
// PLACEDRAWS draws, 48 for the fit, kitPlaces leaves the fewest; a recording's
// own read of a place (pieceRises) is right more often than the commonest place
func TestTheBandsArePlacedInTheRunsThatKeepTheMostStrikesApart(t *testing.T) {
	var out bytes.Buffer
	s := newTestSession(t, &out)
	defer teardown(t, s)
	draws := 12 // the flat curve wants 48 to decide: the suite reads the rest
	if n, err := strconv.Atoi(os.Getenv("PLACEDRAWS")); err == nil {
		draws = n
	}
	takes := placeTakes(t, s, 31, draws)
	var rows strings.Builder
	fit, places := placeCount{unheard: 2}, 0
	for n := 1; n <= 16; n++ {
		c := countPlaces(takes, n)
		fmt.Fprintf(&rows, "\n  %2d places: %.3f share a frame and a place, %.3f alone rise most elsewhere, %.3f unheard", n, c.shared, c.elsewhere, c.unheard)
		if c.unheard < fit.unheard {
			fit, places = c, n
		}
	}
	var all, read, right int
	at := map[int]int{}
	for _, k := range takes {
		frames, keys := make([]int, len(k.strikes)), make([]int, len(k.strikes))
		for i, st := range k.strikes {
			frames[i], keys[i] = st.frame, st.piece
		}
		order, piece, rose := pieceRises(slices.Values(k.rows), frames, keys, pieceAlone, kitPlaces)
		for i := range k.strikes {
			all++
			if p := piece[i]; rose[p] != nil {
				own := risePlace(k.alone[order[p]], kitPlaces)
				read, right, at[own] = read+1, right+b2i(risePlace(rose[p], kitPlaces) == own), at[own]+1
			}
		}
	}
	commonest := slices.Max(slices.Collect(maps.Values(at)))
	t.Logf("%d recordings, %d strikes:%s\n  at %d places, a recording's own read of a piece's place: %.3f of the strikes read, %.3f of those right, the commonest place %.3f",
		len(takes), all, rows.String(), kitPlaces, float64(read)/float64(all), float64(right)/float64(read), float64(commonest)/float64(read))
	if draws >= 48 && places != kitPlaces || right <= commonest {
		t.Errorf("over %d draws the fewest strikes are unheard at %d places, kit.go's %d; read from the recordings %d strikes of %d at their place, the commonest place holds %d", draws, places, kitPlaces, right, read, commonest)
	}
}

// TestARecordedKitsStrikesAreItsPiecesByTheirKeys: a recording of drawn drums
// whose file strikes each piece on a key of its own is labelled by piece, the
// kit a piece a key in the keys' order with its place, each strike its key's
// piece and its line's drum where its key has one, no names but the keys; the
// manifest states the places
func TestARecordedKitsStrikesAreItsPiecesByTheirKeys(t *testing.T) {
	var out bytes.Buffer
	s := newTestSession(t, &out)
	defer teardown(t, s)
	dir := t.TempDir()
	at := func(name string) string { return filepath.Join(dir, name) }
	keys := []int{36, 38, 42, 45, 47, 49, 51, 41, 43, 46} // a piece's: of a line, or of none
	var x []float64
	var strikes []kitStrike
	for i := 0; len(strikes) == 0 || len(slices.Compact(slices.Sorted(slices.Values(piecesOf(strikes))))) < 5; i++ { // some of no line
		if i == 12 {
			t.Fatal("no kit of five pieces or more in twelve draws at seed 41")
		}
		x, strikes, _ = kitDrums(t, s, 41, i, 4)
	}
	struck := map[int][]int{} // the keys each frame's strikes are struck on
	for _, st := range strikes {
		struck[st.frame] = append(struck[st.frame], keys[st.piece])
	}
	writeFile(t, at("a.mid"), string(drumTrack(strikes, keys)))
	writeWAV(t, at("a.wav"), x)
	writeFile(t, at("list.csv"), "audio_filename,midi_filename,velocity\na.wav,a.mid,none\n")
	runOrFatal(t, s, &out, fmt.Sprintf("corpus %s from %s\n", at("corpus"), at("list.csv")))
	var l corpusLoop
	var man corpusManifest
	for file, v := range map[string]any{"loop_000.json": &l, "corpus.json": &man} {
		data, err := os.ReadFile(filepath.Join(at("corpus"), file))
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(data, v); err != nil {
			t.Fatal(err)
		}
	}
	if l.Lag == nil {
		t.Fatalf("its hits left out for %q", l.Skip)
	}
	lag := *l.Lag
	var listed []int
	for _, pc := range l.Kit {
		if pc.Key == nil || pc.Genes != nil || pc.Place < -1 || pc.Place >= kitPlaces {
			t.Errorf("a recorded piece %+v", pc)
			continue
		}
		listed = append(listed, *pc.Key)
	}
	if !slices.IsSorted(listed) || len(slices.Compact(slices.Clone(listed))) != len(listed) {
		t.Errorf("the kit's keys %v: each once, in order", listed)
	}
	score, err := audio.ReadMIDI(bytes.NewReader(drumTrack(strikes, keys)))
	if err != nil {
		t.Fatal(err)
	}
	lines := map[int]int{} // each key's line as the file reads it, -1 none
	for _, h := range score.Hits {
		lines[h.Key] = int(h.Drum)
	}
	for _, e := range l.Events {
		if e.Piece == nil || *e.Piece < 0 || *e.Piece >= len(l.Kit) {
			t.Fatalf("a strike %+v of a kit of %d", e, len(l.Kit))
		}
		key := *l.Kit[*e.Piece].Key
		drum, ok := lines[key]
		if !ok {
			drum = -1
		}
		near := false
		for f := range struck { // the file's frame, moved by the lag that puts it on its sound
			near = near || slices.Contains(struck[f], key) && abs(int(math.Floor(float64(f)+lag/audio.RipFrameSec+0.5))-e.Frame) <= 1
		}
		if !near || e.Drum != drum || e.Instr != "" {
			t.Errorf("a strike at frame %d of key %d labelled drum %d, instr %q: its file strikes %v there", e.Frame, key, e.Drum, e.Instr, struck[e.Frame])
		}
	}
	if man.Places != kitPlaces {
		t.Errorf("the manifest states %d places, want %d", man.Places, kitPlaces)
	}

}

// piecesOf is the pieces strikes strike, a strike each
func piecesOf(strikes []kitStrike) []int {
	out := make([]int, len(strikes))
	for i, st := range strikes {
		out[i] = st.piece
	}
	return out
}

// drumTrack is a MIDI file of one track on the drums' channel at 120 a minute,
// each strike its piece's key at its frame's seconds, a sixteenth long
func drumTrack(strikes []kitStrike, keys []int) []byte {
	const division = 480 // ticks a quarter; at 120 a minute a second is 960
	type at struct {
		tick int
		ev   []byte
	}
	var evs []at
	for _, st := range strikes {
		on := int(math.Round(float64(st.frame) * audio.RipFrameSec * 2 * division))
		evs = append(evs, at{on, []byte{0x99, byte(keys[st.piece]), 100}}, at{on + division/4, []byte{0x89, byte(keys[st.piece]), 0}})
	}
	slices.SortStableFunc(evs, func(x, y at) int { return cmp.Compare(x.tick, y.tick) })
	body := []byte{0, 0xff, 0x51, 3, 0x07, 0xa1, 0x20} // a quarter of 500,000 us
	last := 0
	for _, e := range evs {
		d := e.tick - last
		vlq := []byte{byte(d & 0x7f)}
		for d >>= 7; d > 0; d >>= 7 {
			vlq = append([]byte{byte(d&0x7f | 0x80)}, vlq...)
		}
		body, last = append(append(body, vlq...), e.ev...), e.tick
	}
	body = append(body, 0, 0xff, 0x2f, 0)
	out := []byte("MThd\x00\x00\x00\x06\x00\x00\x00\x01")
	out = append(out, byte(division>>8), byte(division&0xff))
	out = append(out, "MTrk"...)
	out = append(out, byte(len(body)>>24), byte(len(body)>>16), byte(len(body)>>8), byte(len(body)))
	return append(out, body...)
}
