package stapes

import (
	"bytes"
	"cmp"
	"encoding/csv"
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
// draw, its strikes, the rise of each piece's hit alone as its loop strikes it
// (placed) and its band frames
type placeTake struct {
	draw    int
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
			out = append(out, placeTake{i, strikes, alone, audio.BandFrames(y, audio.AudioSampleRate)})
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

// readPiece is a piece of a drawn kit as its recording's own read gives it: how
// sure the read is, whether its place is its own, its strikes, and whether its
// draw fits the bound or reads it back
type readPiece struct {
	sure         float64
	right, fit   bool
	strikes, all int
}

// readPieces is each piece of a draw's recordings as its own read gives it at
// kitPlaces (pieceRises), the even draws fitting, and the strikes of none read
func readPieces(takes []placeTake) (out []readPiece, unread int) {
	for _, k := range takes {
		frames, keys := make([]int, len(k.strikes)), make([]int, len(k.strikes))
		for i, st := range k.strikes {
			frames[i], keys[i] = st.frame, st.piece
		}
		order, piece, read := pieceRises(slices.Values(k.rows), frames, keys, pieceAlone, kitPlaces)
		count := make([]int, len(order))
		for _, p := range piece {
			count[p]++
		}
		for p, n := range order {
			if read[p].rose == nil {
				unread += count[p]
				continue
			}
			right := risePlace(read[p].rose, kitPlaces) == risePlace(k.alone[n], kitPlaces)
			out = append(out, readPiece{read[p].sure(kitPlaces), right, k.draw%2 == 0, count[p], len(k.strikes)})
		}
	}
	return out, unread
}

// kept is the share of the strikes of pieces whose sureness reaches bound, of
// those that fit or read back, and the share right of those kept
func kept(ps []readPiece, fit bool, bound float64) (share, right float64) {
	var all, in, ok int
	for _, p := range ps {
		if p.fit != fit {
			continue
		}
		if all += p.strikes; p.sure >= bound {
			in, ok = in+p.strikes, ok+p.strikes*b2i(p.right)
		}
	}
	return float64(in) / float64(max(all, 1)), float64(ok) / float64(max(in, 1))
}

// TestTheBandsArePlacedInTheRunsThatKeepTheMostStrikesApart: drawn kits' drums
// alone, no room and through degrade, PLACEDRAWS draws, 48 for the fits. A strike
// is unheard where another within a frame shares its piece's place or its sound
// rises most elsewhere: kitPlaces leaves the fewest. pieceSure is the least
// sureness the even draws' pieces kept are right at nine times in ten, which
// keeps the odd draws' righter than no bound does
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
	measured, counts := make([][4]int, 4), []int{2, 3, 4, 6} // read as a recording's are
	for _, k := range takes {
		frames, keys := make([]int, len(k.strikes)), make([]int, len(k.strikes))
		for i, st := range k.strikes {
			frames[i], keys[i] = st.frame, st.piece
		}
		measuredTable(measured, counts, frames, keys, k.rows)
	}
	rows.WriteString("\n  read as a recording's:" + tableLines(measured, counts))
	pieces, unread := readPieces(takes)
	bound := 1.0 // the least sureness whose pieces kept on the fit's draws are right nine times in ten
	sures := slices.Sorted(maps.Keys(maps.Collect(func(yield func(float64, bool) bool) {
		for _, p := range pieces {
			if !yield(p.sure, true) {
				return
			}
		}
	})))
	for _, b := range sures {
		if share, right := kept(pieces, true, b); share > 0 && right >= 0.9 {
			bound = b
			break
		}
	}
	strikes := unread
	for _, p := range pieces {
		strikes += p.strikes
	}
	_, readRight := kept(pieces, false, 0)
	keptBack, rightBack := kept(pieces, false, pieceSure)
	keptFit, rightFit := kept(pieces, true, pieceSure)
	var curve strings.Builder
	for _, b := range []float64{0.5, 0.6, 0.7, 0.8} {
		k, r := kept(pieces, false, b)
		fmt.Fprintf(&curve, " %.1f %.3f, %.3f;", b, k, r)
	}
	t.Logf("%d recordings, %d strikes, %.3f of them of pieces read:%s\n  at %d places the least sureness right nine times in ten on the even draws is %.4f; at %.2f there %.3f of the strikes read are kept, %.3f of those right, and on the odd draws %.3f kept, %.3f right, where with no bound %.3f are right\n  the odd draws' strikes read kept and right by bound:%s",
		len(takes), strikes, 1-float64(unread)/float64(strikes), rows.String(), kitPlaces, bound, pieceSure, keptFit, rightFit, keptBack, rightBack, readRight, curve.String())
	if draws >= 48 && (places != kitPlaces || math.Floor(100*bound)/100 != pieceSure) || rightBack <= readRight {
		t.Errorf("over %d draws the fewest strikes are unheard at %d places, kit.go's %d; the fit's bound %.4f, kit.go's %.2f, keeps the read-back pieces right %.3f of their strikes against %.3f with none",
			draws, places, kitPlaces, bound, pieceSure, rightBack, readRight)
	}
}

// TestARecordedKitsStrikesAreItsPiecesByTheirKeys: a recording of drawn drums
// whose file strikes each piece on a key of its own is labelled by piece, a key
// each in order at a place it is sure of or at none and why, each strike its
// key's piece and its line's drum where its key has one, no names but the
// keys; the manifest states the places and counts the pieces at none by why
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
	unplaced := map[string]int{}
	for _, pc := range l.Kit {
		placed := pc.Place >= 0 && pc.Place < kitPlaces && pc.Why == "" && pc.Sure >= pieceSure
		if pc.Key == nil || pc.Genes != nil || !placed && (pc.Place != -1 || !slices.Contains(placeWhy, pc.Why)) {
			t.Errorf("a recorded piece %+v", pc)
			continue
		}
		listed, unplaced[pc.Why] = append(listed, *pc.Key), unplaced[pc.Why]+b2i(!placed)
	}
	delete(unplaced, "")
	if len(unplaced) == 0 {
		unplaced = nil
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
	if man.Places != kitPlaces || !maps.Equal(man.Unplaced, unplaced) {
		t.Errorf("the manifest states %d places and %v at none, want %d and the label's %v", man.Places, man.Unplaced, kitPlaces, unplaced)
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

// measuredTable adds to c, at each count of places, a recording's strikes read
// that share a frame and a place their pieces' own reads give (pieceRises), and
// its lone strikes whose own rise is off the place its piece's other lone
// strikes give: [shared, read, off, lone] a count
func measuredTable(c [][4]int, counts, frames, keys []int, bands [][]float32) {
	lone := make([]bool, len(frames))
	for i, f := range frames {
		lone[i] = f >= 2 && f+1 < len(bands)
		for j, g := range frames {
			lone[i] = lone[i] && (j == i || abs(g-f) >= pieceAlone)
		}
	}
	for k, n := range counts {
		_, piece, read := pieceRises(slices.Values(bands), frames, keys, pieceAlone, n)
		for i, f := range frames {
			if read[piece[i]].rose == nil {
				continue
			}
			own, with := risePlace(read[piece[i]].rose, n), false
			for j, g := range frames {
				with = with || j != i && abs(g-f) <= 1 && read[piece[j]].rose != nil && risePlace(read[piece[j]].rose, n) == own
			}
			c[k][0], c[k][1] = c[k][0]+b2i(with), c[k][1]+1
			if !lone[i] {
				continue
			}
			var others [][]float64
			for j, g := range frames {
				if j != i && lone[j] && piece[j] == piece[i] {
					others = append(others, strikeAdded(bands, g))
				}
			}
			if len(others) > 0 {
				c[k][2], c[k][3] = c[k][2]+b2i(risePlace(strikeAdded(bands, f), n) != risePlace(sum(others), n)), c[k][3]+1
			}
		}
	}
}

// tableLines is measuredTable's counts as a log's lines
func tableLines(c [][4]int, counts []int) string {
	var b strings.Builder
	for k, n := range counts {
		fmt.Fprintf(&b, "\n  %d places: %.3f of %d strikes read share a frame and a place, %.3f of %d lone ones off their piece's place", n, float64(c[k][0])/float64(max(c[k][1], 1)), c[k][1], float64(c[k][2])/float64(max(c[k][3], 1)), c[k][3])
	}
	return b.String()
}

// TestARecordingsPiecesArePlacedAlikeInTakesOfOneKit: with PLACECORPUS naming
// corpora of recordings by piece and PLACELIST their lists, whose kit_name names
// a take's kit, logs the strikes kept at a place, by key how often takes of one
// kit place a piece alike, and measuredTable at 2, 3, 4 and 6 places; takes of
// one kit alike more often than the commonest place
func TestARecordingsPiecesArePlacedAlikeInTakesOfOneKit(t *testing.T) {
	dirs, lists := filepath.SplitList(os.Getenv("PLACECORPUS")), filepath.SplitList(os.Getenv("PLACELIST"))
	if len(dirs) == 0 || len(dirs) != len(lists) {
		t.Skip("PLACECORPUS and PLACELIST name no corpora of recordings and their lists, a list each")
	}
	kitOf := map[string]string{} // a take's kit by its corpus and source
	for k, list := range lists {
		f, err := os.Open(list)
		if err != nil {
			t.Fatal(err)
		}
		rows, err := csv.NewReader(f).ReadAll()
		f.Close()
		if err != nil || len(rows) < 2 {
			t.Fatalf("%s: %v", list, err)
		}
		col := func(name string) int { return slices.Index(rows[0], name) }
		for _, r := range rows[1:] {
			kitOf[dirs[k]+"\x00"+r[col("audio_filename")]] = r[col("kit_name")]
		}
	}
	counts := []int{2, 3, 4, 6}
	table := make([][4]int, len(counts))
	var all, keptStrikes, pieces, keptPieces int
	takes := map[[2]any]map[int]int{} // a kit's key: its takes by place
	for _, dir := range dirs {
		skip, _ := os.ReadFile(filepath.Join(dir, "skip.txt"))
		for i := 0; ; i++ {
			name := fmt.Sprintf("loop_%03d", i)
			data, err := os.ReadFile(filepath.Join(dir, name+".json"))
			if err != nil {
				break
			}
			var l corpusLoop
			if err := json.Unmarshal(data, &l); err != nil {
				t.Fatal(err)
			}
			if strings.Contains(string(skip), name) || len(l.Kit) == 0 {
				continue
			}
			var frames, keys []int
			for _, e := range l.Events {
				if e.Piece != nil {
					frames, keys = append(frames, e.Frame), append(keys, *l.Kit[*e.Piece].Key)
					all, keptStrikes = all+1, keptStrikes+b2i(l.Kit[*e.Piece].Place >= 0)
				}
			}
			for _, pc := range l.Kit {
				pieces, keptPieces = pieces+1, keptPieces+b2i(pc.Place >= 0)
				if pc.Place >= 0 {
					k := [2]any{kitOf[dir+"\x00"+l.Source], *pc.Key}
					if takes[k] == nil {
						takes[k] = map[int]int{}
					}
					takes[k][pc.Place]++
				}
			}
			var bands [][]float32
			for row := range bandsRows(filepath.Join(dir, name+".bands")) { // one buffer, read again each frame
				bands = append(bands, slices.Clone(row))
			}
			measuredTable(table, counts, frames, keys, bands)
		}
	}
	if all == 0 {
		t.Fatalf("%v hold no recording of strikes by piece", dirs)
	}
	var line strings.Builder
	byKey, alike, placed := map[int][2]int{}, 0, 0
	at := map[int]int{}
	for k, by := range takes {
		n, most := 0, 0
		for _, c := range by {
			n, most = n+c, max(most, c)
		}
		if n > 1 {
			key := k[1].(int)
			byKey[key] = [2]int{byKey[key][0] + most, byKey[key][1] + n}
			alike, placed = alike+most, placed+n
			for p, c := range by {
				at[p] += c
			}
		}
	}
	for _, k := range slices.Sorted(maps.Keys(byKey)) {
		fmt.Fprintf(&line, "\n  key %d: %d of %d takes of its kits at the place most of them give", k, byKey[k][0], byKey[k][1])
	}
	commonest := slices.Max(append(slices.Collect(maps.Values(at)), 0))
	t.Logf("%d of %d pieces and %.3f of %d strikes kept at a place; takes of one kit alike %.3f, the commonest place %.3f:%s%s",
		keptPieces, pieces, float64(keptStrikes)/float64(all), all, float64(alike)/float64(max(placed, 1)), float64(commonest)/float64(max(placed, 1)), tableLines(table, counts), line.String())
	if alike <= commonest {
		t.Errorf("takes of one kit place a piece alike %d of %d times, no oftener than the commonest place's %d", alike, placed, commonest)
	}
}

// strikeAdded is the energy a strike on frame f of bands adds to each band
// over the frames riseBand reads (adds)
func strikeAdded(bands [][]float32, f int) []float64 {
	up := make([]float64, audio.RipBands)
	for u := f - 1; u <= f+1; u++ {
		adds(up, bands[u][audio.RipBands:], bands[u-1][audio.RipBands:])
	}
	return up
}
