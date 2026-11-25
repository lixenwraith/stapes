package stapes

import (
	"bytes"
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
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
// hit reads the same band laid on a later frame, so strikeAt's silence before
// it is enough
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
		if at, later := riseBand(x, strikeAt), riseBand(x, 2*strikeAt); at != later {
			t.Errorf("piece %d reads band %d laid on frame %d and %d on frame %d", i, at, strikeAt, later, 2*strikeAt)
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
