package stapes

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/lixenwraith/audio"
)

// kitTake is a recording of the engine's drums alone on a kit drawn for it: its
// rise, and where its drums are struck in it, in seconds
type kitTake struct {
	flux, hits []float64
}

// drawKit binds each drum to the bank's own under an eq of five bands 10 dB
// either way, at a pitch half an octave and a length an octave either way, two
// strokes a velocity layer to keep a draw's cost down
func drawKit(t *testing.T, s *Session, rng *rand.Rand) {
	t.Helper()
	kit := map[string]string{}
	for _, d := range []string{"kick", "snare", "hihat"} {
		c := audio.SoundDefByName(d).Clone()
		c.Name, c.Variants = "lag_"+d, 2
		eq := audio.Proc{Kind: "eq"}
		for _, hz := range []float64{80, 250, 800, 2500, 8000} {
			eq.Bands = append(eq.Bands, audio.Band{Freq: hz, DB: 20*rng.Float64() - 10})
		}
		c.Chain = append(c.Chain, eq)
		if _, err := s.eng.DefineSound(c); err != nil {
			t.Fatal(err)
		}
		if err := s.eng.SetSoundShape(c.Name, audio.SoundShape{Pitch: math.Pow(2, rng.Float64()-0.5), Length: math.Pow(2, 2*rng.Float64()-1)}); err != nil {
			t.Fatal(err)
		}
		kit[d] = c.Name
	}
	if err := s.eng.SetKit(kit); err != nil {
		t.Fatal(err)
	}
}

// drawDrums is recording i of a draw at seed: four bars of the arranger's drums,
// played and drifting as its loops are, on a kit drawn for it, as samples and
// its hits where they sound
func drawDrums(t *testing.T, s *Session, seed uint64, i int) ([]float64, []audio.RipHit) {
	t.Helper()
	stream := func(salt uint64) *rand.Rand { return rand.New(rand.NewPCG(seed^salt, uint64(i)+1)) }
	drawKit(t, s, stream(0x6b6974))
	a := newArranger(stream(0x5a5a5a5a), stream(0x3e7e3e7e), 4, tuneTempos[i%len(tuneTempos)], true, false)
	for _, d := range []audio.InstrumentType{audio.InstrKick, audio.InstrSnare, audio.InstrHihat} {
		a.parts = append(a.parts, &part{kind: &partKinds[0], instr: d})
	}
	writeDrums(a, a.parts)
	a.perform()
	for _, p := range a.parts {
		p.notes = slices.DeleteFunc(p.notes, func(n note) bool { return n.at < a.from || n.at >= a.to })
	}
	m, err := a.music(false)
	if err != nil {
		t.Fatal(err)
	}
	buf, err := s.eng.RenderBars(m.timeline, m.heads, m.root, m.scale, m.prog, int64(i+1), nil)
	if err != nil {
		t.Fatal(err)
	}
	var hits []audio.RipHit
	for _, e := range audio.ExpectedEvents(m.timeline, m.heads, m.root, m.scale, m.prog) {
		if e.At >= m.from && e.At < m.to {
			hits = append(hits, audio.RipHit{Start: float64(e.At-m.from) / audio.AudioSampleRate, Drum: e.Instr, Vel: e.Vel})
		}
	}
	return slices.Clip(buf[m.from:m.to]), hits
}

// hitTimes is where hits start
func hitTimes(hits []audio.RipHit) []float64 {
	at := make([]float64, len(hits))
	for i, h := range hits {
		at[i] = h.Start
	}
	return at
}

// drawTakes is a draw's n recordings at seed, each with no room and then
// through degrade
func drawTakes(t *testing.T, s *Session, seed uint64, n int) []kitTake {
	t.Helper()
	var out []kitTake
	for i := range n {
		x, hits := drawDrums(t, s, seed, i)
		for _, room := range []*rand.Rand{nil, roomOf(true, seed, uint64(i))} {
			y := x
			if room != nil {
				y = degrade(slices.Clone(x), room)
			}
			out = append(out, kitTake{rise(slices.Values(audio.BandFrames(y, audio.AudioSampleRate))), strikeTimes(hitTimes(hits))})
		}
	}
	return out
}

// lagCase is a recording's hits as a set's file might hold them, and what was
// done to them: moved by laid seconds, steady or none, drifting over the
// recording, another recording's, or its own over no sound
type lagCase struct {
	kind       string
	flux, hits []float64
	laid       float64
}

// lagCases are a draw's recordings under each kind, the lags laid from a
// stream of the draw's seed. A steady one lies two frames short of its
// recording's reach, a share of its usual gap between strikes, either way, or
// evenly between: a frame for a kit's own rise, one for its peak to be seen
// whole. A drift moves three to six frames over the recording.
func lagCases(takes []kitTake, seed uint64, share float64) []lagCase {
	rng := rand.New(rand.NewPCG(seed, 0x6c6167))
	var out []lagCase
	moved := func(hits []float64, by func(float64) float64) []float64 {
		var to []float64
		for _, h := range hits {
			if v := h - by(h); v >= 0 {
				to = append(to, v)
			}
		}
		return to
	}
	for i, k := range takes {
		edge := max(reachOf(k.hits, len(k.flux), share)-2*audio.RipFrameSec, 0)
		total := float64(len(k.flux)) * audio.RipFrameSec
		drift := (3 + 3*rng.Float64()) * audio.RipFrameSec * float64(2*rng.IntN(2)-1)
		out = append(out, lagCase{"none", k.flux, k.hits, 0})
		for _, steady := range []float64{-edge, edge * (2*rng.Float64() - 1), edge} {
			out = append(out, lagCase{"steady", k.flux, moved(k.hits, func(float64) float64 { return steady }), steady})
		}
		out = append(out,
			lagCase{"drifting", k.flux, moved(k.hits, func(h float64) float64 { return drift * (h/total - 0.5) }), 0},
			lagCase{"another's", k.flux, takes[(i+2)%len(takes)].hits, 0},
			lagCase{"silent", make([]float64, len(k.flux)), k.hits, 0})
	}
	return out
}

// lagFit is what hits moved by none or a steady lag fit at a share of the usual
// gap: the median lag read less the lag laid, the largest miss about it to the
// nearest step and one more, halfway from the most another lag meets to a tie,
// and from the least share of hits drawing an onset to the most of another's or
// of silence. It holds where each is within a frame, its halves in the bound.
type lagFit struct {
	holds                     bool
	rise, bound, clear, drawn float64
	said                      string
}

func fitLag(takes []kitTake, seed uint64, share float64) lagFit {
	var miss, others, drawn, strays, drifts []float64
	for _, c := range lagCases(takes, seed, share) {
		r := lagOf(c.flux, c.hits, reachOf(c.hits, len(c.flux), share))
		switch c.kind {
		case "none", "steady":
			miss, others, drawn, drifts = append(miss, r.lag-c.laid), append(others, r.other), append(drawn, r.drawn), append(drifts, r.drift)
		case "another's", "silent":
			strays = append(strays, r.drawn)
		}
	}
	f := lagFit{holds: true, rise: math.Round(median(slices.Clone(miss))/lagStep) * lagStep}
	var far float64
	for _, m := range miss {
		far = max(far, math.Abs(m-f.rise))
	}
	f.bound = (math.Round(far/lagStep) + 1) * lagStep
	f.holds = far < audio.RipFrameSec && slices.Max(drifts) <= f.bound && slices.Min(drawn) > slices.Max(strays)
	f.clear = math.Round(50*(slices.Max(others)+1)) / 100
	f.drawn = math.Round(50*(slices.Min(drawn)+slices.Max(strays))) / 100
	f.said = fmt.Sprintf("%d recordings of hits moved by none or a steady lag: the rise %.1f ms after a hit, the furthest %.1f ms from it; another lag meets %.3f at most; %.3f of the hits draw an onset at least, of another's or of silence %.3f at most; the halves lie %.1f ms apart at most",
		len(miss), 1000*f.rise, 1000*far, slices.Max(others), slices.Min(drawn), slices.Max(strays), 1000*slices.Max(drifts))
	return f
}

// TestARecordedKitsHitsAreMovedOntoTheirSound: the engine's drums alone on drawn
// kits, with no room and through degrade, their hits moved by a lag laid.
// lagReach is the widest share of the usual gap, doubling from a quarter, at
// which lagFit holds on one draw, the four others what it fits there; on a second
// each lag is found within lagBound, a drifting, a silent and another's left out.
func TestARecordedKitsHitsAreMovedOntoTheirSound(t *testing.T) {
	var out bytes.Buffer
	s := newTestSession(t, &out)
	defer teardown(t, s)
	takes := drawTakes(t, s, 21, 12)
	fit := lagFit{}
	reach := 0.0
	for share := 0.25; share <= 2; share *= 2 {
		f := fitLag(takes, 21, share)
		t.Logf("at %.2f of the usual gap, %s", share, f.said)
		if !f.holds {
			break
		}
		fit, reach = f, share
	}
	if reach != lagReach || math.Abs(fit.rise-lagRise) > 1e-9 || math.Abs(fit.bound-lagBound) > 1e-9 || fit.clear != lagClear || fit.drawn != lagDrawn {
		t.Errorf("the fit holds to %.2f of the usual gap, with a rise of %.0f steps, a bound of %.0f, %.2f for a lag clear and %.2f for hits drawn; lag.go holds %.2f, %.0f, %.0f, %.2f and %.2f",
			reach, fit.rise/lagStep, fit.bound/lagStep, fit.clear, fit.drawn, lagReach, lagRise/lagStep, lagBound/lagStep, lagClear, lagDrawn)
	}
	var rows strings.Builder
	refused := map[string]map[string]int{}
	for i, c := range lagCases(drawTakes(t, s, 22, 6), 22, lagReach) {
		lag, why := placeHits(c.flux, c.hits)
		switch c.kind {
		case "none", "steady":
			fmt.Fprintf(&rows, "\n  %2d %-6s laid %+7.1f ms, found %+7.1f, %+5.1f off", i/7, c.kind, 1000*c.laid, 1000*lag, 1000*(lag-c.laid))
			if why != "" || math.Abs(lag-c.laid) > lagBound+1e-9 {
				t.Errorf("recording %d, a lag of %.1f ms laid: found %.1f ms, left out for %q; want it within %.1f ms", i/7, 1000*c.laid, 1000*lag, why, 1000*lagBound)
			}
		default:
			if refused[c.kind] == nil {
				refused[c.kind] = map[string]int{}
			}
			refused[c.kind][why]++
			if why == "" || c.kind == "silent" && why != lagWhy[0] {
				t.Errorf("recording %d, %s: placed at %.1f ms or left out for %q, want it left out", i/7, c.kind, 1000*lag, why)
			}
		}
	}
	t.Logf("the readback, the bound %.1f ms:%s\n  left out: %v", 1000*lagBound, rows.String(), refused)
}

// TestACorpusFromPutsARecordedKitsHitsOnItsSound: a recording whose file holds
// hits alone is labelled with them and its beats moved by the lag its label
// states, each hit within a frame of where its drum sounds; a silent one keeps
// its label, says why and joins skip.txt, the run going on and its last line
// counting both; labels reads the same from the frames on disk.
func TestACorpusFromPutsARecordedKitsHitsOnItsSound(t *testing.T) {
	var out bytes.Buffer
	s := newTestSession(t, &out)
	defer teardown(t, s)
	dir := t.TempDir()
	at := func(name string) string { return filepath.Join(dir, name) }
	frame := func(sec float64) int { return int(math.Floor(sec/audio.RipFrameSec + 0.5)) }
	truth, laid := map[string][]int{}, map[string]float64{}
	csv := "audio_filename,midi_filename,beats,velocity\n"
	for i, name := range []string{"on", "late", "silent"} {
		x, hits := drawDrums(t, s, 23, i)
		if name == "late" { // its sound a quarter of its reach after its file's hits
			laid[name] = reachOf(strikeTimes(hitTimes(hits)), loopFrames(len(x)), lagReach) / 2
		}
		var moved []audio.RipHit
		for _, h := range hits {
			truth[name] = append(truth[name], frame(h.Start))
			if h.Start -= laid[name]; h.Start >= 0 {
				moved = append(moved, h)
			}
		}
		var mid bytes.Buffer
		if err := audio.WriteMIDI(&mid, nil, moved, tuneTempos[i], audio.Meter{}); err != nil {
			t.Fatal(err)
		}
		writeFile(t, at(name+".mid"), mid.String())
		if name == "silent" {
			clear(x)
		}
		writeWAV(t, at(name+".wav"), x)
		csv += fmt.Sprintf("%s.wav,%s.mid,%s.mid,none\n", name, name, name)
	}
	writeFile(t, at("list.csv"), csv)
	corpus := at("corpus")
	// each recording's label, as read and as written, the silent one's loop, and skip.txt
	read := func() (labs map[string]corpusLoop, raw map[string][]byte, silent, skip string) {
		labs, raw = map[string]corpusLoop{}, map[string][]byte{}
		for i := range 3 {
			name := fmt.Sprintf("loop_%03d", i)
			data, _ := os.ReadFile(filepath.Join(corpus, name+".json"))
			var l corpusLoop
			if err := json.Unmarshal(data, &l); err != nil {
				t.Fatal(err)
			}
			if labs[l.Source], raw[l.Source] = l, data; l.Source == "silent.wav" {
				silent = name
			}
		}
		held, _ := os.ReadFile(filepath.Join(corpus, "skip.txt"))
		return labs, raw, silent, string(held)
	}
	runOrFatal(t, s, &out, fmt.Sprintf("corpus %s from %s\n", corpus, at("list.csv")))
	labs, raw, silent, skip := read()
	for _, name := range []string{"on", "late"} {
		l := labs[name+".wav"]
		if l.Lag == nil || l.Skip != "" || math.Abs(*l.Lag-laid[name]) > lagBound {
			t.Errorf("%s: a lag of %v and left out for %q, want %.1f ms within %.1f", name, l.Lag, l.Skip, 1000*laid[name], 1000*lagBound)
			continue
		}
		for _, e := range l.Events {
			if near := slices.MinFunc(truth[name], func(a, b int) int { return abs(a-e.Frame) - abs(b-e.Frame) }); abs(near-e.Frame) > 1 {
				t.Errorf("%s: a hit labelled at frame %d, its drum sounding nearest at %d", name, e.Frame, near)
			}
		}
		beats, _, err := readBeats(at(name + ".mid"))
		if err != nil {
			t.Fatal(err)
		}
		for k, f := range l.Beats {
			if want := frame(beats[k] + *l.Lag); len(l.Beats) > len(beats) || abs(f-want) > 1 {
				t.Errorf("%s: beat %d of %d labelled at frame %d, its file's moved by the lag at %d", name, k, len(l.Beats), f, want)
				break
			}
		}
		if len(l.Events) < len(truth[name])-2 || len(l.Beats) < 8 {
			t.Errorf("%s: %d hit(s) and %d beat(s) labelled of %d hit(s) played over four bars", name, len(l.Events), len(l.Beats), len(truth[name]))
		}
	}
	if l := labs["silent.wav"]; l.Lag != nil || l.Skip != lagWhy[0] || len(l.Events) == 0 || skip != silent+"\n" {
		t.Errorf("the silent recording: a lag of %v, left out for %q with %d event(s), skip.txt %q; want no lag, %q, its hits and %s alone", l.Lag, l.Skip, len(l.Events), skip, lagWhy[0], silent)
	}
	for _, said := range []string{"silent.wav: left out for " + lagWhy[0], "the hits of 2 recording(s) put on their sound and 1 left out in skip.txt: 1 for " + lagWhy[0]} {
		if !strings.Contains(out.String(), said) {
			t.Errorf("the run says %q, want %q in it", out.String(), said)
		}
	}
	for _, name := range []string{"on", "late", "silent"} {
		os.Remove(at(name + ".wav"))
	}
	runOrFatal(t, s, &out, fmt.Sprintf("corpus %s from %s labels\n", corpus, at("list.csv")))
	_, again, _, skipped := read()
	for source, data := range raw {
		if !bytes.Equal(again[source], data) {
			t.Errorf("labels wrote %s as\n%s\nwant the export's\n%s", source, again[source], data)
		}
	}
	if skipped != skip {
		t.Errorf("labels left skip.txt as %q, want %q", skipped, skip)
	}
}
