// enginelines <spec.json> <out dir>: lines of one engine voice rendered offline as a spec lays them (lines.py writes
// the spec and the truth). A line is a voice definition, a tempo, a tuning in cents off A440 and notes on the
// sixteenth-step grid: position, length, key, velocity, a bend in semitones and the share of the note before it
// moves. Each is written as <out dir>/<name>.raw.wav, 44.1 kHz mono float, as RenderLines gives it, with a bar of
// silence after its last bar for the last note's release.
package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/lixenwraith/audio"
)

type note struct {
	Pos, Dur, MIDI int
	Vel, Bend, At  float64
}

type line struct {
	Name   string
	BPM    int
	Tuning float64 // cents off A440 the line is rendered at
	Voice  audio.VoiceDef
	Notes  []note
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func main() {
	data, err := os.ReadFile(os.Args[1])
	check(err)
	var lines []line
	check(json.Unmarshal(data, &lines))
	const root, scale = audio.DefaultRootNote, audio.ScalePhrygian
	for _, l := range lines {
		bars := 0
		for _, n := range l.Notes {
			bars = max(bars, n.Pos/audio.StepsPerBar+1)
		}
		events := make([][]audio.Step, bars)
		for _, n := range l.Notes {
			deg, oct, acc := audio.PitchStep(root, scale, n.MIDI)
			if err := audio.StepPitchBounds(deg, oct, acc); err != nil || audio.StepPitch(root, scale, deg, oct, acc) != n.MIDI {
				check(fmt.Errorf("%s: key %d is not a step the engine plays (%v)", l.Name, n.MIDI, err))
			}
			b := n.Pos / audio.StepsPerBar
			events[b] = append(events[b], audio.Step{Pos: n.Pos % audio.StepsPerBar, Vel: n.Vel, Deg: deg, Oct: oct, Acc: acc, Dur: n.Dur, Bend: n.Bend, BendAt: n.At})
		}
		v := l.Voice
		v.Name = "lead"
		cfg := audio.DefaultAudioConfig()
		cfg.Enabled = true
		cfg.ForceBackend = audio.BackendNameNull
		cfg.BaseVoices = []*audio.VoiceDef{&v}
		cfg.BasePatterns = nil
		name := func(b int) string { return fmt.Sprintf("bar%d", b) }
		for b, ev := range events {
			if len(ev) > 0 {
				cfg.BasePatterns = append(cfg.BasePatterns, &audio.Pattern{Name: name(b), Steps: audio.StepsPerBar, Tracks: []audio.Track{{Instr: audio.InstrLead, Events: ev}}})
			}
		}
		ae, err := audio.NewAudioEngine(cfg)
		check(err)
		check(ae.Start())
		ae.SetTuning(l.Tuning)
		timeline := make([][audio.MusicSlots]audio.PatternID, bars+1)
		for b, ev := range events {
			if len(ev) > 0 {
				timeline[b][1] = audio.PatternIDByName(name(b))
			}
		}
		x, err := ae.RenderLines(timeline, l.BPM, audio.Meter{}, root, scale, nil, 1, []audio.InstrumentType{audio.InstrLead})
		ae.Stop()
		audio.ResetRegistries()
		check(err)
		peak := 0.0
		for _, s := range x {
			peak = max(peak, math.Abs(s))
		}
		check(writeWAV(filepath.Join(os.Args[2], l.Name+".raw.wav"), x))
		fmt.Printf("%s\t%d notes\t%d bars at %d bpm\t%d samples\tpeak %.4f\n", l.Name, len(l.Notes), bars, l.BPM, len(x), peak)
	}
}

// writeWAV writes mono float samples as a 32-bit float WAV at the engine's rate
func writeWAV(path string, x []float64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	n := uint32(4 * len(x))
	head := []any{[]byte("RIFF"), 36 + n, []byte("WAVE"), []byte("fmt "), uint32(16), uint16(3), uint16(1), uint32(audio.AudioSampleRate),
		uint32(4 * audio.AudioSampleRate), uint16(4), uint16(32), []byte("data"), n}
	for _, h := range head {
		if err := binary.Write(f, binary.LittleEndian, h); err != nil {
			return err
		}
	}
	buf := make([]byte, 4*len(x))
	for i, s := range x {
		binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(float32(s)))
	}
	_, err = f.Write(buf)
	return err
}
