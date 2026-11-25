package stapes

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/lixenwraith/audio"
)

// kindFrames is frames of n frames whose pitch head is sure of one pitch on
// the first sounding of them and whose family head gives share of every frame
// to the family named, the rest to the piano
func kindFrames(n, sounding int, family string, share float32) *audio.RipFrames {
	k, pitches := len(audio.RipFamilies), audio.RipPitches
	f := &audio.RipFrames{Hop: audio.RipFrameSec, Salience: make([]float32, n*pitches), Family: make([]float32, n*k)}
	at := slices.Index(audio.RipFamilies[:], family)
	for t := range n {
		if t < sounding {
			f.Salience[t*pitches+40] = 0.9
		}
		f.Family[t*k+at], f.Family[t*k] = share, f.Family[t*k]+1-share
	}
	return f
}

// TestAKindIsReadFromWhatTheModelsSay: a rip that holds hits is of drums,
// whatever is named; with none, singing or a guitar where the family head gives
// it half or more of the frames something sounds in, and no kind where it gives
// less, where nothing sounds or where no head says
func TestAKindIsReadFromWhatTheModelsSay(t *testing.T) {
	quiet := &audio.Rip{}
	struck := &audio.Rip{Meter: audio.Meter{Num: 4, Den: 4}, Beats: []float64{0, 0.5, 1, 1.5, 2}, Bars: []audio.RipBar{{}}}
	struck.Bars[0].Drums[audio.InstrKick] = []audio.Step{{Pos: 0, Vel: 1}}
	if len(struck.Hits()) != 1 {
		t.Fatalf("the struck rip holds %d hit(s), want 1", len(struck.Hits()))
	}
	for _, c := range []struct {
		name string
		r    *audio.Rip
		f    *audio.RipFrames
		want string
	}{
		{"singing", quiet, kindFrames(100, 60, "voice", 0.8), soundVoice},
		{"a guitar", quiet, kindFrames(100, 60, "guitar", 0.5), soundGuitar},
		{"a guitar under half", quiet, kindFrames(100, 60, "guitar", 0.4), ""},
		{"strings", quiet, kindFrames(100, 60, "strings", 1), ""},
		{"nothing sounding", quiet, kindFrames(100, 0, "voice", 1), ""},
		{"no frames", quiet, nil, ""},
		{"no family head", quiet, &audio.RipFrames{Salience: make([]float32, audio.RipPitches)}, ""},
		{"drums under singing", struck, kindFrames(100, 60, "voice", 1), soundDrums},
		{"no rip", nil, kindFrames(100, 60, "voice", 1), ""},
	} {
		if got := kindOf(c.r, c.f); got != c.want {
			t.Errorf("%s: read as %q, want %q", c.name, got, c.want)
		}
	}
}

// TestARootsKindsAreTheSetsItHolds: a root's rip.<kind>.toml is the set a
// recording read as that kind is ripped under; one that does not load is named
// and left out, and a root with none holds none
func TestARootsKindsAreTheSetsItHolds(t *testing.T) {
	root := t.TempDir()
	var said []string
	say := func(l string) { said = append(said, l) }
	if kinds := loadKinds(root, say); kinds != nil || said != nil {
		t.Errorf("an empty root: %v, %q", kinds, said)
	}
	writeFile(t, filepath.Join(root, "rip.voice.toml"), "[rip]\nrelease = 0.575\n")
	writeFile(t, filepath.Join(root, "rip.guitar.toml"), "[rip]\nstart = 0.5\n")
	kinds := loadKinds(root, say)
	want := audio.DefaultRipParams()
	want.Release = 0.575
	if len(kinds) != 1 || kinds[soundVoice] != want {
		t.Errorf("the root's kinds %+v, want singing's alone, the built-in thresholds with its release", kinds)
	}
	if len(said) != 2 || !strings.Contains(said[0], "rip.voice.toml, for a recording read as voice") || !strings.Contains(said[1], "rip.guitar.toml") || !strings.Contains(said[1], "the root's own used for guitar") {
		t.Errorf("said %q", said)
	}
	if _, err := os.Stat(kindFile(root, soundVoice)); err != nil {
		t.Errorf("singing's file is %s: %v", kindFile(root, soundVoice), err)
	}
}
