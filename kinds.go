package stapes

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/lixenwraith/audio"
)

// The kinds of sound a recording is read as, each with a set of its own in the
// root where rip.<kind>.toml holds one: one that drums play in, alone or under
// music, and with no drum singing alone and a guitar alone
const (
	soundDrums  = "drums"
	soundVoice  = "voice"
	soundGuitar = "guitar"
)

var ripKinds = []string{soundDrums, soundVoice, soundGuitar}

// What kindOf reads a kind by: how sure the pitch head is of a frame for
// something to sound in it, and the share of those frames the frame model's
// family head gives a family for the recording to be its own
const (
	kindSounds = 0.5
	kindShare  = 0.5
)

// kindOf is the kind of sound a recording is, by what the models already say
// of it: r, its rip at the root's own thresholds, for whether drums play, and
// f, its frames, for the family the frame model's head hears where something
// sounds; "" where they tell none. Drums alone are not told from drums under
// music: over a drum kit the pitch head finds pitches and the family head a bass.
func kindOf(r *audio.Rip, f *audio.RipFrames) string {
	if r == nil {
		return ""
	}
	if len(r.Hits()) > 0 { // audio keeps a drum that strikes four times a minute or more
		return soundDrums
	}
	share := familyShares(f)
	for _, k := range []string{soundVoice, soundGuitar} {
		if share != nil && share[slices.Index(audio.RipFamilies[:], k)] >= kindShare {
			return k
		}
	}
	return ""
}

// familyShares is the mean share the frame model's family head gives each of
// audio's RipFamilies over the frames something sounds in, those whose surest
// pitch reaches kindSounds; nil where no head says or nothing sounds
func familyShares(f *audio.RipFrames) []float64 {
	if f == nil || len(f.Family) == 0 || len(f.Salience) == 0 {
		return nil
	}
	k, pitches := len(audio.RipFamilies), f.Pitches().Len()
	share, n := make([]float64, k), 0
	for t := 0; (t+1)*k <= len(f.Family) && (t+1)*pitches <= len(f.Salience); t++ {
		if slices.Max(f.Salience[t*pitches:(t+1)*pitches]) < kindSounds {
			continue
		}
		n++
		for j, v := range f.Family[t*k : (t+1)*k] {
			share[j] += float64(v)
		}
	}
	if n == 0 {
		return nil
	}
	for j := range share {
		share[j] /= float64(n)
	}
	return share
}

// kindFile is where a root keeps kind's set, beside its rip.toml
func kindFile(root, kind string) string {
	return filepath.Join(root, strings.TrimSuffix(ripFile, ".toml")+"."+kind+".toml")
}

// loadKinds reads the sets a root holds for a kind of sound, each a document as
// rip.toml is; a file that does not load is named and left out
func loadKinds(root string, say func(string)) map[string]audio.RipParams {
	if root == "" {
		return nil
	}
	var kinds map[string]audio.RipParams
	for _, k := range ripKinds {
		f := kindFile(root, k)
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		p, err := audio.LoadRipParams(data)
		if err != nil {
			say(fmt.Sprintf("rip params %s: %v (the root's own used for %s)", f, err, k))
			continue
		}
		if kinds == nil {
			kinds = map[string]audio.RipParams{}
		}
		kinds[k] = p
		say(fmt.Sprintf("rip params %s, for a recording read as %s", f, k))
	}
	return kinds
}
