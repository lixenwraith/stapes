package stapes

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strings"

	"github.com/lixenwraith/audio"
	"github.com/lixenwraith/toml"
)

// ripBundle is a root's rip.bundle.toml, the rip's files as one line, what a
// client takes in place of a file: the models in the order they load, the frame
// model first, each with its heads; the judge, with the digests of the heads its
// row reads; the sets by kind; the read-back's table they were read with, a line a
// row; and the audio whose decoders they were read under. Each file is named in
// the root as the root names it, with its sha256.
type ripBundle struct {
	Needs  string        `toml:"needs"`
	Table  []string      `toml:"table"`
	Models []bundleModel `toml:"model"`
	Judge  *bundleJudge  `toml:"judge,omitempty"`
	Set    *bundleSet    `toml:"set,omitempty"`
	Kinds  []bundleSet   `toml:"kind,omitempty"`
}

type bundleModel struct {
	File   string   `toml:"file"`
	Sha256 string   `toml:"sha256"`
	Heads  []string `toml:"heads"`
}

type bundleJudge struct {
	File        string `toml:"file"`
	Sha256      string `toml:"sha256"`
	NotesDigest string `toml:"notes_digest"`
	FrameDigest string `toml:"frame_digest"`
}

// bundleSet is a set's file: the root's own, rip.toml, or a kind's, which Name names
type bundleSet struct {
	Name   string `toml:"name,omitempty"`
	File   string `toml:"file"`
	Sha256 string `toml:"sha256"`
}

// rootRipFiles are the files a root's rip reads by their names
func rootRipFiles() []string {
	files := []string{frameFile, oldFrameFile, notesFile, judgeFile, ripFile}
	for _, k := range ripKinds {
		files = append(files, filepath.Base(kindFile("", k)))
	}
	return files
}

// count is how many files the bundle names
func (b *ripBundle) count() int {
	n := len(b.Models) + len(b.Kinds)
	if b.Judge != nil {
		n++
	}
	if b.Set != nil {
		n++
	}
	return n
}

// files is the bundle's model files in root, in the order they load
func (b *ripBundle) files(root string) []string {
	out := make([]string, len(b.Models))
	for i, m := range b.Models {
		out[i] = filepath.Join(root, m.File)
	}
	return out
}

// fileSum is the sha256 of a file, in hex
func fileSum(file string) (string, error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

// readBundle reads root's bundle and holds the root to it: each file it names
// there with its sha256, the frame model first as ripframe.bin and each other
// under the name the root gives it, a kind's set one this stapes reads a recording
// as, and no other of the files a root's rip reads; a refusal names the file.
func readBundle(root string) (*ripBundle, error) {
	data, err := os.ReadFile(filepath.Join(root, bundleFile))
	if err != nil {
		return nil, err
	}
	doc, err := toml.ParseDocument(data)
	if err != nil {
		return nil, err
	}
	var b ripBundle
	if keys := doc.Unmapped(&b); len(keys) > 0 {
		return nil, toml.UnknownKeys(keys)
	}
	if err := doc.Decode(&b); err != nil {
		return nil, err
	}
	if len(b.Models) == 0 || b.Models[0].File != frameFile {
		return nil, fmt.Errorf("it names no frame model %s first", frameFile)
	}
	named := map[string]bool{}
	hold := func(file, sum, as string) error {
		if as != "" && file != as {
			return fmt.Errorf("it names %s where the root's is %s", file, as)
		}
		if file == "" || file != filepath.Base(file) || named[file] {
			return fmt.Errorf("it names %q, no file of the root's own or one named twice", file)
		}
		got, err := fileSum(filepath.Join(root, file))
		if err != nil {
			return fmt.Errorf("%s: %w", file, err)
		}
		if got != sum {
			return fmt.Errorf("%s is not the bundle's: its sha256 is %.16s, the bundle's %.16s", file, got, sum)
		}
		named[file] = true
		return nil
	}
	for _, m := range b.Models {
		if err := hold(m.File, m.Sha256, ""); err != nil {
			return nil, err
		}
	}
	if j := b.Judge; j != nil {
		if err := hold(j.File, j.Sha256, judgeFile); err != nil {
			return nil, err
		}
	}
	if set := b.Set; set != nil {
		if err := hold(set.File, set.Sha256, ripFile); err != nil {
			return nil, err
		}
	}
	for _, k := range b.Kinds {
		if !slices.Contains(ripKinds, k.Name) {
			return nil, fmt.Errorf("it gives a set for %q, a kind this stapes reads no recording as (%s)", k.Name, strings.Join(ripKinds, ", "))
		}
		if err := hold(k.File, k.Sha256, filepath.Base(kindFile("", k.Name))); err != nil {
			return nil, err
		}
	}
	for _, f := range rootRipFiles() {
		if !named[f] && regular(filepath.Join(root, f)) != "" {
			return nil, fmt.Errorf("%s is not the bundle's: it names no such file", f)
		}
	}
	return &b, nil
}

// cmdBundle writes a root's bundle of the files its rip reads (bundleOf)
func cmdBundle(s *Session, a []string) error {
	usage := errors.New("want bundle <dir> [table <file>] [needs <version>]")
	dir, o := s.path(a[0]), bundleOptions{}
	for w := a[1:]; len(w) > 0; w = w[2:] {
		switch {
		case len(w) > 1 && w[0] == "table":
			o.table = s.path(w[1])
		case len(w) > 1 && w[0] == "needs":
			o.needs = w[1]
		default:
			return usage
		}
	}
	b, err := bundleOf(dir, o)
	if err != nil {
		return err
	}
	data, err := toml.Marshal(b)
	if err != nil {
		return err
	}
	file := filepath.Join(dir, bundleFile)
	if err := os.WriteFile(file+".part", data, 0o644); err != nil {
		return err
	}
	if err := os.Rename(file+".part", file); err != nil {
		return err
	}
	fmt.Fprintf(s.out, "wrote %s: %d file(s) as one line, %d model(s), a table of %d line(s), read under audio %s\n", file, b.count(), len(b.Models), len(b.Table), b.Needs)
	return nil
}

// bundleOptions is what the bundle verb is told: the read-back's table the files
// were read with, and the audio they were read under where not this build's
type bundleOptions struct {
	table, needs string
}

// bundleOf is the bundle of the files dir's rip reads, as a root reads them, once
// they load together: its models, each with its heads; its judge, loaded through
// them, with the digests of the heads its row reads; its sets; o's table and the
// audio o names, or this build's
func bundleOf(dir string, o bundleOptions) (*ripBundle, error) {
	var b ripBundle
	files := []string{filepath.Join(dir, frameFile)}
	if f := regular(filepath.Join(dir, notesFile)); f != "" {
		files = append(files, f)
	}
	fe, _, err := loadFrontEnd(files)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", dir, err)
	}
	for i, f := range files {
		sum, err := fileSum(f)
		if err != nil {
			return nil, err
		}
		m := bundleModel{File: filepath.Base(f), Sha256: sum}
		for _, h := range fe.Models[i].Config().Outputs {
			m.Heads = append(m.Heads, h.Name)
		}
		b.Models = append(b.Models, m)
	}
	if f := regular(filepath.Join(dir, judgeFile)); f != "" {
		data, err := os.ReadFile(f)
		if err == nil {
			_, err = audio.LoadRipJudge(strings.NewReader(string(data)), fe.Weights)
		}
		if err != nil {
			return nil, fmt.Errorf("%s: %w", f, err)
		}
		sum := sha256.Sum256(data)
		b.Judge = &bundleJudge{File: judgeFile, Sha256: hex.EncodeToString(sum[:]),
			NotesDigest: fmt.Sprintf("%016x", fe.Weights.Notes), FrameDigest: fmt.Sprintf("%016x", fe.Weights.Frame)}
	}
	set := func(name, file string) (*bundleSet, error) {
		data, err := os.ReadFile(file)
		if err != nil {
			return nil, err
		}
		if _, err := audio.LoadRipParams(data); err != nil {
			return nil, fmt.Errorf("%s: %w", file, err)
		}
		sum := sha256.Sum256(data)
		return &bundleSet{Name: name, File: filepath.Base(file), Sha256: hex.EncodeToString(sum[:])}, nil
	}
	if f := regular(filepath.Join(dir, ripFile)); f != "" {
		if b.Set, err = set("", f); err != nil {
			return nil, err
		}
	}
	for _, k := range ripKinds {
		if f := regular(kindFile(dir, k)); f != "" {
			s, err := set(k, f)
			if err != nil {
				return nil, err
			}
			b.Kinds = append(b.Kinds, *s)
		}
	}
	if o.table != "" {
		data, err := os.ReadFile(o.table)
		if err != nil {
			return nil, err
		}
		b.Table = strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	}
	if b.Needs = o.needs; b.Needs == "" {
		b.Needs = builtAudio()
	}
	return &b, nil
}

// builtAudio is the version of the audio this stapes was built with
func builtAudio() string {
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, m := range info.Deps {
			if m.Path == "github.com/lixenwraith/audio" {
				return m.Version
			}
		}
	}
	return "unknown"
}
