package diarization

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func fixtureDir(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("NEMOTRON_TEST_BUNDLE")
	if dir == "" {
		t.Skip("set NEMOTRON_TEST_BUNDLE; see scripts/prepare_go_fixtures.py")
	}
	return dir
}
func readFloats(t *testing.T, dir, name string) []float32 {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		t.Fatal(err)
	}
	v, err := DecodePCM(data)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func TestFrontendAgainstUpstream(t *testing.T) {
	dir := fixtureDir(t)
	c, k, err := LoadBundle(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, count := range []int{160, 161, 319, 320, 511, 512, 959, 960, 1120, 1280, 1600, 1607, 12017} {
		audio := readFloats(t, dir, fmt.Sprintf("audio-%d.f32", count))
		expected := readFloats(t, dir, fmt.Sprintf("mel-%d.f32", count))
		f := newFrontend(c, k)
		var actual []float32
		emit := func(mel []float32, valid bool) error {
			if valid != (len(actual)/c.MelBins < count/c.HopLength) {
				t.Fatal("wrong feature mask")
			}
			actual = append(actual, mel...)
			return nil
		}
		for start := 0; start < count; start += 17 {
			if err = f.push(audio[start:min(start+17, count)], emit); err != nil {
				t.Fatal(err)
			}
		}
		if err = f.finish(emit); err != nil {
			t.Fatal(err)
		}
		if len(actual) != len(expected) {
			t.Fatal("wrong feature shape")
		}
		var maxError float64
		for i, x := range actual {
			maxError = max(maxError, math.Abs(float64(x-expected[i])))
		}
		if !(maxError <= 3e-5) {
			t.Fatalf("%d samples: frontend max error %.8g", count, maxError)
		}
		t.Logf("samples=%d max mel error=%.8g", count, maxError)
	}
}

func TestCacheSelectionAgainstReference(t *testing.T) {
	dir := fixtureDir(t)
	c, _ := testConfig()
	c.HiddenSize = 1
	c.Speakers = 8
	c.CacheLength = 264
	s := speakerCache{c: c, probs: readFloats(t, dir, "cache-probs.f32"), embeds: make([]float32, 564)}
	for i := range s.embeds {
		s.embeds[i] = float32(i)
	}
	s.compress([]float32{-1})
	expected := readFloats(t, dir, "cache-selected.f32")
	if !slices.Equal(s.embeds, expected) {
		t.Fatal("cache frame selection differs from the NeMo policy")
	}
}

func TestProductionFrontendAgainstUpstream(t *testing.T) {
	dir := fixtureDir(t)
	c, k := testConfig()
	c.MelBins = 128
	constants := readFloats(t, dir, "frontend128-constants.f32")
	k.Window, k.MelFilters = constants[:c.NFFT], constants[c.NFFT:]
	audio := readFloats(t, dir, "frontend128-audio.f32")
	expected := readFloats(t, dir, "frontend128-mel.f32")
	f := newFrontend(c, k)
	var actual []float32
	emit := func(mel []float32, _ bool) error { actual = append(actual, mel...); return nil }
	for start := 0; start < len(audio); start += 137 {
		if err := f.push(audio[start:min(start+137, len(audio))], emit); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.finish(emit); err != nil {
		t.Fatal(err)
	}
	if len(actual) != len(expected) {
		t.Fatal("wrong production feature shape")
	}
	var maxError float64
	for i, x := range actual {
		maxError = max(maxError, math.Abs(float64(x-expected[i])))
	}
	if !(maxError <= 3e-5) {
		t.Fatalf("production frontend max error %.8g", maxError)
	}
	t.Logf("128-channel frontend max mel error=%.8g", maxError)
}

func TestStateUpdatesAgainstNeMo(t *testing.T) {
	dir := fixtureDir(t)
	data, err := os.ReadFile(filepath.Join(dir, "nemo-updates.json"))
	if os.IsNotExist(err) {
		t.Skip("prepare fixtures with --nemo-source to check original NeMo methods")
	}
	if err != nil {
		t.Fatal(err)
	}
	var steps []struct {
		Inputs, Logits, Cache, Probs, FIFO []float32
		Lookback, Frames                   int
	}
	if err = json.Unmarshal(data, &steps); err != nil {
		t.Fatal(err)
	}
	c, _ := testConfig()
	s := speakerCache{c: c}
	for step, r := range steps {
		valid := make([]bool, len(r.Inputs)/c.HiddenSize)
		for i := range valid {
			valid[i] = true
		}
		s.update(r.Inputs, r.Logits, valid, r.Lookback, r.Frames, []float32{-1, -1, -1, -1})
		for name, pair := range map[string][2][]float32{"cache": {s.embeds, r.Cache}, "FIFO": {s.fifo, r.FIFO}, "probs": {s.probs, r.Probs}} {
			if len(pair[0]) != len(pair[1]) {
				t.Fatalf("step %d %s shape differs from NeMo", step, name)
			}
			for i, x := range pair[0] {
				if math.Abs(float64(x-pair[1][i])) > 3e-6 {
					t.Fatalf("step %d %s[%d]: Go=%g NeMo=%g", step, name, i, x, pair[1][i])
				}
			}
		}
	}
	if !s.compressed {
		t.Fatal("NeMo fixture did not exercise compression")
	}
}
