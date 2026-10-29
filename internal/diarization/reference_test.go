package diarization

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"testing/iotest"
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
		maxError := compareFeatures(t, c, k, audio, expected)
		t.Logf("samples=%d max mel error=%.8g", count, maxError)
	}
}

func TestCacheSelectionAgainstReference(t *testing.T) {
	dir := fixtureDir(t)
	c, _ := testConfig()
	c.HiddenSize = 1
	c.Speakers = 8
	c.CacheLength = 264
	s := speakerCache{config: c, probabilities: readFloats(t, dir, "cache-probs.f32"), embeddings: make([]float32, 564)}
	for i := range s.embeddings {
		s.embeddings[i] = float32(i)
	}
	s.compress([]float32{-1})
	expected := readFloats(t, dir, "cache-selected.f32")
	if !slices.Equal(s.embeddings, expected) {
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
	maxError := compareFeatures(t, c, k, audio, expected)
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
	s := speakerCache{config: c}
	for step, r := range steps {
		prefix := (len(s.embeddings) + len(s.fifo)) / c.HiddenSize
		window := chunk{embeddings: r.Inputs[prefix*c.HiddenSize:], leftFrames: r.Lookback, newFrames: r.Frames}
		s.update(window, r.Logits, []float32{-1, -1, -1, -1})
		for name, pair := range map[string][2][]float32{"cache": {s.embeddings, r.Cache}, "FIFO": {s.fifo, r.FIFO}, "probs": {s.probabilities, r.Probs}} {
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

func compareFeatures(t *testing.T, config Config, constants Constants, audio, expected []float32) float64 {
	t.Helper()
	reader := newFeatureReader(iotest.OneByteReader(bytes.NewReader(EncodeFloat32(audio))), config, constants)
	buffer := make([]float32, 17*config.MelBins)
	var actual []float32
	validFrames := 0
	for {
		frames, valid, err := reader.readFrames(context.Background(), buffer)
		actual = append(actual, buffer[:frames*config.MelBins]...)
		validFrames += valid
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(actual) != len(expected) || validFrames != len(audio)/config.HopLength {
		t.Fatal("wrong feature shape or valid length")
	}
	var maxError float64
	for i, value := range actual {
		maxError = max(maxError, math.Abs(float64(value-expected[i])))
	}
	if !(maxError <= 3e-5) {
		t.Fatalf("frontend max error %.8g", maxError)
	}
	return maxError
}
