package diarization

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"
)

func testConfig() (Config, Constants) {
	pre := float32(.97)
	c := Config{Version: 1, MAXVersion: "26.7.0", Target: "cpu", SampleRate: 16000, HopLength: 160, NFFT: 512, Preemphasis: &pre,
		MelBins: 4, HiddenSize: 4, Factor: 2, RotaryDim: 2, Speakers: 2, ChunkLength: 3, RightContext: 1, LeftContext: 1,
		FIFOLength: 3, UpdatePeriod: 2, CacheLength: 8, SilenceFrames: 1, PredictionThreshold: .25, LatestBoost: .05, StrongBoost: .75, WeakBoost: 1.5, MinPositiveRate: .5}
	k := Constants{Window: make([]float32, c.NFFT), MelFilters: make([]float32, c.MelBins*(c.NFFT/2+1)), Silence: make([]float32, c.HiddenSize)}
	for i := range k.Window {
		k.Window[i] = 1
	}
	for i := range k.MelFilters {
		k.MelFilters[i] = float32(i%7+1) * .001
	}
	return c, k
}

// A cheap state-sensitive engine for transport, memory and cancellation tests.
// Neural correctness is checked separately against the exported upstream model.
type testEngine struct {
	c                Config
	calls, maxFrames int
}

func (e *testEngine) Embed(ctx context.Context, features []float32) ([]float32, error) {
	out := make([]float32, len(features)/(e.c.MelBins*e.c.Factor)*e.c.HiddenSize)
	for i := range out {
		out[i] = features[(i/e.c.HiddenSize)*e.c.MelBins*e.c.Factor+i%e.c.MelBins]
	}
	return out, ctx.Err()
}
func (e *testEngine) Infer(ctx context.Context, embeds []float32, valid []bool) ([]float32, error) {
	e.calls++
	e.maxFrames = max(e.maxFrames, len(valid))
	var sum float32
	for i, v := range valid {
		if v {
			sum += embeds[i*e.c.HiddenSize]
		}
	}
	mean := sum / float32(len(valid))
	out := make([]float32, len(valid)*e.c.Factor*e.c.Speakers)
	for i := range out {
		out[i] = mean*.01 + embeds[(i/(e.c.Speakers*e.c.Factor))*e.c.HiddenSize]*.1
	}
	return out, ctx.Err()
}

func TestTransportBoundariesAndLastFrames(t *testing.T) {
	for _, right := range []int{0, 1, 4} {
		for _, count := range []int{160, 161, 319, 320, 511, 959, 960, 1120, 1280, 1607, 12017} {
			c, k := testConfig()
			c.RightContext = right
			audio := make([]float32, count)
			for i := range audio {
				audio[i] = float32(math.Sin(float64(i)*.17)) * .1
			}
			var expected []float32
			for _, chunk := range []int{count, 1, 17, 160, 513} {
				p := NewPipeline(c, k, &testEngine{c: c})
				var actual []float32
				emit := func(r Result) error {
					if int(r.StartFrame) != len(actual)/c.Speakers {
						t.Fatal("discontinuous frame offsets")
					}
					actual = append(actual, r.Logits...)
					return nil
				}
				for start := 0; start < count; start += chunk {
					if err := p.Push(context.Background(), audio[start:min(start+chunk, count)], emit); err != nil {
						t.Fatal(err)
					}
				}
				if err := p.Finish(context.Background(), emit); err != nil {
					t.Fatal(err)
				}
				if len(actual) != count/c.HopLength*c.Speakers {
					t.Fatalf("right=%d samples=%d chunk=%d: wrong output length %d", right, count, chunk, len(actual))
				}
				if expected == nil {
					expected = actual
				} else if !slices.Equal(expected, actual) {
					t.Fatalf("right=%d samples=%d chunk=%d: transport boundary changed results", right, count, chunk)
				}
			}
		}
	}
}

func TestTwoHourStateIsBounded(t *testing.T) {
	c, k := testConfig()
	c.ChunkLength = 340
	c.RightContext = 40
	c.LeftContext = 0
	c.Factor = 8
	c.Speakers = 8
	c.CacheLength = 264
	c.FIFOLength = 40
	c.UpdatePeriod = 300
	e := &testEngine{c: c}
	p := NewPipeline(c, k, e)
	mel := []float32{.1, .2, .3, .4}
	emit := func(Result) error { return nil }
	// Exercise two hours of 10 ms frames without spending time on repeated FFTs
	// or neural inference. This verifies duration-independent orchestration state.
	for i := 0; i < 720000; i++ {
		if err := p.feature(context.Background(), mel, true, emit); err != nil {
			t.Fatal(err)
		}
		if len(p.features) > (c.ChunkLength+c.RightContext)*c.Factor*c.MelBins || len(p.embeddings) > (c.LeftContext+c.ChunkLength+c.RightContext)*c.HiddenSize || len(p.cache.embeds) > c.CacheLength*c.HiddenSize || len(p.cache.fifo) > c.FIFOLength*c.HiddenSize {
			t.Fatal("state exceeded its configured bound")
		}
	}
	if !p.cache.compressed {
		t.Fatal("speaker cache was never compressed")
	}
	if e.calls < 260 || e.calls > 265 {
		t.Fatalf("unexpected step count: %d", e.calls)
	}
}

func TestCancellationAndBackpressureErrors(t *testing.T) {
	c, k := testConfig()
	p := NewPipeline(c, k, &testEngine{c: c})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := p.Push(ctx, make([]float32, 512), func(Result) error { return nil }); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
	p = NewPipeline(c, k, &testEngine{c: c})
	sentinel := errors.New("consumer stopped")
	err := p.Push(context.Background(), make([]float32, 5000), func(Result) error { return sentinel })
	if !errors.Is(err, sentinel) {
		t.Fatalf("send error lost: %v", err)
	}
}

func TestPCMValidation(t *testing.T) {
	for _, data := range [][]byte{{1}, EncodeFloat32([]float32{float32(math.NaN())}), EncodeFloat32([]float32{float32(math.Inf(1))})} {
		if _, err := DecodePCM(data); err == nil {
			t.Fatal("accepted invalid PCM")
		}
	}
}
