package diarization

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"slices"
	"testing"
	"testing/iotest"
)

func testConfig() (Config, Constants) {
	preemphasis := float32(.97)
	config := Config{
		Version: 1, MAXVersion: "26.7.0", Target: "cpu",
		SampleRate: 16000, HopLength: 160, NFFT: 512, Preemphasis: &preemphasis,
		MelBins: 4, HiddenSize: 4, Factor: 2, RotaryDim: 2, Speakers: 2,
		ChunkLength: 3, RightContext: 1, LeftContext: 1,
		FIFOLength: 3, UpdatePeriod: 2, CacheLength: 8, SilenceFrames: 1,
		PredictionThreshold: .25, LatestBoost: .05, StrongBoost: .75, WeakBoost: 1.5, MinPositiveRate: .5,
	}
	constants := Constants{
		Window:     make([]float32, config.NFFT),
		MelFilters: make([]float32, config.MelBins*(config.NFFT/2+1)),
		Silence:    make([]float32, config.HiddenSize),
	}
	for i := range constants.Window {
		constants.Window[i] = 1
	}
	for i := range constants.MelFilters {
		constants.MelFilters[i] = float32(i%7+1) * .001
	}
	return config, constants
}

// Numerical correctness uses the exported MAX fixture. This inexpensive model
// makes framing, state lifetime and duration tests independent of the SDK.
type testModel struct {
	config                          Config
	calls, maxFrames, maxProjection int
}

func (m *testModel) Embed(features []float32) ([]float32, error) {
	m.maxProjection = max(m.maxProjection, len(features)/(m.config.MelBins*m.config.Factor))
	output := make([]float32, len(features)/(m.config.MelBins*m.config.Factor)*m.config.HiddenSize)
	for i := range output {
		frame, channel := i/m.config.HiddenSize, i%m.config.MelBins
		output[i] = features[frame*m.config.MelBins*m.config.Factor+channel]
	}
	return output, nil
}

func (m *testModel) Predict(embeddings []float32, validFrames int) ([]float32, error) {
	frames := len(embeddings) / m.config.HiddenSize
	m.calls++
	m.maxFrames = max(m.maxFrames, frames)
	var sum float32
	for frame := 0; frame < validFrames; frame++ {
		sum += embeddings[frame*m.config.HiddenSize]
	}
	mean := sum / float32(frames)
	output := make([]float32, frames*m.config.Factor*m.config.Speakers)
	for i := range output {
		frame := i / (m.config.Speakers * m.config.Factor)
		output[i] = mean*.01 + embeddings[frame*m.config.HiddenSize]*.1
	}
	return output, nil
}

func TestTransportBoundariesAndLastFrames(t *testing.T) {
	for _, right := range []int{0, 1, 4} {
		for _, samples := range []int{160, 161, 319, 320, 511, 959, 960, 1120, 1280, 1607, 12017} {
			config, constants := testConfig()
			config.RightContext = right
			audio := make([]float32, samples)
			for i := range audio {
				audio[i] = float32(math.Sin(float64(i)*.17)) * .1
			}
			data := EncodeFloat32(audio)
			readers := []io.Reader{
				bytes.NewReader(data),
				iotest.OneByteReader(bytes.NewReader(data)),
				iotest.HalfReader(bytes.NewReader(data)),
			}
			var expected []float32
			for _, reader := range readers {
				pipeline := New(config, constants, &testModel{config: config})
				var actual []float32
				err := pipeline.Run(context.Background(), reader, func(result Result) error {
					if int(result.StartFrame) != len(actual)/config.Speakers {
						t.Fatal("discontinuous frame offsets")
					}
					actual = append(actual, result.Logits...)
					return nil
				})
				if err != nil {
					t.Fatal(err)
				}
				if len(actual) != samples/config.HopLength*config.Speakers {
					t.Fatalf("right=%d samples=%d: wrong output length %d", right, samples, len(actual))
				}
				if expected == nil {
					expected = actual
				} else if !slices.Equal(expected, actual) {
					t.Fatal("transport boundaries changed results")
				}
			}
		}
	}
}

type silenceReader struct{}

func (silenceReader) Read(dst []byte) (int, error) { clear(dst); return len(dst), nil }

func TestTwoHourRecording(t *testing.T) {
	config, constants := testConfig()
	config.ChunkLength, config.RightContext, config.LeftContext = 340, 40, 0
	config.Factor, config.Speakers = 8, 8
	config.CacheLength, config.FIFOLength, config.UpdatePeriod = 264, 40, 300
	model := &testModel{config: config}
	pipeline := New(config, constants, model)
	// Real PCM decoding and FFT, inexpensive model, no two-hour input allocation.
	audio := io.LimitReader(silenceReader{}, 2*3600*int64(config.SampleRate)*4)
	var frames uint64
	err := pipeline.Run(context.Background(), audio, func(result Result) error {
		if result.StartFrame != frames {
			t.Fatal("discontinuous output")
		}
		frames += uint64(len(result.Logits) / config.Speakers)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if frames != 720000 || model.calls != 265 {
		t.Fatalf("got %d frames in %d model calls", frames, model.calls)
	}
	if model.maxProjection > 380 || model.maxFrames > 684 {
		t.Fatal("model input exceeded the configured window")
	}
}

func TestCancellationAndOutputErrors(t *testing.T) {
	config, constants := testConfig()
	pipeline := New(config, constants, &testModel{config: config})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := pipeline.Run(ctx, bytes.NewReader(make([]byte, 40000)), func(Result) error { return nil })
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
	sentinel := errors.New("consumer stopped")
	err = pipeline.Run(context.Background(), bytes.NewReader(make([]byte, 40000)), func(Result) error { return sentinel })
	if !errors.Is(err, sentinel) {
		t.Fatalf("output error lost: %v", err)
	}
	err = pipeline.Run(context.Background(), iotest.ErrReader(sentinel), func(Result) error { return nil })
	if !errors.Is(err, sentinel) {
		t.Fatalf("read error lost: %v", err)
	}
}

func TestSequentialRecordingsStartFresh(t *testing.T) {
	config, constants := testConfig()
	pipeline := New(config, constants, &testModel{config: config})
	audio := make([]float32, 12017)
	for i := range audio {
		audio[i] = float32(math.Sin(float64(i)*.17)) * .1
	}
	data := EncodeFloat32(audio)
	run := func() []float32 {
		t.Helper()
		var logits []float32
		err := pipeline.Run(context.Background(), bytes.NewReader(data), func(result Result) error {
			if result.StartFrame != uint64(len(logits)/config.Speakers) {
				t.Fatal("recording inherited an output offset")
			}
			logits = append(logits, result.Logits...)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return logits
	}
	expected := run()
	sentinel := errors.New("stop this recording")
	err := pipeline.Run(context.Background(), bytes.NewReader(make([]byte, 40000)), func(Result) error { return sentinel })
	if !errors.Is(err, sentinel) {
		t.Fatalf("got %v", err)
	}
	if actual := run(); !slices.Equal(actual, expected) {
		t.Fatal("a previous recording changed the result")
	}
}

func TestInvalidPCM(t *testing.T) {
	config, constants := testConfig()
	pipeline := New(config, constants, &testModel{config: config})
	for _, data := range [][]byte{nil, {1}, make([]byte, 639), EncodeFloat32([]float32{float32(math.NaN())}), EncodeFloat32([]float32{float32(math.Inf(1))})} {
		err := pipeline.Run(context.Background(), bytes.NewReader(data), func(Result) error { return nil })
		if !errors.Is(err, ErrInvalidAudio) {
			t.Fatalf("invalid audio returned %v", err)
		}
	}
}
