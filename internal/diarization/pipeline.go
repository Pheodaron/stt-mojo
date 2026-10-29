package diarization

import (
	"context"
	"errors"
	"fmt"
	"io"
)

var ErrInvalidAudio = errors.New("invalid PCM audio")

// Model is the two synchronous computations needed by the pipeline.
// Predict masks embeddings after validFrames; padding is always a suffix.
type Model interface {
	Embed(features []float32) ([]float32, error)
	Predict(embeddings []float32, validFrames int) ([]float32, error)
}

type Result struct {
	StartFrame uint64
	Logits     []float32
}

// Pipeline processes one recording at a time. Model weights are shared between
// calls; all audio and speaker state is local to Run.
type Pipeline struct {
	config    Config
	constants Constants
	model     Model
}

func New(config Config, constants Constants, model Model) *Pipeline {
	return &Pipeline{config: config, constants: constants, model: model}
}

func (p *Pipeline) Run(ctx context.Context, audio io.Reader, emit func(Result) error) error {
	features := newFeatureReader(audio, p.config, p.constants)
	chunks := newChunkReader(features, p.model, p.config)
	cache := speakerCache{config: p.config}
	var outputFrame int64

	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		chunk, err := chunks.next(ctx)
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		frames := min(chunk.newFrames*p.config.Factor, int(chunks.validMelFrames-outputFrame))
		if frames == 0 {
			return nil // Only the frontend's final padding frame remains.
		}

		input, prefix := cache.input(chunk.embeddings)
		logits, err := p.model.Predict(input, prefix+chunk.validFrames)
		if err != nil {
			return fmt.Errorf("predict: %w", err)
		}
		expected := len(input) / p.config.HiddenSize * p.config.Factor * p.config.Speakers
		if len(logits) != expected {
			return fmt.Errorf("invalid prediction shape: got %d logits, want %d", len(logits), expected)
		}

		if !chunk.last {
			cache.update(chunk, logits, p.constants.Silence)
		}
		start := (prefix + chunk.leftFrames) * p.config.Factor * p.config.Speakers
		end := start + frames*p.config.Speakers
		if err := emit(Result{StartFrame: uint64(outputFrame), Logits: logits[start:end]}); err != nil {
			return err
		}
		outputFrame += int64(frames)
		if chunk.last {
			return nil
		}
	}
}
