package diarization

import (
	"context"
	"fmt"
)

// Engine executes the two exported MAX graphs. One shared engine serves all
// recordings; each Pipeline owns its frontend and speaker-cache state.
type Engine interface {
	Embed(context.Context, []float32) ([]float32, error)
	Infer(context.Context, []float32, []bool) ([]float32, error)
}

type Result struct {
	StartFrame uint64
	Logits     []float32
}

type Pipeline struct {
	c                    Config
	k                    Constants
	engine               Engine
	frontend             *frontend
	cache                speakerCache
	features             []float32 // Log-mel frames waiting for projection.
	embeddings           []float32 // Retained left context followed by unconsumed frames.
	valid                []bool    // Encoder masks for embeddings and pending feature groups.
	featureCount         int       // Mel frames currently in features.
	lookback             int       // Left-context prefix length in embeddings.
	validFrames, emitted int64
	closed               bool
}

func NewPipeline(c Config, k Constants, engine Engine) *Pipeline {
	return &Pipeline{c: c, k: k, engine: engine, frontend: newFrontend(c, k), cache: speakerCache{c: c}}
}

// Push consumes a bounded PCM portion. emit provides backpressure: the next
// inference step does not run until the caller has consumed this result.
func (p *Pipeline) Push(ctx context.Context, audio []float32, emit func(Result) error) error {
	if p.closed {
		return fmt.Errorf("audio stream is already closed")
	}
	return p.frontend.push(audio, func(feature []float32, valid bool) error {
		return p.feature(ctx, feature, valid, emit)
	})
}

func (p *Pipeline) Finish(ctx context.Context, emit func(Result) error) error {
	if p.closed {
		return fmt.Errorf("audio stream is already closed")
	}
	p.closed = true
	if p.frontend.samples < int64(p.c.HopLength) {
		return fmt.Errorf("audio must contain at least %d samples", p.c.HopLength)
	}
	if err := p.frontend.finish(func(feature []float32, valid bool) error { return p.feature(ctx, feature, valid, emit) }); err != nil {
		return err
	}
	if len(p.features) > 0 {
		padding := (p.c.Factor - p.featureCount%p.c.Factor) % p.c.Factor
		p.features = append(p.features, make([]float32, padding*p.c.MelBins)...)
		if err := p.embed(ctx); err != nil {
			return err
		}
	}
	return p.drain(ctx, true, emit)
}

func (p *Pipeline) feature(ctx context.Context, feature []float32, valid bool, emit func(Result) error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if valid {
		p.validFrames++
	}
	if p.featureCount%p.c.Factor == 0 {
		p.valid = append(p.valid, valid)
	}
	p.features = append(p.features, feature...)
	p.featureCount++
	queued := len(p.embeddings)/p.c.HiddenSize - p.lookback
	// Batch the projection for exactly the next chunk plus lookahead. No
	// embeddings for the whole recording are materialized.
	needed := p.c.ChunkLength + p.c.RightContext - queued
	if p.featureCount == needed*p.c.Factor {
		if err := p.embed(ctx); err != nil {
			return err
		}
		if p.closed {
			return nil
		}
		return p.drain(ctx, false, emit)
	}
	return nil
}

func (p *Pipeline) embed(ctx context.Context) error {
	values, err := p.engine.Embed(ctx, p.features)
	if err != nil {
		return err
	}
	if len(values) != len(p.features)/(p.c.MelBins*p.c.Factor)*p.c.HiddenSize {
		return fmt.Errorf("MAX returned an invalid embedding shape")
	}
	p.embeddings = append(p.embeddings, values...)
	p.features = p.features[:0]
	p.featureCount = 0
	return nil
}

func (p *Pipeline) drain(ctx context.Context, final bool, emit func(Result) error) error {
	c := p.c
	for {
		remaining := len(p.embeddings)/c.HiddenSize - p.lookback
		if remaining <= 0 || (!final && remaining < c.ChunkLength+c.RightContext) {
			return nil
		}
		frames := min(c.ChunkLength, remaining)
		end := p.lookback + min(frames+c.RightContext, remaining)
		inputs := make([]float32, 0, len(p.cache.embeds)+len(p.cache.fifo)+end*c.HiddenSize)
		inputs = append(inputs, p.cache.embeds...)
		inputs = append(inputs, p.cache.fifo...)
		cached := len(inputs) / c.HiddenSize
		inputs = append(inputs, p.embeddings[:end*c.HiddenSize]...)
		valid := make([]bool, cached+end)
		for i := 0; i < cached; i++ {
			valid[i] = true
		}
		copy(valid[cached:], p.valid[:end])
		logits, err := p.engine.Infer(ctx, inputs, valid)
		if err != nil {
			return err
		}
		if len(logits) != len(valid)*c.Factor*c.Speakers {
			return fmt.Errorf("MAX returned an invalid prediction shape")
		}
		last := final && frames == remaining
		if !last {
			p.cache.update(inputs, logits, valid, p.lookback, frames, p.k.Silence)
		}
		count := frames * c.Factor
		if final {
			count = min(count, int(p.validFrames-p.emitted))
		}
		if count > 0 {
			lo := (cached + p.lookback) * c.Factor * c.Speakers
			if err = emit(Result{StartFrame: uint64(p.emitted), Logits: logits[lo : lo+count*c.Speakers]}); err != nil {
				return err
			}
			p.emitted += int64(count)
		}
		consumed := p.lookback + frames
		drop := max(0, consumed-c.LeftContext)
		p.embeddings = p.embeddings[:copy(p.embeddings, p.embeddings[drop*c.HiddenSize:])]
		p.valid = p.valid[:copy(p.valid, p.valid[drop:])]
		p.lookback = consumed - drop
	}
}
