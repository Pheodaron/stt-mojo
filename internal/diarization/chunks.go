package diarization

import (
	"context"
	"fmt"
	"io"
)

type chunk struct {
	embeddings  []float32
	leftFrames  int
	newFrames   int
	validFrames int
	last        bool
}

// chunkReader retains only left context and the unconsumed lookahead.
// The returned embeddings remain valid until the next call to next.
type chunkReader struct {
	config                     Config
	features                   *featureReader
	model                      Model
	mel                        []float32
	embeddings                 []float32
	leftFrames, previousFrames int
	startFrame, validMelFrames int64
	eof                        bool
}

func newChunkReader(features *featureReader, model Model, config Config) *chunkReader {
	return &chunkReader{
		config: config, features: features, model: model,
		mel: make([]float32, (config.ChunkLength+config.RightContext)*config.Factor*config.MelBins),
	}
}

func (r *chunkReader) next(ctx context.Context) (chunk, error) {
	config := r.config
	// Advance past the previous chunk, keeping its configured left context.
	consumed := r.leftFrames + r.previousFrames
	drop := max(0, consumed-config.LeftContext)
	kept := copy(r.embeddings, r.embeddings[drop*config.HiddenSize:])
	r.embeddings = r.embeddings[:kept]
	r.startFrame += int64(drop)
	r.leftFrames = consumed - drop

	available := len(r.embeddings)/config.HiddenSize - r.leftFrames
	needed := config.ChunkLength + config.RightContext - available
	if !r.eof && needed > 0 {
		buffer := r.mel[:needed*config.Factor*config.MelBins]
		frames, valid, err := r.features.readFrames(ctx, buffer)
		if err != nil && err != io.EOF {
			return chunk{}, err
		}
		r.eof = err == io.EOF
		r.validMelFrames += int64(valid)
		if frames > 0 {
			blocks := (frames + config.Factor - 1) / config.Factor
			clear(buffer[frames*config.MelBins : blocks*config.Factor*config.MelBins])
			projected, err := r.model.Embed(buffer[:blocks*config.Factor*config.MelBins])
			if err != nil {
				return chunk{}, fmt.Errorf("embed: %w", err)
			}
			if len(projected) != blocks*config.HiddenSize {
				return chunk{}, fmt.Errorf("invalid embedding shape")
			}
			r.embeddings = append(r.embeddings, projected...)
		}
	}

	available = len(r.embeddings)/config.HiddenSize - r.leftFrames
	if available == 0 {
		return chunk{}, io.EOF
	}
	r.previousFrames = min(config.ChunkLength, available)
	end := r.leftFrames + min(r.previousFrames+config.RightContext, available)
	validEmbeddings := (r.validMelFrames + int64(config.Factor) - 1) / int64(config.Factor)
	return chunk{
		embeddings:  r.embeddings[:end*config.HiddenSize],
		leftFrames:  r.leftFrames,
		newFrames:   r.previousFrames,
		validFrames: min(end, int(validEmbeddings-r.startFrame)),
		last:        r.eof && r.previousFrames == available,
	}, nil
}
