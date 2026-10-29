package diarization

import (
	"cmp"
	"math"
	"slices"
)

// speakerCache keeps recent context in FIFO and representative older frames
// grouped by speaker. Its policy follows NeMo streaming_update/_compress_spkcache.
type speakerCache struct {
	config        Config
	embeddings    []float32
	probabilities []float32
	fifo          []float32
	compressed    bool
}

func (s *speakerCache) input(embeddings []float32) ([]float32, int) {
	input := make([]float32, 0, len(s.embeddings)+len(s.fifo)+len(embeddings))
	input = append(input, s.embeddings...)
	input = append(input, s.fifo...)
	prefix := len(input) / s.config.HiddenSize
	return append(input, embeddings...), prefix
}

func (s *speakerCache) update(chunk chunk, logits, silence []float32) {
	config := s.config
	cacheFrames := len(s.embeddings) / config.HiddenSize
	fifoFrames := len(s.fifo) / config.HiddenSize
	start := chunk.leftFrames * config.HiddenSize
	end := start + chunk.newFrames*config.HiddenSize
	s.fifo = append(s.fifo, chunk.embeddings[start:end]...)
	if fifoFrames+chunk.newFrames <= config.FIFOLength {
		return
	}

	pop := min(max(config.UpdatePeriod, fifoFrames+chunk.newFrames-config.FIFOLength), fifoFrames+chunk.newFrames)
	probs := poolProbabilities(logits, config.Factor, config.Speakers)
	if !s.compressed {
		// Until compression, cache frames are chronological: refresh their scores.
		s.probabilities = append(s.probabilities[:0], probs[:cacheFrames*config.Speakers]...)
	}
	oldFIFO := probs[cacheFrames*config.Speakers : (cacheFrames+fifoFrames)*config.Speakers]
	newStart := (cacheFrames + fifoFrames + chunk.leftFrames) * config.Speakers
	newChunk := probs[newStart : newStart+chunk.newFrames*config.Speakers]
	fifoProbs := append(slices.Clone(oldFIFO), newChunk...)

	s.embeddings = append(s.embeddings, s.fifo[:pop*config.HiddenSize]...)
	s.probabilities = append(s.probabilities, fifoProbs[:pop*config.Speakers]...)
	s.fifo = slices.Clone(s.fifo[pop*config.HiddenSize:])
	if len(s.embeddings)/config.HiddenSize > config.CacheLength {
		s.compress(silence)
		s.compressed = true
	}
}

func poolProbabilities(logits []float32, factor, speakers int) []float32 {
	pooled := make([]float32, len(logits)/factor)
	for frame := 0; frame < len(pooled)/speakers; frame++ {
		averages := pooled[frame*speakers : (frame+1)*speakers]
		for step := 0; step < factor; step++ {
			start := (frame*factor + step) * speakers
			for speaker, logit := range logits[start : start+speakers] {
				averages[speaker] += sigmoid(logit)
			}
		}
		for speaker := range averages {
			averages[speaker] /= float32(factor)
		}
	}
	return pooled
}

func sigmoid(value float32) float32 {
	if value >= 0 {
		return float32(1 / (1 + math.Exp(-float64(value))))
	}
	exponent := math.Exp(float64(value))
	return float32(exponent / (1 + exponent))
}

// scores favors confident single-speaker frames. A column belongs to one
// speaker; silence and overlap are excluded according to the checkpoint policy.
func (s *speakerCache) scores() [][]float32 {
	config := s.config
	frames := len(s.probabilities) / config.Speakers
	columns := make([][]float32, config.Speakers)
	for speaker := range columns {
		columns[speaker] = make([]float32, frames)
	}
	complements := make([]float32, config.Speakers)
	for frame := 0; frame < frames; frame++ {
		row := s.probabilities[frame*config.Speakers : (frame+1)*config.Speakers]
		var sum float32
		for speaker, probability := range row {
			complements[speaker] = float32(math.Log(float64(max(1-probability, config.PredictionThreshold))))
			sum += complements[speaker]
		}
		for speaker, probability := range row {
			logProbability := float32(math.Log(float64(max(probability, config.PredictionThreshold))))
			score := logProbability - complements[speaker] + sum + float32(math.Ln2)
			if probability <= .5 {
				score = float32(math.Inf(-1))
			}
			columns[speaker][frame] = score
		}
	}
	budget := config.CacheLength/config.Speakers - config.SilenceFrames
	minimum := int(math.Floor(float64(budget) * config.MinPositiveRate))
	for _, column := range columns {
		positive := 0
		for _, score := range column {
			if score > 0 {
				positive++
			}
		}
		if positive >= minimum {
			for frame, score := range column {
				if score <= 0 {
					column[frame] = float32(math.Inf(-1))
				}
			}
		}
	}
	return columns
}

type cacheFrame struct {
	speaker int
	frame   int
	score   float32
	silence bool
}

func (s *speakerCache) compress(silence []float32) {
	config := s.config
	frames := len(s.embeddings) / config.HiddenSize
	budget := config.CacheLength/config.Speakers - config.SilenceFrames
	var candidates []cacheFrame
	for speaker, scores := range s.scores() {
		for frame := config.CacheLength; frame < frames; frame++ {
			scores[frame] += config.LatestBoost
		}
		boostTop(scores, int(math.Floor(float64(budget)*config.StrongBoost)), 2*float32(math.Ln2))
		boostTop(scores, int(math.Floor(float64(budget)*config.WeakBoost)), float32(math.Ln2))
		for frame, score := range scores {
			if !math.IsInf(float64(score), -1) {
				candidates = append(candidates, cacheFrame{speaker: speaker, frame: frame, score: score})
			}
		}
		for extra := 0; extra < config.SilenceFrames; extra++ {
			candidates = append(candidates, cacheFrame{
				speaker: speaker, frame: frames + extra, score: float32(math.Inf(1)), silence: true,
			})
		}
	}
	// Stable ties use the original speaker/frame order. torch.topk leaves ties
	// unspecified; either choice follows the same policy for equal scores.
	slices.SortStableFunc(candidates, func(a, b cacheFrame) int { return cmp.Compare(b.score, a.score) })
	selected := candidates[:min(len(candidates), config.CacheLength)]
	slices.SortFunc(selected, func(a, b cacheFrame) int {
		if a.speaker != b.speaker {
			return cmp.Compare(a.speaker, b.speaker)
		}
		return cmp.Compare(a.frame, b.frame)
	})

	embeddings := make([]float32, config.CacheLength*config.HiddenSize)
	probabilities := make([]float32, config.CacheLength*config.Speakers)
	for frame := 0; frame < config.CacheLength; frame++ {
		dst := embeddings[frame*config.HiddenSize : (frame+1)*config.HiddenSize]
		if frame >= len(selected) || selected[frame].silence {
			copy(dst, silence)
			continue
		}
		source := selected[frame].frame
		copy(dst, s.embeddings[source*config.HiddenSize:(source+1)*config.HiddenSize])
		copy(probabilities[frame*config.Speakers:], s.probabilities[source*config.Speakers:(source+1)*config.Speakers])
	}
	s.embeddings, s.probabilities = embeddings, probabilities
}

func boostTop(scores []float32, count int, amount float32) {
	indices := make([]int, len(scores))
	for frame := range indices {
		indices[frame] = frame
	}
	slices.SortStableFunc(indices, func(a, b int) int { return cmp.Compare(scores[b], scores[a]) })
	for _, frame := range indices[:min(count, len(indices))] {
		scores[frame] += amount
	}
}
