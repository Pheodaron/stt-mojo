// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Port of NeMo Sortformer streaming_update and _compress_spkcache. See NOTICE.
package diarization

import (
	"math"
	"slices"
	"sort"
)

// speakerCache is model state for one recording, bounded by the exported policy.
type speakerCache struct {
	c                   Config
	embeds, probs, fifo []float32
	compressed          bool
}

func (s *speakerCache) update(inputs, logits []float32, valid []bool, lookback, frames int, silence []float32) {
	c := s.c
	nCache, nFIFO := len(s.embeds)/c.HiddenSize, len(s.fifo)/c.HiddenSize
	start := nCache + nFIFO + lookback
	s.fifo = append(s.fifo, inputs[start*c.HiddenSize:(start+frames)*c.HiddenSize]...)
	if nFIFO+frames <= c.FIFOLength {
		return
	}
	pop := min(max(c.UpdatePeriod, nFIFO+frames-c.FIFOLength), nFIFO+frames)
	// NeMo refreshes FIFO predictions every step and cache predictions until its
	// first compression. Pool frame probabilities, not logits, over subsampling.
	pooled := make([]float32, len(valid)*c.Speakers)
	for t, ok := range valid {
		if !ok {
			continue
		}
		for f := 0; f < c.Factor; f++ {
			for sp := 0; sp < c.Speakers; sp++ {
				x := logits[(t*c.Factor+f)*c.Speakers+sp]
				pooled[t*c.Speakers+sp] += sigmoid(x)
			}
		}
		for sp := 0; sp < c.Speakers; sp++ {
			pooled[t*c.Speakers+sp] /= float32(c.Factor)
		}
	}
	if !s.compressed {
		s.probs = append(s.probs[:0], pooled[:nCache*c.Speakers]...)
	}
	oldFIFOProbs := pooled[nCache*c.Speakers : (nCache+nFIFO)*c.Speakers]
	newProbs := pooled[start*c.Speakers : (start+frames)*c.Speakers]
	fifoProbs := append(slices.Clone(oldFIFOProbs), newProbs...)
	s.embeds = append(s.embeds, s.fifo[:pop*c.HiddenSize]...)
	s.probs = append(s.probs, fifoProbs[:pop*c.Speakers]...)
	s.fifo = slices.Clone(s.fifo[pop*c.HiddenSize:])
	if len(s.embeds)/c.HiddenSize > c.CacheLength {
		s.compress(silence)
		s.compressed = true
	}
}

func sigmoid(x float32) float32 {
	if x >= 0 {
		return float32(1 / (1 + math.Exp(-float64(x))))
	}
	e := math.Exp(float64(x))
	return float32(e / (1 + e))
}

// topIndices resolves ties by original index. torch.topk leaves tie order
// unspecified; choosing it explicitly makes CPU runs reproducible.
func topIndices(values []float32, n int) []int {
	indices := make([]int, len(values))
	for i := range indices {
		indices[i] = i
	}
	sort.SliceStable(indices, func(i, j int) bool { return values[indices[i]] > values[indices[j]] })
	return indices[:min(n, len(indices))]
}

func boostTop(scores []float32, count int, amount float32) {
	for _, i := range topIndices(scores, count) {
		scores[i] += amount
	}
}

func (s *speakerCache) compress(silence []float32) {
	c := s.c
	frames := len(s.probs) / c.Speakers
	scoredFrames := frames + c.SilenceFrames
	budget := c.CacheLength/c.Speakers - c.SilenceFrames
	// Speaker-major order is essential: the final sort preserves arrival order
	// within each speaker, rather than sorting all selected frames by time.
	scores := make([]float32, c.Speakers*scoredFrames)
	logComplements := make([]float32, c.Speakers)
	const logHalf = -math.Ln2
	for t := 0; t < frames; t++ {
		var complementSum float32
		for sp := 0; sp < c.Speakers; sp++ {
			p := s.probs[t*c.Speakers+sp]
			logComplements[sp] = float32(math.Log(float64(max(1-p, c.PredictionThreshold))))
			complementSum += logComplements[sp]
		}
		for sp := 0; sp < c.Speakers; sp++ {
			p := s.probs[t*c.Speakers+sp]
			logP := float32(math.Log(float64(max(p, c.PredictionThreshold))))
			score := logP - logComplements[sp] + complementSum - float32(logHalf)
			if p <= .5 {
				score = float32(math.Inf(-1))
			}
			scores[sp*scoredFrames+t] = score
		}
	}
	for sp := 0; sp < c.Speakers; sp++ {
		column := scores[sp*scoredFrames : sp*scoredFrames+frames]
		positive := 0
		for _, x := range column {
			if x > 0 {
				positive++
			}
		}
		if positive >= int(math.Floor(float64(budget)*c.MinPositiveRate)) {
			for i, x := range column {
				if x <= 0 {
					column[i] = float32(math.Inf(-1))
				}
			}
		}
		for i := c.CacheLength; i < frames; i++ {
			column[i] += c.LatestBoost
		}
		boostTop(column, int(math.Floor(float64(budget)*c.StrongBoost)), -2*float32(logHalf))
		boostTop(column, int(math.Floor(float64(budget)*c.WeakBoost)), -float32(logHalf))
		for i := frames; i < scoredFrames; i++ {
			scores[sp*scoredFrames+i] = float32(math.Inf(1))
		}
	}
	indices := topIndices(scores, c.CacheLength)
	sentinel := len(scores)
	for i, index := range indices {
		if math.IsInf(float64(scores[index]), -1) {
			indices[i] = sentinel
		}
	}
	slices.Sort(indices)
	embeds := make([]float32, c.CacheLength*c.HiddenSize)
	probs := make([]float32, c.CacheLength*c.Speakers)
	for i, index := range indices {
		t := index % scoredFrames
		if index == sentinel || t >= frames {
			copy(embeds[i*c.HiddenSize:], silence)
			continue
		}
		copy(embeds[i*c.HiddenSize:], s.embeds[t*c.HiddenSize:(t+1)*c.HiddenSize])
		copy(probs[i*c.Speakers:], s.probs[t*c.Speakers:(t+1)*c.Speakers])
	}
	s.embeds, s.probs = embeds, probs
}
