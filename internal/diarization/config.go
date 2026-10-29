// Package diarization implements the Nemotron/NeMo long-recording pipeline.
package diarization

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// Config describes an exported CPU bundle, not a per-request tuning API.
type Config struct {
	Version             int      `json:"version"`
	MAXVersion          string   `json:"max_version"`
	Target              string   `json:"target"`
	SampleRate          int      `json:"sample_rate"`
	HopLength           int      `json:"hop_length"`
	NFFT                int      `json:"n_fft"`
	Preemphasis         *float32 `json:"preemphasis"`
	MelBins             int      `json:"mel_bins"`
	HiddenSize          int      `json:"hidden_size"`
	Factor              int      `json:"subsampling_factor"`
	RotaryDim           int      `json:"rotary_dim"`
	Speakers            int      `json:"num_speakers"`
	ChunkLength         int      `json:"chunk_length"`
	RightContext        int      `json:"right_context"`
	LeftContext         int      `json:"left_context"`
	FIFOLength          int      `json:"fifo_length"`
	UpdatePeriod        int      `json:"update_period"`
	CacheLength         int      `json:"cache_length"`
	SilenceFrames       int      `json:"silence_frames"`
	PredictionThreshold float32  `json:"prediction_threshold"`
	LatestBoost         float32  `json:"latest_boost"`
	StrongBoost         float64  `json:"strong_boost"`
	WeakBoost           float64  `json:"weak_boost"`
	MinPositiveRate     float64  `json:"min_positive_rate"`
}

type Constants struct {
	Window, MelFilters, InvFreq, Silence []float32
}

func LoadBundle(dir string) (Config, Constants, error) {
	var c Config
	var k Constants
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return c, k, err
	}
	if err = json.Unmarshal(data, &c); err != nil {
		return c, k, err
	}
	if err = c.Validate(); err != nil {
		return c, k, err
	}
	data, err = os.ReadFile(filepath.Join(dir, "constants.f32"))
	if err != nil {
		return c, k, err
	}
	n := c.NFFT + c.MelBins*(c.NFFT/2+1) + c.RotaryDim/2 + c.HiddenSize
	if len(data) != n*4 {
		return c, k, fmt.Errorf("constants.f32: expected %d bytes, got %d", n*4, len(data))
	}
	v, err := DecodePCM(data)
	if err != nil {
		return c, k, fmt.Errorf("constants: %w", err)
	}
	k.Window = v[:c.NFFT]
	v = v[c.NFFT:]
	n = c.MelBins * (c.NFFT/2 + 1)
	k.MelFilters, v = v[:n], v[n:]
	k.InvFreq, k.Silence = v[:c.RotaryDim/2], v[c.RotaryDim/2:]
	return c, k, nil
}

func (c Config) Validate() error {
	if c.Version != 1 || c.MAXVersion != "26.7.0" || c.Target != "cpu" {
		return fmt.Errorf("expected bundle v1 for MAX 26.7.0 CPU")
	}
	if c.SampleRate != 16000 || c.HopLength != 160 ||
		c.NFFT < 2*c.HopLength || c.NFFT > 8192 || c.NFFT&(c.NFFT-1) != 0 {
		return fmt.Errorf("unsupported PCM/frontend configuration")
	}
	dimensions := []int{
		c.MelBins, c.HiddenSize, c.Factor, c.RotaryDim, c.Speakers,
		c.ChunkLength, c.FIFOLength, c.UpdatePeriod, c.CacheLength,
	}
	for _, v := range dimensions {
		if v < 1 || v > 16384 {
			return fmt.Errorf("invalid model dimension %d", v)
		}
	}
	if c.RightContext < 0 || c.LeftContext < 0 || c.RightContext > 16384 || c.LeftContext > 16384 {
		return fmt.Errorf("invalid context lengths")
	}
	if c.SilenceFrames < 0 || c.RotaryDim%2 != 0 || c.CacheLength < (1+c.SilenceFrames)*c.Speakers {
		return fmt.Errorf("invalid contexts or speaker cache dimensions")
	}
	if c.PredictionThreshold <= 0 || c.PredictionThreshold > .5 ||
		c.StrongBoost < 0 || c.WeakBoost < 0 || c.MinPositiveRate < 0 {
		return fmt.Errorf("invalid speaker cache policy")
	}
	return nil
}

// DecodePCM accepts whole, finite IEEE float32 samples, without a file header.
func DecodePCM(data []byte) ([]float32, error) {
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("PCM byte length must be divisible by 4")
	}
	v := make([]float32, len(data)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		if math.IsNaN(float64(v[i])) || math.IsInf(float64(v[i]), 0) {
			return nil, fmt.Errorf("PCM contains a non-finite sample")
		}
	}
	return v, nil
}

func EncodeFloat32(v []float32) []byte {
	data := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(x))
	}
	return data
}
