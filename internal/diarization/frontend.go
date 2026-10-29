package diarization

import (
	"math"

	"gonum.org/v1/gonum/dsp/fourier"
)

// frontend keeps one STFT window. Transport boundaries never reset preemphasis
// or window overlap. Its output matches the centered, unnormalized HF frontend.
type frontend struct {
	c               Config
	k               Constants
	buffer          []float32
	previous        float32
	samples, frames int64
	fft             *fourier.FFT
	windowed        []float64
	spectrum        []complex128
	filters         []melFilter
	power, mel      []float32
}

type melFilter struct {
	start   int
	weights []float32
}

func newFrontend(c Config, k Constants) *frontend {
	f := &frontend{
		c: c, k: k,
		buffer:   make([]float32, c.NFFT/2, c.NFFT),
		fft:      fourier.NewFFT(c.NFFT),
		windowed: make([]float64, c.NFFT),
		spectrum: make([]complex128, c.NFFT/2+1),
		filters:  make([]melFilter, c.MelBins),
		power:    make([]float32, c.NFFT/2+1),
		mel:      make([]float32, c.MelBins),
	}
	// Triangular mel filters are sparse. Ignore zero regions rather than doing
	// 128*257 multiplications for every 10 ms frame.
	for m := range f.filters {
		row := k.MelFilters[m*len(f.power) : (m+1)*len(f.power)]
		first, last := 0, len(row)
		for first < last && row[first] == 0 {
			first++
		}
		for last > first && row[last-1] == 0 {
			last--
		}
		f.filters[m] = melFilter{start: first, weights: row[first:last]}
	}
	return f
}

func (f *frontend) push(audio []float32, emit func([]float32, bool) error) error {
	for _, x := range audio {
		y := x
		if f.samples > 0 && f.c.Preemphasis != nil {
			y -= *f.c.Preemphasis * f.previous
		}
		f.previous = x
		f.samples++
		f.buffer = append(f.buffer, y)
		if len(f.buffer) == f.c.NFFT {
			if err := f.frame(true, emit); err != nil {
				return err
			}
		}
	}
	return nil
}

func (f *frontend) finish(emit func([]float32, bool) error) error {
	valid := f.samples / int64(f.c.HopLength)
	// Centered STFT has one extra frame; its features and mask are zero.
	for f.frames <= valid {
		used := len(f.buffer)
		f.buffer = f.buffer[:f.c.NFFT]
		clear(f.buffer[used:])
		if err := f.frame(f.frames < valid, emit); err != nil {
			return err
		}
	}
	return nil
}

func (f *frontend) frame(valid bool, emit func([]float32, bool) error) error {
	if valid {
		for i, x := range f.buffer {
			f.windowed[i] = float64(x * f.k.Window[i])
		}
		f.fft.Coefficients(f.spectrum, f.windowed)
		for i := range f.power {
			// Round at the same float32 magnitude -> square boundaries as upstream.
			r, im := float32(real(f.spectrum[i])), float32(imag(f.spectrum[i]))
			mag := float32(math.Sqrt(float64(r*r + im*im)))
			f.power[i] = mag * mag
		}
		for m, filter := range f.filters {
			var sum float32
			for i, weight := range filter.weights {
				sum += f.power[filter.start+i] * weight
			}
			f.mel[m] = float32(math.Log(float64(sum + float32(1.0/(1<<24)))))
		}
	} else {
		clear(f.mel)
	}
	f.frames++
	if err := emit(f.mel, valid); err != nil {
		return err
	}
	n := copy(f.buffer, f.buffer[f.c.HopLength:])
	f.buffer = f.buffer[:n]
	return nil
}
