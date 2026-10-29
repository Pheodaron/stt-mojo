package diarization

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"gonum.org/v1/gonum/dsp/fourier"
)

// featureReader reads centered log-mel frames without buffering the recording.
// The window, preemphasis history and FFT workspace belong to this reader.
type featureReader struct {
	config  Config
	audio   *bufio.Reader
	window  []float32
	filters []melFilter
	fft     *fourier.FFT

	raw      []byte
	samples  []float32
	windowed []float64
	spectrum []complex128
	power    []float32

	previous    float32
	sampleCount int64
	frameCount  int64
	fillFrom    int
	eof         bool
}

type melFilter struct {
	start   int
	weights []float32
}

func newFeatureReader(audio io.Reader, config Config, constants Constants) *featureReader {
	r := &featureReader{
		config:   config,
		audio:    bufio.NewReaderSize(audio, 64<<10),
		window:   constants.Window,
		filters:  make([]melFilter, config.MelBins),
		fft:      fourier.NewFFT(config.NFFT),
		raw:      make([]byte, config.NFFT*4),
		samples:  make([]float32, config.NFFT),
		windowed: make([]float64, config.NFFT),
		spectrum: make([]complex128, config.NFFT/2+1),
		power:    make([]float32, config.NFFT/2+1),
		fillFrom: config.NFFT / 2,
	}
	// Slaney filters are triangular: skip their zero regions.
	for mel := range r.filters {
		row := constants.MelFilters[mel*len(r.power) : (mel+1)*len(r.power)]
		first, last := 0, len(row)
		for first < last && row[first] == 0 {
			first++
		}
		for last > first && row[last-1] == 0 {
			last--
		}
		r.filters[mel] = melFilter{start: first, weights: row[first:last]}
	}
	return r
}

// readFrames fills whole mel frames in dst. valid excludes the final STFT padding
// frame. Like io.Reader, it may return both data and io.EOF.
func (r *featureReader) readFrames(ctx context.Context, dst []float32) (frames, valid int, err error) {
	for len(dst) >= r.config.MelBins {
		if err = ctx.Err(); err != nil {
			return frames, valid, err
		}
		var ok bool
		ok, err = r.readFrame(dst[:r.config.MelBins])
		if err != nil {
			return frames, valid, err
		}
		frames++
		if ok {
			valid++
		}
		dst = dst[r.config.MelBins:]
	}
	return frames, valid, nil
}

func (r *featureReader) readFrame(mel []float32) (valid bool, err error) {
	if !r.eof {
		if err = r.readAudio(r.samples[r.fillFrom:]); err != nil {
			return false, err
		}
	} else {
		clear(r.samples[r.fillFrom:])
	}
	validFrames := r.sampleCount / int64(r.config.HopLength)
	if r.eof && validFrames == 0 {
		return false, fmt.Errorf("%w: at least %d samples are required", ErrInvalidAudio, r.config.HopLength)
	}
	if r.eof && r.frameCount > validFrames {
		return false, io.EOF
	}

	valid = r.frameCount < validFrames
	if valid {
		r.logMel(mel)
	} else {
		clear(mel)
	}
	r.frameCount++
	r.fillFrom = copy(r.samples, r.samples[r.config.HopLength:])
	return valid, nil
}

func (r *featureReader) readAudio(dst []float32) error {
	data := r.raw[:len(dst)*4]
	n, err := io.ReadFull(r.audio, data)
	if err != nil && err != io.EOF && err != io.ErrUnexpectedEOF {
		return err
	}
	if n%4 != 0 {
		return fmt.Errorf("%w: incomplete float32 sample", ErrInvalidAudio)
	}
	r.eof = err != nil
	for i := 0; i < n/4; i++ {
		sample := math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			return fmt.Errorf("%w: non-finite sample", ErrInvalidAudio)
		}
		value := sample
		if r.sampleCount > 0 && r.config.Preemphasis != nil {
			value -= *r.config.Preemphasis * r.previous
		}
		dst[i] = value
		r.previous = sample
		r.sampleCount++
	}
	clear(dst[n/4:])
	return nil
}

func (r *featureReader) logMel(dst []float32) {
	for i, sample := range r.samples {
		r.windowed[i] = float64(sample * r.window[i])
	}
	r.fft.Coefficients(r.spectrum, r.windowed)
	for i, value := range r.spectrum {
		realPart, imagPart := float32(real(value)), float32(imag(value))
		// Preserve the float32 magnitude -> square rounding of the checkpoint.
		magnitude := float32(math.Sqrt(float64(realPart*realPart + imagPart*imagPart)))
		r.power[i] = magnitude * magnitude
	}
	for mel, filter := range r.filters {
		var energy float32
		for i, weight := range filter.weights {
			energy += r.power[filter.start+i] * weight
		}
		dst[mel] = float32(math.Log(float64(energy + float32(1.0/(1<<24)))))
	}
}
