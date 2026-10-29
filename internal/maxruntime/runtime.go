// Package maxruntime calls the MAX C API directly. It never starts Python.
package maxruntime

/*
#include <stdlib.h>
#include "shim.h"
*/
import "C"

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"unsafe"

	"github.com/Pheodaron/stt-mojo/internal/diarization"
)

type Runtime struct {
	handle         *C.NemotronMAX
	config         diarization.Config
	cosines, sines []float32
}

// Open creates the process's single MAX context and loads the compiled graphs.
// root is the Modular SDK package directory containing lib/.
func Open(bundle, root string, config diarization.Config, constants diarization.Constants) (*Runtime, error) {
	if root == "" {
		return nil, fmt.Errorf("set --max-root or MAX_ROOT to the Modular SDK directory")
	}
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	for name, value := range map[string]string{
		"MODULAR_MAX_PACKAGE_ROOT":         root,
		"MODULAR_MAX_LIB_PATH":             filepath.Join(root, "lib"),
		"MODULAR_MOJO_MAX_COMPILERRT_PATH": filepath.Join(root, "lib/libKGENCompilerRTShared.so"),
		"MODULAR_MOJO_MAX_MGPRT_PATH":      filepath.Join(root, "lib/libMGPRT.so"),
	} {
		if os.Getenv(name) == "" {
			if err = os.Setenv(name, value); err != nil {
				return nil, err
			}
		}
	}
	dir := C.CString(bundle)
	defer C.free(unsafe.Pointer(dir))
	var message *C.char
	ptr := C.nemotron_open(dir, &message)
	if ptr == nil {
		return nil, nativeError(message)
	}
	r := &Runtime{handle: ptr, config: config}
	// Positions restart at zero for each [speaker cache, FIFO, context, chunk].
	// These immutable tables replace trigonometry and allocations per step.
	maxFrames := config.CacheLength + config.FIFOLength + config.LeftContext + config.ChunkLength + config.RightContext
	r.cosines = make([]float32, maxFrames*config.RotaryDim)
	r.sines = make([]float32, len(r.cosines))
	for t := 0; t < maxFrames; t++ {
		for j, inv := range constants.InvFreq {
			angle := float32(t) * inv
			s, co := math.Sincos(float64(angle))
			first := t*config.RotaryDim + j
			second := first + config.RotaryDim/2
			r.cosines[first], r.cosines[second] = float32(co), float32(co)
			r.sines[first], r.sines[second] = float32(s), float32(s)
		}
	}
	return r, nil
}

func nativeError(message *C.char) error {
	if message == nil {
		return fmt.Errorf("MAX operation failed")
	}
	defer C.free(unsafe.Pointer(message))
	return fmt.Errorf("MAX: %s", C.GoString(message))
}

// Close releases both graphs and their weights. The caller finishes Run first.
func (r *Runtime) Close() {
	if r.handle != nil {
		C.nemotron_close(r.handle)
		r.handle = nil
	}
}

func floatPointer(values []float32) *C.float { return (*C.float)(unsafe.Pointer(&values[0])) }

func (r *Runtime) Embed(features []float32) ([]float32, error) {
	width := r.config.Factor * r.config.MelBins
	if len(features) == 0 || len(features)%width != 0 {
		return nil, fmt.Errorf("invalid feature shape")
	}
	if r.handle == nil {
		return nil, fmt.Errorf("MAX runtime is closed")
	}
	frames := len(features) / width
	output := make([]float32, frames*r.config.HiddenSize)
	var message *C.char
	ok := C.nemotron_embed(
		r.handle, floatPointer(features),
		C.size_t(frames), C.size_t(width), C.size_t(r.config.HiddenSize),
		floatPointer(output), &message,
	)
	runtime.KeepAlive(features)
	if ok == 0 {
		return nil, nativeError(message)
	}
	return output, nil
}

func (r *Runtime) Predict(embeddings []float32, validFrames int) ([]float32, error) {
	frames := len(embeddings) / r.config.HiddenSize
	if frames == 0 || len(embeddings) != frames*r.config.HiddenSize || frames*r.config.RotaryDim > len(r.cosines) {
		return nil, fmt.Errorf("invalid encoder shape")
	}
	if r.handle == nil {
		return nil, fmt.Errorf("MAX runtime is closed")
	}
	mask := make([]float32, frames)
	if validFrames < 1 || validFrames > frames {
		return nil, fmt.Errorf("invalid encoder valid length")
	}
	for i := validFrames; i < frames; i++ {
		mask[i] = -math.MaxFloat32
	}
	output := make([]float32, frames*r.config.Factor*r.config.Speakers)
	var message *C.char
	ok := C.nemotron_infer(
		r.handle, floatPointer(embeddings), floatPointer(r.cosines), floatPointer(r.sines), floatPointer(mask),
		C.size_t(frames), C.size_t(r.config.HiddenSize), C.size_t(r.config.RotaryDim),
		C.size_t(len(output)), floatPointer(output), &message,
	)
	runtime.KeepAlive(embeddings)
	runtime.KeepAlive(mask)
	if ok == 0 {
		return nil, nativeError(message)
	}
	return output, nil
}
