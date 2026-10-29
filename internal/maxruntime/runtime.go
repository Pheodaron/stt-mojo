// Package maxruntime calls the MAX C API directly. It never starts Python.
package maxruntime

/*
#include <stdlib.h>
#include "shim.h"
*/
import "C"

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"unsafe"

	"github.com/Pheodaron/stt-mojo/internal/diarization"
)

type Runtime struct {
	ptr      *C.NemotronMAX
	c        diarization.Config
	gate     chan struct{}
	cos, sin []float32
}

// Open creates the process's single MAX context and loads the compiled graphs.
// root is the Modular SDK package directory containing lib/.
func Open(bundle, root string, c diarization.Config, k diarization.Constants) (*Runtime, error) {
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
	r := &Runtime{ptr: ptr, c: c, gate: make(chan struct{}, 1)}
	// Positions restart at zero for each [speaker cache, FIFO, context, chunk].
	// These immutable tables replace trigonometry and allocations per step.
	maxFrames := c.CacheLength + c.FIFOLength + c.LeftContext + c.ChunkLength + c.RightContext
	r.cos = make([]float32, maxFrames*c.RotaryDim)
	r.sin = make([]float32, len(r.cos))
	for t := 0; t < maxFrames; t++ {
		for j, inv := range k.InvFreq {
			angle := float32(t) * inv
			s, co := math.Sincos(float64(angle))
			first := t*c.RotaryDim + j
			second := first + c.RotaryDim/2
			r.cos[first], r.cos[second] = float32(co), float32(co)
			r.sin[first], r.sin[second] = float32(s), float32(s)
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

// Close waits for active execution. Subsequent callers get an error.
func (r *Runtime) Close() {
	r.gate <- struct{}{}
	defer func() { <-r.gate }()
	if r.ptr != nil {
		C.nemotron_close(r.ptr)
		r.ptr = nil
	}
}

func (r *Runtime) acquire(ctx context.Context) error {
	select {
	case r.gate <- struct{}{}:
	case <-ctx.Done():
		return ctx.Err()
	}
	if err := ctx.Err(); err != nil {
		<-r.gate
		return err
	}
	if r.ptr == nil {
		<-r.gate
		return fmt.Errorf("MAX runtime is closed")
	}
	return nil
}

func pointer(v []float32) *C.float { return (*C.float)(unsafe.Pointer(&v[0])) }

func (r *Runtime) Embed(ctx context.Context, features []float32) ([]float32, error) {
	width := r.c.Factor * r.c.MelBins
	if len(features) == 0 || len(features)%width != 0 {
		return nil, fmt.Errorf("invalid feature shape")
	}
	if err := r.acquire(ctx); err != nil {
		return nil, err
	}
	defer func() { <-r.gate }()
	frames := len(features) / width
	output := make([]float32, frames*r.c.HiddenSize)
	var message *C.char
	ok := C.nemotron_embed(r.ptr, pointer(features), C.size_t(frames), C.size_t(width), C.size_t(r.c.HiddenSize), pointer(output), &message)
	runtime.KeepAlive(features)
	if ok == 0 {
		return nil, nativeError(message)
	}
	return output, ctx.Err()
}

func (r *Runtime) Infer(ctx context.Context, embeddings []float32, valid []bool) ([]float32, error) {
	frames := len(valid)
	if frames == 0 || len(embeddings) != frames*r.c.HiddenSize || frames*r.c.RotaryDim > len(r.cos) {
		return nil, fmt.Errorf("invalid encoder shape")
	}
	if err := r.acquire(ctx); err != nil {
		return nil, err
	}
	defer func() { <-r.gate }()
	mask := make([]float32, frames)
	hasValid := false
	for i, v := range valid {
		if v {
			hasValid = true
		} else {
			mask[i] = -math.MaxFloat32
		}
	}
	if !hasValid {
		return nil, fmt.Errorf("encoder needs at least one valid key")
	}
	output := make([]float32, frames*r.c.Factor*r.c.Speakers)
	var message *C.char
	ok := C.nemotron_infer(r.ptr, pointer(embeddings), pointer(r.cos), pointer(r.sin), pointer(mask), C.size_t(frames), C.size_t(r.c.HiddenSize), C.size_t(r.c.RotaryDim), C.size_t(len(output)), pointer(output), &message)
	runtime.KeepAlive(embeddings)
	runtime.KeepAlive(mask)
	if ok == 0 {
		return nil, nativeError(message)
	}
	return output, ctx.Err()
}
