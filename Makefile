GO ?= go
GO_TEST_FLAGS ?=
PYTHON ?= .venv/bin/python
MAX_SITE ?= $(shell $(PYTHON) -c 'import site; print(site.getsitepackages()[0])')
MAX_ROOT ?= $(MAX_SITE)/modular
export CGO_CFLAGS := -I$(MAX_SITE)/max/include
export CGO_LDFLAGS := -L$(MAX_ROOT)/lib -Wl,-rpath,$(MAX_ROOT)/lib -lmax
export MAX_ROOT

.PHONY: build test proto
build:
	$(GO) build -o bin/nemotron-server ./cmd/server
	$(GO) build -o bin/diarize ./cmd/diarize

test:
	$(GO) test $(GO_TEST_FLAGS) ./...

proto:
	$(PYTHON) -m grpc_tools.protoc -I . --go_out=. --go_opt=module=github.com/Pheodaron/stt-mojo --go-grpc_out=. --go-grpc_opt=module=github.com/Pheodaron/stt-mojo api/diarization/v1/diarization.proto
