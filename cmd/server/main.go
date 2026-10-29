package main

import (
	"flag"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	pb "github.com/Pheodaron/stt-mojo/api/diarization/v1"
	"github.com/Pheodaron/stt-mojo/internal/diarization"
	"github.com/Pheodaron/stt-mojo/internal/maxruntime"
	"github.com/Pheodaron/stt-mojo/internal/server"
	"google.golang.org/grpc"
)

func main() {
	if err := run(); err != nil {
		slog.Error("server stopped", "error", err)
		os.Exit(1)
	}
}

func run() error {
	address := flag.String("listen", "127.0.0.1:50051", "gRPC listen address")
	bundle := flag.String("model", "models/nemotron-cpu", "exported MAX bundle directory")
	root := flag.String("max-root", os.Getenv("MAX_ROOT"), "Modular SDK directory containing lib/")
	flag.Parse()
	c, k, err := diarization.LoadBundle(*bundle)
	if err != nil {
		return err
	}
	engine, err := maxruntime.Open(*bundle, *root, c, k)
	if err != nil {
		return err
	}
	defer engine.Close()
	listener, err := net.Listen("tcp", *address)
	if err != nil {
		return err
	}
	defer listener.Close()
	// Input messages are bounded; the total recording length is unrestricted.
	service := grpc.NewServer(grpc.MaxRecvMsgSize(1<<20), grpc.MaxConcurrentStreams(8))
	pb.RegisterDiarizationServer(service, &server.Service{Config: c, Constants: k, Engine: engine})
	stopped := make(chan os.Signal, 1)
	signal.Notify(stopped, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(stopped)
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-done:
			return
		case <-stopped:
		}
		// Give in-flight work time to finish, then cancel remaining streams.
		timer := time.AfterFunc(5*time.Second, service.Stop)
		service.GracefulStop()
		timer.Stop()
	}()
	slog.Info("ready", "address", *address, "backend", "MAX C API CPU", "chunk_seconds", float64(c.ChunkLength*c.Factor*c.HopLength)/float64(c.SampleRate))
	return service.Serve(listener)
}
