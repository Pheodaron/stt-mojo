package main

import (
	"flag"
	"log"
	"net"
	"os"

	pb "github.com/Pheodaron/stt-mojo/api/diarization/v1"
	"github.com/Pheodaron/stt-mojo/internal/diarization"
	"github.com/Pheodaron/stt-mojo/internal/maxruntime"
	"github.com/Pheodaron/stt-mojo/internal/server"
	"google.golang.org/grpc"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	address := flag.String("listen", "127.0.0.1:50051", "gRPC listen address")
	bundle := flag.String("model", "models/nemotron-cpu", "exported MAX bundle")
	root := flag.String("max-root", os.Getenv("MAX_ROOT"), "Modular SDK directory")
	flag.Parse()

	config, constants, err := diarization.LoadBundle(*bundle)
	if err != nil {
		return err
	}
	model, err := maxruntime.Open(*bundle, *root, config, constants)
	if err != nil {
		return err
	}
	defer model.Close()

	listener, err := net.Listen("tcp", *address)
	if err != nil {
		return err
	}
	defer listener.Close()

	service := grpc.NewServer(grpc.MaxRecvMsgSize(1 << 20))
	pb.RegisterDiarizationServer(service, &server.Service{
		Pipeline: diarization.New(config, constants, model), Speakers: config.Speakers,
	})
	log.Printf("ready: %s (MAX CPU, one recording at a time)", *address)
	return service.Serve(listener)
}
