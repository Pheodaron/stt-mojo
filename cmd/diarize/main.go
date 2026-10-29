// diarize streams an already-decoded float32 file and writes frame logits.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"

	pb "github.com/Pheodaron/stt-mojo/api/diarization/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	address := flag.String("address", "127.0.0.1:50051", "gRPC server")
	input := flag.String("input", "", "mono 16 kHz float32-le PCM file (no header)")
	output := flag.String("output", "logits.f32", "output raw float32-le logits file")
	flag.Parse()
	if *input == "" {
		return fmt.Errorf("--input is required")
	}
	in, err := os.Open(*input)
	if err != nil {
		return err
	}
	defer in.Close()
	info, err := in.Stat()
	if err != nil {
		return err
	}
	if info.Size()%4 != 0 {
		return fmt.Errorf("PCM file contains an incomplete float32 sample")
	}
	if outputInfo, err := os.Stat(*output); err == nil && os.SameFile(info, outputInfo) {
		return fmt.Errorf("input and output must be different files")
	}
	out, err := os.Create(*output)
	if err != nil {
		return err
	}
	defer out.Close()
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	ctx, cancelStream := context.WithCancel(ctx)
	defer cancelStream()
	conn, err := grpc.NewClient(*address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return err
	}
	defer conn.Close()
	stream, err := pb.NewDiarizationClient(conn).Diarize(ctx)
	if err != nil {
		return err
	}
	sent := make(chan error, 1)
	// Send and receive concurrently: gRPC flow control applies in both directions.
	go func() {
		buffer := make([]byte, 16000*4)
		for {
			n, readErr := io.ReadFull(in, buffer)
			if n > 0 {
				if err := stream.Send(&pb.AudioChunk{PcmF32Le: append([]byte(nil), buffer[:n]...)}); err != nil {
					sent <- err
					return
				}
			}
			if readErr == io.EOF || readErr == io.ErrUnexpectedEOF {
				sent <- stream.CloseSend()
				return
			}
			if readErr != nil {
				sent <- readErr
				cancelStream()
				return
			}
		}
	}()
	var frames uint64
	var speakers uint32
	for {
		prediction, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if prediction.StartFrame != frames || len(prediction.LogitsF32Le) != int(prediction.NumFrames)*int(prediction.NumSpeakers)*4 {
			return fmt.Errorf("invalid server response shape or offset")
		}
		if speakers != 0 && speakers != prediction.NumSpeakers {
			return fmt.Errorf("speaker dimension changed")
		}
		speakers = prediction.NumSpeakers
		if _, err = out.Write(prediction.LogitsF32Le); err != nil {
			return err
		}
		frames += uint64(prediction.NumFrames)
	}
	if err := <-sent; err != nil {
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%d frames × %d speakers, 10 ms per frame → %s\n", frames, speakers, *output)
	return nil
}
