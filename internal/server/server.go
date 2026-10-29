package server

import (
	"context"
	"errors"
	"io"
	"log/slog"

	pb "github.com/Pheodaron/stt-mojo/api/diarization/v1"
	"github.com/Pheodaron/stt-mojo/internal/diarization"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Service has one shared model, with no registry of recordings or cached results.
type Service struct {
	pb.UnimplementedDiarizationServer
	Config    diarization.Config
	Constants diarization.Constants
	Engine    diarization.Engine
}

func (s *Service) Diarize(stream grpc.BidiStreamingServer[pb.AudioChunk, pb.Prediction]) error {
	pipeline := diarization.NewPipeline(s.Config, s.Constants, s.Engine)
	emit := func(result diarization.Result) error {
		return stream.Send(&pb.Prediction{
			StartFrame:  result.StartFrame,
			NumFrames:   uint32(len(result.Logits) / s.Config.Speakers),
			NumSpeakers: uint32(s.Config.Speakers),
			LogitsF32Le: diarization.EncodeFloat32(result.Logits),
		})
	}
	var samples int64
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			if samples < int64(s.Config.HopLength) {
				return status.Error(codes.InvalidArgument, "audio must contain at least 160 samples")
			}
			return inferenceError(pipeline.Finish(stream.Context(), emit))
		}
		if err != nil {
			return err
		}
		audio, err := diarization.DecodePCM(chunk.PcmF32Le)
		if err != nil {
			return status.Error(codes.InvalidArgument, err.Error())
		}
		samples += int64(len(audio))
		if err = pipeline.Push(stream.Context(), audio, emit); err != nil {
			return inferenceError(err)
		}
	}
}

func inferenceError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return status.FromContextError(err).Err()
	}
	if _, ok := status.FromError(err); ok {
		return err
	}
	slog.Error("diarization failed", "error", err)
	return status.Error(codes.Internal, "model inference failed")
}
