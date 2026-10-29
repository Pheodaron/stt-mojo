package server

import (
	"context"
	"errors"
	"fmt"

	pb "github.com/Pheodaron/stt-mojo/api/diarization/v1"
	"github.com/Pheodaron/stt-mojo/internal/diarization"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Service assumes one active recording. The client starts the next RPC after EOF.
type Service struct {
	pb.UnimplementedDiarizationServer
	Pipeline *diarization.Pipeline
	Speakers int
}

func (s *Service) Diarize(stream grpc.BidiStreamingServer[pb.AudioChunk, pb.Prediction]) error {
	err := s.Pipeline.Run(stream.Context(), &audioReader{stream: stream}, func(result diarization.Result) error {
		return stream.Send(&pb.Prediction{
			StartFrame:  result.StartFrame,
			NumFrames:   uint32(len(result.Logits) / s.Speakers),
			NumSpeakers: uint32(s.Speakers),
			LogitsF32Le: diarization.EncodeFloat32(result.Logits),
		})
	})
	if errors.Is(err, diarization.ErrInvalidAudio) {
		return status.Error(codes.InvalidArgument, err.Error())
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return status.FromContextError(err).Err()
	}
	if err != nil {
		if _, ok := status.FromError(err); ok {
			return err
		}
		return status.Error(codes.Internal, err.Error())
	}
	return nil
}

// audioReader adapts gRPC messages to an ordinary PCM byte stream. It contains
// no inference logic; file and network input use the same pipeline.
type audioReader struct {
	stream  grpc.BidiStreamingServer[pb.AudioChunk, pb.Prediction]
	pending []byte
}

func (r *audioReader) Read(dst []byte) (int, error) {
	if len(dst) == 0 {
		return 0, nil
	}
	for len(r.pending) == 0 {
		message, err := r.stream.Recv()
		if err != nil {
			return 0, err
		}
		if len(message.PcmF32Le)%4 != 0 {
			return 0, fmt.Errorf("%w: message contains an incomplete sample", diarization.ErrInvalidAudio)
		}
		r.pending = message.PcmF32Le
	}
	n := copy(dst, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}
