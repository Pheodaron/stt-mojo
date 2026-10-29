package server

import (
	"context"
	"fmt"
	"io"
	"math"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	pb "github.com/Pheodaron/stt-mojo/api/diarization/v1"
	"github.com/Pheodaron/stt-mojo/internal/diarization"
	"github.com/Pheodaron/stt-mojo/internal/maxruntime"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

func TestNativeMAXGRPCAgainstUpstream(t *testing.T) {
	dir := os.Getenv("NEMOTRON_TEST_BUNDLE")
	if dir == "" {
		t.Skip("set NEMOTRON_TEST_BUNDLE to run native CPU integration")
	}
	c, k, err := diarization.LoadBundle(dir)
	if err != nil {
		t.Fatal(err)
	}
	engine, err := maxruntime.Open(dir, os.Getenv("MAX_ROOT"), c, k)
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	listener := bufconn.Listen(1 << 20)
	service := grpc.NewServer(grpc.MaxRecvMsgSize(1 << 20))
	pb.RegisterDiarizationServer(service, &Service{Config: c, Constants: k, Engine: engine})
	go func() { _ = service.Serve(listener) }()
	defer service.Stop()
	conn, err := grpc.NewClient("passthrough:///test", grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	client := pb.NewDiarizationClient(conn)
	read := func(name string) []byte {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	for _, count := range []int{160, 161, 319, 320, 511, 512, 959, 960, 1120, 1280, 1600, 1607, 12017} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			audio := read(fmt.Sprintf("audio-%d.f32", count))
			expected, err := diarization.DecodePCM(read(fmt.Sprintf("logits-%d.f32", count)))
			if err != nil {
				t.Fatal(err)
			}
			for _, chunkSize := range []int{68, 4096} {
				ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
				defer cancel()
				stream, err := client.Diarize(ctx)
				if err != nil {
					t.Fatal(err)
				}
				sent := make(chan error, 1)
				go func() {
					for start := 0; start < len(audio); start += chunkSize {
						if err := stream.Send(&pb.AudioChunk{PcmF32Le: audio[start:min(start+chunkSize, len(audio))]}); err != nil {
							sent <- err
							return
						}
					}
					sent <- stream.CloseSend()
				}()
				var actual []float32
				for {
					r, err := stream.Recv()
					if err == io.EOF {
						break
					}
					if err != nil {
						t.Fatal(err)
					}
					if r.StartFrame != uint64(len(actual)/c.Speakers) || r.NumSpeakers != uint32(c.Speakers) {
						t.Fatal("invalid result offset or shape")
					}
					v, err := diarization.DecodePCM(r.LogitsF32Le)
					if err != nil {
						t.Fatal(err)
					}
					if len(v) != int(r.NumFrames)*c.Speakers {
						t.Fatal("invalid frame count")
					}
					actual = append(actual, v...)
				}
				if err := <-sent; err != nil {
					t.Fatal(err)
				}
				if len(actual) != len(expected) {
					t.Fatalf("expected %d logits, got %d", len(expected), len(actual))
				}
				var maxError float64
				for i, x := range actual {
					maxError = max(maxError, math.Abs(float64(x-expected[i])))
				}
				if maxError > 3e-5 {
					t.Fatalf("native gRPC max error %.8g", maxError)
				}
				t.Logf("samples=%d transport_bytes=%d max logit error=%.8g", count, chunkSize, maxError)
			}
		})
	}
	for _, bad := range [][]byte{nil, {1}, diarization.EncodeFloat32([]float32{float32(math.NaN())})} {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		stream, err := client.Diarize(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if len(bad) > 0 {
			_ = stream.Send(&pb.AudioChunk{PcmF32Le: bad})
		}
		_ = stream.CloseSend()
		_, err = stream.Recv()
		cancel()
		if status.Code(err) != codes.InvalidArgument {
			t.Fatalf("expected InvalidArgument, got %v", err)
		}
	}
	t.Run("independent concurrent recordings", func(t *testing.T) {
		audio := read("audio-12017.f32")
		expected, err := diarization.DecodePCM(read("logits-12017.f32"))
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 8; i++ {
			t.Run(fmt.Sprint(i), func(t *testing.T) {
				t.Parallel()
				ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
				defer cancel()
				stream, err := client.Diarize(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if err = stream.Send(&pb.AudioChunk{PcmF32Le: audio}); err != nil {
					t.Fatal(err)
				}
				if err = stream.CloseSend(); err != nil {
					t.Fatal(err)
				}
				var actual []float32
				for {
					r, err := stream.Recv()
					if err == io.EOF {
						break
					}
					if err != nil {
						t.Fatal(err)
					}
					values, err := diarization.DecodePCM(r.LogitsF32Le)
					if err != nil {
						t.Fatal(err)
					}
					actual = append(actual, values...)
				}
				if len(actual) != len(expected) {
					t.Fatal("concurrent result has wrong shape")
				}
				for j, x := range actual {
					if math.Abs(float64(x-expected[j])) > 3e-5 {
						t.Fatal("recording state leaked between RPCs")
					}
				}
			})
		}
	})
}
