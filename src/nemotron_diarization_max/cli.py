import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Nemotron 3 Diarization on MAX, CPU/float32")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download Transformers safetensors checkpoint")
    download.add_argument("--model", default="nvidia/Nemotron-3-Diarization")
    download.add_argument("--revision", required=True, help="Pin a model repository commit")
    download.add_argument("--output", required=True)
    serve = commands.add_parser("serve", help="Serve HTTP API using MAX inference graphs")
    serve.add_argument(
        "--model", required=True, help="Local checkpoint directory or Hugging Face repo ID"
    )
    serve.add_argument("--revision")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--max-audio-seconds", type=float, default=600)
    diarize = commands.add_parser("diarize", help="Diarize a local audio file")
    diarize.add_argument("audio")
    diarize.add_argument("--model", required=True)
    diarize.add_argument("--revision")
    diarize.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.command == "download":
        from huggingface_hub import snapshot_download

        snapshot_download(
            args.model,
            revision=args.revision,
            local_dir=args.output,
            allow_patterns=[
                "config.json",
                "processor_config.json",
                "*.safetensors",
                "*.safetensors.index.json",
            ],
        )
    elif args.command == "serve":
        import uvicorn

        from .service import create_app

        app = create_app(
            args.model,
            revision=args.revision,
            api_key=os.environ.get("DIARIZATION_API_KEY"),
            max_audio_seconds=args.max_audio_seconds,
        )
        uvicorn.run(app, host=args.host, port=args.port, workers=1)
    else:
        import soundfile as sf

        from .audio import FeatureExtractor, resample_mono, speaker_segments
        from .model import NemotronDiarization
        from .weights import resolve_checkpoint

        path = resolve_checkpoint(args.model, args.revision)
        processor = path / "processor_config.json"
        frontend = (
            FeatureExtractor.from_processor_config(json.loads(processor.read_text()))
            if processor.exists()
            else FeatureExtractor()
        )
        audio, rate = sf.read(args.audio, dtype="float32", always_2d=True)
        audio = resample_mono(audio, rate, frontend.sampling_rate)
        features = frontend(audio, sampling_rate=frontend.sampling_rate)
        model = NemotronDiarization.from_pretrained(path)
        result = model(features.input_features, features.attention_mask)
        print(
            json.dumps(
                {
                    "segments": speaker_segments(
                        result.probabilities[0],
                        threshold=args.threshold,
                        frame_seconds=frontend.hop_length / frontend.sampling_rate,
                        attention_mask=features.attention_mask[0],
                        duration=len(audio) / frontend.sampling_rate,
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
