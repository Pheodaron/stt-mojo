import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Stateless Nemotron diarization on MAX CPU")
    parser.add_argument(
        "--model", required=True, type=Path, help="Local safetensors checkpoint directory"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    from .service import create_app

    uvicorn.run(create_app(args.model), host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
