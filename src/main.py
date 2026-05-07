"""
Entry point for the Interactive Oral Exam system.

Single-file mode (original, unchanged):
    python -m src.main audio.mp3 "What is gradient descent?"
    python -m src.main https://drive.google.com/... "Explain covariance"

Batch pipeline mode:
    python -m src.main --pipeline examples/batch_input_example.json --output output/
    python -m src.main --pipeline examples/batch_input_example.json --output output/ \\
        --methods direct_scoring rubric \\
        --rubric examples/rubric_example.json

Ingest-only (transcribe + store, no evaluation):
    python -m src.main --ingest examples/batch_input_example.json --output output/

Evaluate stored transcripts:
    python -m src.main --evaluate output/exam_001_transcripts.json \\
        --method direct_scoring --output output/

For a richer CLI with all options see: python scripts/run_batch.py --help
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.audio.loader import download_gdrive_audio, is_gdrive_url
from src.orchestration.single_turn import SingleTurnExam


def _run_single(audio_arg: str, question: str) -> None:
    """Original single-file flow — unchanged."""
    tmp_path = None
    if is_gdrive_url(audio_arg):
        print("Downloading audio from Google Drive...")
        tmp_path = download_gdrive_audio(audio_arg)
        audio_path = tmp_path
    else:
        audio_path = audio_arg

    try:
        exam = SingleTurnExam()
        result = exam.run(audio_path, question)
        print(f"Score:    {result.score}/100")
        print(f"Label:    {result.label}")
        print(f"Feedback: {result.feedback}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _run_pipeline(args: argparse.Namespace) -> None:
    from src.orchestration.batch_pipeline import BatchPipeline

    rubric = json.load(open(args.rubric)) if args.rubric else None
    anchors = json.load(open(args.anchors)) if args.anchors else None
    methods = args.methods or None

    pipeline = BatchPipeline(
        output_dir=args.output,
        model_name=args.model,
        pairwise_batch_size=getattr(args, "pairwise_batch_size", None),
        pairwise_shuffle_seed=getattr(args, "pairwise_shuffle_seed", 42),
        tournament_mode=getattr(args, "tournament_mode", "group_ranking"),
        tournament_batch_size=getattr(args, "tournament_batch_size", None),
        tournament_group_size=getattr(args, "tournament_group_size", 10),
        tournament_shuffle_seed=getattr(args, "tournament_shuffle_seed", 42),
        tournament_sample_k=getattr(args, "tournament_sample_k", None),
    )
    results = pipeline.run_pipeline(
        input_path=args.pipeline,
        methods=methods,
        expected_key_points=args.key_points or "Not provided",
        rubric=rubric,
        anchors=anchors,
    )
    print("\nPipeline complete:")
    for method, path in results.items():
        print(f"  {method:20s} → {path}")


def _run_ingest(args: argparse.Namespace) -> None:
    from src.orchestration.batch_pipeline import BatchPipeline

    pipeline = BatchPipeline(output_dir=args.output)
    records = pipeline.ingest(args.ingest)
    ok = sum(1 for r in records if not r.error)
    print(f"Ingestion complete: {ok}/{len(records)} students transcribed")
    print(f"Transcripts saved to: {args.output}/transcripts/")


def _run_evaluate(args: argparse.Namespace) -> None:
    from src.orchestration.batch_pipeline import BatchPipeline

    rubric = json.load(open(args.rubric)) if args.rubric else None
    anchors = json.load(open(args.anchors)) if args.anchors else None

    pipeline = BatchPipeline(
        output_dir=args.output,
        model_name=args.model,
        pairwise_batch_size=getattr(args, "pairwise_batch_size", None),
        pairwise_shuffle_seed=getattr(args, "pairwise_shuffle_seed", 42),
        tournament_mode=getattr(args, "tournament_mode", "group_ranking"),
        tournament_batch_size=getattr(args, "tournament_batch_size", None),
        tournament_group_size=getattr(args, "tournament_group_size", 10),
        tournament_shuffle_seed=getattr(args, "tournament_shuffle_seed", 42),
        tournament_sample_k=getattr(args, "tournament_sample_k", None),
    )
    path = pipeline.evaluate(
        args.evaluate,
        args.method,
        expected_key_points=args.key_points or "Not provided",
        rubric=rubric,
        anchors=anchors,
    )
    print(f"Results saved to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Oral Exam — single file or batch mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Single-file positional args (original interface) ──────────────────────
    parser.add_argument(
        "audio", nargs="?",
        help="[single-file mode] Local path or Google Drive URL to audio/video file",
    )
    parser.add_argument(
        "question", nargs="?",
        help="[single-file mode] The exam question",
    )

    # ── Batch modes ───────────────────────────────────────────────────────────
    parser.add_argument("--pipeline", metavar="INPUT_JSON",
                        help="Run full pipeline: ingest + evaluate")
    parser.add_argument("--ingest", metavar="INPUT_JSON",
                        help="Ingest only (download + transcribe, no evaluation)")
    parser.add_argument("--evaluate", metavar="TRANSCRIPTS_JSON",
                        help="Evaluate stored transcripts")

    # ── Shared batch options ──────────────────────────────────────────────────
    parser.add_argument("--output", default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--method", default="direct_scoring",
                        help="Evaluation method for --evaluate")
    parser.add_argument("--methods", nargs="+",
                        help="Evaluation methods for --pipeline (default: all)")
    parser.add_argument("--rubric", metavar="JSON_FILE",
                        help="Path to rubric JSON (required for rubric method)")
    parser.add_argument("--anchors", metavar="JSON_FILE",
                        help="Path to anchors JSON (required for anchor method)")
    parser.add_argument("--key-points", dest="key_points", default="",
                        help="Expected key points for direct_scoring")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct",
                        help="HuggingFace model ID for LLM evaluation")
    # pairwise
    parser.add_argument("--pairwise-batch-size", dest="pairwise_batch_size",
                        type=int, default=None,
                        help="Run all-vs-all only inside batches of this size")
    parser.add_argument("--pairwise-shuffle-seed", dest="pairwise_shuffle_seed",
                        type=int, default=42)
    # tournament
    parser.add_argument("--tournament-mode", dest="tournament_mode",
                        default="group_ranking",
                        choices=["group_ranking", "round_robin", "sampled"])
    parser.add_argument("--tournament-batch-size", dest="tournament_batch_size",
                        type=int, default=None)
    parser.add_argument("--tournament-group-size", dest="tournament_group_size",
                        type=int, default=10)
    parser.add_argument("--tournament-shuffle-seed", dest="tournament_shuffle_seed",
                        type=int, default=42)
    parser.add_argument("--tournament-sample-k", dest="tournament_sample_k",
                        type=int, default=None)

    args = parser.parse_args()

    if args.pipeline:
        _run_pipeline(args)
    elif args.ingest:
        _run_ingest(args)
    elif args.evaluate:
        _run_evaluate(args)
    elif args.audio and args.question:
        _run_single(args.audio, args.question)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
