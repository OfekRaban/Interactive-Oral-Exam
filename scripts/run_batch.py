#!/usr/bin/env python3
"""
run_batch.py — Full-featured CLI for batch oral exam evaluation.

Subcommands
-----------
ingest      Download + transcribe N student files, save transcripts.
evaluate    Run one evaluation method on stored transcripts.
pipeline    Run ingest + one or more evaluation methods end-to-end.
summary     Print a human-readable summary of stored results.

Examples
--------
# Ingest (download + transcribe)
python scripts/run_batch.py ingest \\
    --input examples/batch_input_example.json \\
    --output output/

# Direct scoring
python scripts/run_batch.py evaluate \\
    --transcripts output/exam_001_transcripts.json \\
    --method direct_scoring \\
    --output output/

# Rubric-based evaluation
python scripts/run_batch.py evaluate \\
    --transcripts output/exam_001_transcripts.json \\
    --method rubric \\
    --rubric examples/rubric_example.json \\
    --output output/

# Anchor-based evaluation
python scripts/run_batch.py evaluate \\
    --transcripts output/exam_001_transcripts.json \\
    --method anchor \\
    --anchors examples/anchor_example.json \\
    --output output/

# All-pairs pairwise
python scripts/run_batch.py evaluate \\
    --transcripts output/exam_001_transcripts.json \\
    --method pairwise \\
    --output output/

# Tournament ranking
python scripts/run_batch.py evaluate \\
    --transcripts output/exam_001_transcripts.json \\
    --method tournament \\
    --output output/

# Full pipeline (ingest + all methods)
python scripts/run_batch.py pipeline \\
    --input examples/batch_input_example.json \\
    --output output/ \\
    --methods all \\
    --rubric examples/rubric_example.json \\
    --anchors examples/anchor_example.json

# Full pipeline (ingest + selected methods)
python scripts/run_batch.py pipeline \\
    --input examples/batch_input_example.json \\
    --output output/ \\
    --methods direct_scoring rubric tournament \\
    --rubric examples/rubric_example.json

# Print summary of results
python scripts/run_batch.py summary \\
    --results output/results/exam_001_direct_scoring_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Subcommand handlers ───────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> None:
    from src.orchestration.batch_pipeline import BatchPipeline

    pipeline = BatchPipeline(
        output_dir=args.output,
        stt_backend=args.stt_backend,
        max_stt_workers=args.stt_workers,
    )
    records = pipeline.ingest(args.input)
    ok = sum(1 for r in records if not r.error)
    failed = [r for r in records if r.error]

    print(f"\n{'─'*60}")
    print(f"Ingestion complete: {ok}/{len(records)} students transcribed")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  ✗ {r.student_id} ({r.name}): {r.error}")
    print(f"Transcripts saved to: {args.output}/")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from src.orchestration.batch_pipeline import BatchPipeline

    rubric = json.load(open(args.rubric)) if args.rubric else None
    anchors = json.load(open(args.anchors)) if args.anchors else None

    pipeline = BatchPipeline(
        output_dir=args.output,
        model_name=args.model,
        pairwise_batch_size=args.pairwise_batch_size,
        pairwise_shuffle_seed=args.pairwise_shuffle_seed,
        tournament_mode=args.tournament_mode,
        tournament_batch_size=args.tournament_batch_size,
        tournament_group_size=args.tournament_group_size,
        tournament_shuffle_seed=args.tournament_shuffle_seed,
        tournament_sample_k=args.tournament_sample_k,
    )
    path = pipeline.evaluate(
        args.transcripts,
        args.method,
        expected_key_points=args.key_points or "Not provided",
        rubric=rubric,
        anchors=anchors,
    )
    print(f"\nResults saved to: {path}")
    _print_result_summary(str(path), args.method)


def cmd_pipeline(args: argparse.Namespace) -> None:
    from src.orchestration.batch_pipeline import BatchPipeline

    rubric = json.load(open(args.rubric)) if args.rubric else None
    anchors = json.load(open(args.anchors)) if args.anchors else None
    methods: list | None = None if args.methods == ["all"] else args.methods

    pipeline = BatchPipeline(
        output_dir=args.output,
        model_name=args.model,
        stt_backend=args.stt_backend,
        max_stt_workers=args.stt_workers,
        pairwise_batch_size=args.pairwise_batch_size,
        pairwise_shuffle_seed=args.pairwise_shuffle_seed,
        tournament_mode=args.tournament_mode,
        tournament_batch_size=args.tournament_batch_size,
        tournament_group_size=args.tournament_group_size,
        tournament_shuffle_seed=args.tournament_shuffle_seed,
        tournament_sample_k=args.tournament_sample_k,
    )
    results = pipeline.run_pipeline(
        input_path=args.input,
        methods=methods,
        expected_key_points=args.key_points or "Not provided",
        rubric=rubric,
        anchors=anchors,
        skip_ingestion=args.skip_ingestion,
        transcript_path=args.transcripts,
    )

    print(f"\n{'─'*60}")
    print("Pipeline complete:")
    for method, path in results.items():
        status = "✓" if not str(path).startswith("ERROR") else "✗"
        print(f"  {status} {method:20s} → {path}")


def cmd_summary(args: argparse.Namespace) -> None:
    _print_result_summary(args.results, method=None, verbose=args.verbose)


# ── Result summary printer ────────────────────────────────────────────────────

def _print_result_summary(path: str, method: str | None = None, verbose: bool = False) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Could not read results: {e}")
        return

    method = method or data.get("method", "unknown")
    results = data.get("results", [])
    if not results:
        print("No results found.")
        return

    print(f"\n{'─'*60}")
    print(f"Method : {method}")
    print(f"Count  : {len(results)}")
    print(f"{'─'*60}")

    if method == "direct_scoring":
        _summary_direct(results, verbose)
    elif method == "rubric":
        _summary_rubric(results, verbose)
    elif method == "anchor":
        _summary_anchor(results, verbose)
    elif method == "pairwise":
        _summary_pairwise(results, verbose)
    elif method in ("tournament", "tournament_group_rankings"):
        if results and "ranked_student_ids" in results[0]:
            _summary_group_rankings(results, verbose)
        else:
            _summary_tournament(results, verbose)
    else:
        for r in results:
            print(json.dumps(r, ensure_ascii=False, indent=2))


def _summary_direct(results: list, verbose: bool) -> None:
    for r in sorted(results, key=lambda x: -x.get("score", 0)):
        name = r.get("name", r.get("student_id", "?"))
        score = r.get("score", 0)
        label = r.get("label", "?")
        error = r.get("error")
        line = f"  {score:3d}/100  [{label:9s}]  {name}"
        if error:
            line += f"  ✗ ERROR: {error}"
        print(line)
        if verbose and r.get("feedback"):
            print(f"           {r['feedback']}")
    scores = [r["score"] for r in results if "score" in r and not r.get("error")]
    if scores:
        print(f"\n  Average: {sum(scores)/len(scores):.1f}  "
              f"Min: {min(scores)}  Max: {max(scores)}")


def _summary_rubric(results: list, verbose: bool) -> None:
    for r in sorted(results, key=lambda x: -x.get("normalized_score", 0)):
        name = r.get("name", r.get("student_id", "?"))
        ns = r.get("normalized_score", 0)
        ts = r.get("total_score", 0)
        ms = r.get("max_score", 0)
        label = r.get("label", "?")
        print(f"  {ns:5.1f}/100  ({ts:.0f}/{ms:.0f})  [{label:9s}]  {name}")
        if verbose:
            for c in r.get("criteria", []):
                print(f"    {c['name']:30s}  {c['score']:.0f}/{c['max_score']:.0f}  {c['feedback']}")


def _summary_anchor(results: list, verbose: bool) -> None:
    for r in sorted(results, key=lambda x: -x.get("score", 0)):
        name = r.get("name", r.get("student_id", "?"))
        score = r.get("score", 0)
        anchor = r.get("closest_anchor", "?")
        label = r.get("label", "?")
        print(f"  {score:5.1f}/100  [closest: {anchor:10s}]  [{label:9s}]  {name}")
        if verbose and r.get("feedback"):
            print(f"           {r['feedback']}")


def _summary_pairwise(results: list, verbose: bool) -> None:
    print(f"  Total matchups: {len(results)}")
    for r in results:
        a = r.get("student_a_id", "?")
        b = r.get("student_b_id", "?")
        winner = r.get("winner", "?")
        conf = r.get("confidence", 0)
        winner_label = a if winner == "A" else (b if winner == "B" else "TIE")
        print(f"  {a:10s} vs {b:10s}  →  {winner_label}  (conf {conf:.2f})")
        if verbose and r.get("reasoning"):
            print(f"    {r['reasoning']}")


def _summary_tournament(results: list, verbose: bool) -> None:
    for r in sorted(results, key=lambda x: x.get("rank", 999)):
        rank = r.get("rank", "?")
        name = r.get("name", r.get("student_id", "?"))
        pts = r.get("points", 0)
        w = r.get("wins", 0)
        l = r.get("losses", 0)
        t = r.get("ties", 0)
        print(f"  #{rank:2d}  {pts:4.1f}pts  W{w}/L{l}/T{t}  {name}")


def _summary_group_rankings(results: list, verbose: bool) -> None:
    """Summary for tournament_group_rankings results (one entry per group)."""
    print(f"  Total groups: {len(results)}")
    for r in results:
        gid = r.get("group_id", "?")
        bid = r.get("batch_id", "?")
        ranked = r.get("ranked_student_ids", [])
        err = r.get("error")
        status = f"  ✗ ERROR: {err}" if err else ""
        print(f"  [{bid}] {gid}: {' > '.join(ranked)}{status}")
        if verbose and r.get("reasoning"):
            print(f"    {r['reasoning']}")


# ── Parser construction ───────────────────────────────────────────────────────

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct",
                        help="LLM model name")
    parser.add_argument("--rubric", metavar="JSON_FILE",
                        help="Rubric JSON (required for method=rubric)")
    parser.add_argument("--anchors", metavar="JSON_FILE",
                        help="Anchors JSON (required for method=anchor)")
    parser.add_argument("--key-points", dest="key_points", default="",
                        help="Expected key points (for direct_scoring)")
    # ── pairwise ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--pairwise-batch-size", dest="pairwise_batch_size", type=int, default=None,
        help="If set, shuffle students and run all-vs-all only inside each batch "
             "of this size.  None (default) = global all-vs-all.",
    )
    parser.add_argument(
        "--pairwise-shuffle-seed", dest="pairwise_shuffle_seed", type=int, default=42,
        help="RNG seed for pairwise batch shuffling (default: 42)",
    )
    # ── tournament ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--tournament-mode", dest="tournament_mode", default="group_ranking",
        choices=["group_ranking", "round_robin", "sampled"],
        help="Tournament mode (default: group_ranking)",
    )
    parser.add_argument(
        "--tournament-batch-size", dest="tournament_batch_size", type=int, default=None,
        help="[group_ranking] Outer batch size.  None = one global batch.",
    )
    parser.add_argument(
        "--tournament-group-size", dest="tournament_group_size", type=int, default=10,
        help="[group_ranking] Students per LLM ranking call (default: 10)",
    )
    parser.add_argument(
        "--tournament-shuffle-seed", dest="tournament_shuffle_seed", type=int, default=42,
        help="[group_ranking] RNG seed for shuffling before forming groups (default: 42)",
    )
    parser.add_argument(
        "--tournament-sample-k", dest="tournament_sample_k", type=int, default=None,
        help="[legacy sampled mode] Opponents per student",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch Interactive Oral Exam — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # ── ingest ────────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Download + transcribe student files")
    p_ingest.add_argument("--input", required=True, metavar="JSON_FILE",
                          help="Path to batch_input.json")
    p_ingest.add_argument("--output", default="output")
    p_ingest.add_argument("--stt-backend", dest="stt_backend",
                          default="elevenlabs", choices=["elevenlabs", "whisper"])
    p_ingest.add_argument("--stt-workers", dest="stt_workers", type=int, default=1,
                          help="Parallel workers for API STT (use 1 for local Whisper)")

    # ── evaluate ──────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate", help="Evaluate stored transcripts")
    p_eval.add_argument("--transcripts", required=True, metavar="JSON_FILE",
                        help="Path to transcripts JSON file")
    p_eval.add_argument("--method", required=True,
                        choices=["direct_scoring", "rubric", "anchor", "pairwise", "tournament"],
                        help="Evaluation method")
    _add_common_args(p_eval)

    # ── pipeline ──────────────────────────────────────────────────────────────
    p_pipe = sub.add_parser("pipeline", help="Full pipeline: ingest + evaluate")
    p_pipe.add_argument("--input", required=True, metavar="JSON_FILE",
                        help="Path to batch_input.json")
    p_pipe.add_argument("--methods", nargs="+", default=["all"],
                        help="Methods to run (default: all)")
    p_pipe.add_argument("--stt-backend", dest="stt_backend",
                        default="elevenlabs", choices=["elevenlabs", "whisper"])
    p_pipe.add_argument("--stt-workers", dest="stt_workers", type=int, default=1)
    p_pipe.add_argument("--skip-ingestion", dest="skip_ingestion",
                        action="store_true",
                        help="Skip ingestion and use --transcripts instead")
    p_pipe.add_argument("--transcripts", metavar="JSON_FILE",
                        help="Transcript file to use when --skip-ingestion is set")
    _add_common_args(p_pipe)

    # ── summary ───────────────────────────────────────────────────────────────
    p_sum = sub.add_parser("summary", help="Print human-readable result summary")
    p_sum.add_argument("--results", required=True, metavar="JSON_FILE",
                       help="Path to a results JSON file")
    p_sum.add_argument("--verbose", "-v", action="store_true",
                       help="Show per-criterion / per-matchup details")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.subcommand == "ingest":
        cmd_ingest(args)
    elif args.subcommand == "evaluate":
        cmd_evaluate(args)
    elif args.subcommand == "pipeline":
        cmd_pipeline(args)
    elif args.subcommand == "summary":
        cmd_summary(args)


if __name__ == "__main__":
    main()
