# Interactive Oral Exam

An automated pipeline for evaluating student oral exam answers at scale.

**Flow:** Google Drive / local file → transcription → LLM evaluation → structured results

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ELEVENLABS_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here      # optional, not used by default
```

---

## Quick Start — Single file (original interface, unchanged)

```bash
python -m src.main audio.mp3 "Explain gradient descent"
python -m src.main https://drive.google.com/file/d/FILE_ID/view "Explain gradient descent"
```

Output:
```
Score:    85/100
Label:    correct
Feedback: Student correctly describes the update rule and learning rate role.
```

---

## Batch Evaluation

### 1. Prepare the input file

See `examples/batch_input_example.json`:

```json
{
  "exam_id": "exam_ml_2024_01",
  "question": "Explain gradient descent",
  "expected_key_points": ["update rule", "learning rate", "convergence"],
  "students": [
    {
      "student_id": "s001",
      "name": "Alice Cohen",
      "file": "https://drive.google.com/file/d/FILE_ID/view"
    },
    {
      "student_id": "s002",
      "name": "Bob Levi",
      "file": "/local/path/to/bob.mp4"
    }
  ]
}
```

Each student's `file` can be a Google Drive share link or a local file path.

---

### 2. Ingest (download + transcribe)

```bash
python scripts/run_batch.py ingest \
    --input examples/batch_input_example.json \
    --output output/
```

**What happens:**
- Downloaded files are saved persistently to `output/raw_files/{student_id}.ext`.
- If a file already exists in `raw_files/`, it is reused without re-downloading.
- Transcripts are saved to `output/transcripts/{exam_id}_transcripts.json`.
- If some students are already transcribed, only the new ones are processed.
  Re-running ingest is safe and idempotent.

**STT backend:**
```bash
# Use local Whisper instead of ElevenLabs API
STT_BACKEND=whisper python scripts/run_batch.py ingest ...
python scripts/run_batch.py ingest --stt-backend whisper ...

# Parallelize API-based STT
python scripts/run_batch.py ingest --stt-workers 4 ...
```

---

### 3. Run Evaluation Methods

All evaluation methods load transcripts from `output/transcripts/` and save
results to `output/results/`.

#### Direct Scoring

Assigns a numeric score 0-100 and a `correct`/`partial`/`incorrect` label to each answer.
**1 LLM call per student.**

```bash
python scripts/run_batch.py evaluate \
    --transcripts output/exam_ml_2024_01_transcripts.json \
    --method direct_scoring \
    --output output/
```

#### Rubric-Based Evaluation

Scores each answer against a structured rubric with multiple weighted criteria.
**1 LLM call per student.** See `examples/rubric_example.json`.

```bash
python scripts/run_batch.py evaluate \
    --transcripts output/transcripts/exam_ml_2024_01_transcripts.json \
    --method rubric \
    --rubric examples/rubric_example.json \
    --output output/
```

#### Anchor-Based Evaluation

Grades by comparing each answer to calibrated reference anchor answers at known quality levels.
**1 LLM call per student.** See `examples/anchor_example.json`.

```bash
python scripts/run_batch.py evaluate \
    --transcripts output/transcripts/exam_ml_2024_01_transcripts.json \
    --method anchor \
    --anchors examples/anchor_example.json \
    --output output/
```

#### Pairwise Evaluation

Compares pairs of students — asks the LLM which answer is better.

**Global all-vs-all (default):** N*(N-1)/2 LLM calls. Every student is compared to every other student.

```bash
python scripts/run_batch.py evaluate \
    --transcripts output/transcripts/exam_ml_2024_01_transcripts.json \
    --method pairwise \
    --output output/
```

**Batched all-vs-all:** shuffle students, split into batches of `--pairwise-batch-size`, and run all-vs-all comparisons *only inside each batch*. No cross-batch comparisons are made. Rankings are local to each batch.

```bash
# 120 students, batches of 30 → 4 batches × 435 pairs = 1,740 LLM calls
python scripts/run_batch.py evaluate \
    --transcripts output/transcripts/exam_ml_2024_01_transcripts.json \
    --method pairwise \
    --pairwise-batch-size 30 \
    --pairwise-shuffle-seed 42 \
    --output output/
```

Each `PairwiseResult` has a `batch_id` field when batching is active (e.g. `"batch_001"`).

> **Note:** Rankings from batched pairwise are local to each batch. Cross-batch
> calibration (e.g. anchor-based normalization or batch overlap) is not
> currently implemented.

#### Tournament / Ranking Evaluation

Ranks all students from best to worst.

---

##### New default: `group_ranking` mode

Sends a group of answers to the LLM **all at once** and asks it to rank them.
Much more LLM-call-efficient than pairwise for large cohorts.

Algorithm:
1. Shuffle all students (reproducibly with `--tournament-shuffle-seed`).
2. Split into outer batches of `--tournament-batch-size` (default: one global batch).
3. Inside each batch, form groups of `--tournament-group-size` (default 10).
4. For each group: one LLM call → ranked list of student IDs + reasoning.
5. Rankings are local to each outer batch.

**Example — 120 students, batch_size=40, group_size=10:**
- 3 outer batches × 4 groups = **12 LLM calls total**.

```bash
# Default group_ranking with default group size (10)
python scripts/run_batch.py evaluate \
    --transcripts output/transcripts/exam_ml_2024_01_transcripts.json \
    --method tournament \
    --output output/

# Custom batch and group sizes
python scripts/run_batch.py evaluate \
    --transcripts output/transcripts/exam_ml_2024_01_transcripts.json \
    --method tournament \
    --tournament-batch-size 40 \
    --tournament-group-size 10 \
    --tournament-shuffle-seed 42 \
    --output output/
```

Saved outputs:
- `output/results/{exam_id}_tournament_results.json` — `TournamentRanking` per student (within-batch rank)
- `output/results/{exam_id}_tournament_group_rankings.json` — per-group raw results including `batch_id`, `group_id`, `ranked_student_ids`, `reasoning`, `raw_response`

> **Note:** Ranking is local to each outer batch. Students in different batches
> are not directly compared. Cross-batch calibration is not currently
> implemented.

---

##### Legacy: `round_robin` mode (pairwise-based)

Every student vs every other student. N*(N-1)/2 LLM calls.

```bash
python scripts/run_batch.py evaluate \
    --transcripts output/transcripts/exam_ml_2024_01_transcripts.json \
    --method tournament \
    --tournament-mode round_robin \
    --output output/
```

##### Legacy: `sampled` mode

Each student plays `--tournament-sample-k` random opponents.

```bash
python scripts/run_batch.py evaluate \
    --transcripts output/transcripts/exam_ml_2024_01_transcripts.json \
    --method tournament \
    --tournament-mode sampled \
    --tournament-sample-k 5 \
    --output output/
```

---

### 4. Full Pipeline (ingest + evaluate)

```bash
# Run all methods
python scripts/run_batch.py pipeline \
    --input examples/batch_input_example.json \
    --output output/ \
    --methods all \
    --rubric examples/rubric_example.json \
    --anchors examples/anchor_example.json

# Run selected methods
python scripts/run_batch.py pipeline \
    --input examples/batch_input_example.json \
    --output output/ \
    --methods direct_scoring rubric tournament \
    --rubric examples/rubric_example.json \
    --tournament-batch-size 40 \
    --tournament-group-size 10

# Skip ingestion (transcripts already exist)
python scripts/run_batch.py pipeline \
    --input examples/batch_input_example.json \
    --output output/ \
    --skip-ingestion \
    --transcripts output/transcripts/exam_ml_2024_01_transcripts.json \
    --methods direct_scoring rubric
```

---

### 5. Print Results Summary

```bash
# Compact summary
python scripts/run_batch.py summary \
    --results output/results/exam_ml_2024_01_direct_scoring_results.json

# Verbose (per-criterion / per-matchup details)
python scripts/run_batch.py summary --verbose \
    --results output/results/exam_ml_2024_01_rubric_results.json

# Group ranking summary
python scripts/run_batch.py summary \
    --results output/results/exam_ml_2024_01_tournament_group_rankings.json
```

---

## Output Structure

```
output/
  raw_files/                           downloaded audio/video files (persistent)
    s001.mp4
    s002.mp4
    ...
  transcripts/
    exam_ml_2024_01_transcripts.json   transcripts for all students
  results/
    exam_ml_2024_01_direct_scoring_results.json
    exam_ml_2024_01_rubric_results.json
    exam_ml_2024_01_anchor_results.json
    exam_ml_2024_01_pairwise_results.json
    exam_ml_2024_01_tournament_results.json         per-student rankings
    exam_ml_2024_01_tournament_group_rankings.json  per-group raw results
    exam_ml_2024_01_tournament_matchups.json        (legacy modes only)
```

All files are plain JSON and can be imported into Excel / pandas.

---

## Evaluation Methods Reference

| Method | Mode | LLM calls | Output | Use when |
|--------|------|-----------|--------|----------|
| `direct_scoring` | — | 1x per student | score, label, feedback | Quick overall grade |
| `rubric` | — | 1x per student | per-criterion scores + total | Transparent criterion-level grading |
| `anchor` | — | 1x per student | score + closest anchor | Calibrated grading with reference answers |
| `pairwise` | global | N(N-1)/2 | winner + confidence per pair | Full pairwise comparison |
| `pairwise` | batched | batch*(batch-1)/2 per batch | same + batch_id | Pairwise within isolated groups |
| `tournament` | group_ranking | #batches × #groups per batch | group rankings + leaderboard | Efficient ranking for large cohorts |
| `tournament` | round_robin | N(N-1)/2 | win/loss record + rank | Thorough pairwise-based ranking |
| `tournament` | sampled | N×k | win/loss record + rank | Approximate ranking, fewer calls |

---

## Architecture

```
src/
  main.py                          Entry point (single-file + batch modes)
  audio/loader.py                  Google Drive download
  batch/processor.py               BatchProcessor: download→raw_files/ + transcribe
  config/settings.py               .env loader
  evaluation/
    llm_engine.py                  Shared LLM inference engine (load once, share)
    local_llm_evaluator.py         Original single-answer evaluator (backward compat)
    direct_scoring.py              Batch direct scoring
    rubric_based.py                Rubric evaluation
    anchor_based.py                Anchor comparison
    pairwise.py                    Pairwise comparison (with optional batching)
    tournament.py                  Tournament: group_ranking + legacy round_robin/sampled
    prompts.py                     All LLM prompt templates
  models/
    evaluation_result.py           Result dataclasses for all methods
    student.py                     StudentRecord, TranscriptRecord
  orchestration/
    single_turn.py                 Original single-file orchestrator (unchanged)
    batch_pipeline.py              BatchPipeline top-level orchestrator
  storage/
    transcript_store.py            JSON storage for transcripts
    result_store.py                JSON storage for evaluation results
  stt/
    elevenlabs.py                  ElevenLabs Scribe API (default)
    whisper_local.py               Local Whisper (GPU)
scripts/
  run_batch.py                     Full-featured batch CLI (4 subcommands)
examples/
  batch_input_example.json         Input format
  rubric_example.json              Example rubric
  anchor_example.json              Example anchors
```

### Key design decisions

- **Persistent raw files** — downloaded files are kept in `output/raw_files/`. Re-running ingest reuses them without re-downloading.
- **Idempotent ingestion** — already-transcribed students are skipped automatically. Safe to re-run after partial failure.
- **Organized output** — `raw_files/`, `transcripts/`, `results/` are separate subdirectories.
- **Single shared LLM** — `LLMEngine` is loaded once and passed to all evaluators. No duplicate GPU memory usage.
- **Backward compatible** — `python -m src.main audio.mp3 "question"` works exactly as before.
- **Batched pairwise** — all-vs-all comparisons within isolated batches; rankings are local to each batch.
- **Group ranking tournament** — one LLM call per group instead of N*(N-1)/2; dramatically cheaper for large cohorts.
- **Fault tolerant** — a failed transcription or evaluation for one student does not stop the batch.
- **Storage-first** — transcription and evaluation are decoupled. Re-run evaluations on the same transcripts without re-downloading.

---

## All CLI Options

### `run_batch.py evaluate` / `run_batch.py pipeline`

| Flag | Default | Description |
|------|---------|-------------|
| `--pairwise-batch-size` | None | All-vs-all only inside batches of this size |
| `--pairwise-shuffle-seed` | 42 | Shuffle seed for pairwise batching |
| `--tournament-mode` | group_ranking | `group_ranking`, `round_robin`, or `sampled` |
| `--tournament-batch-size` | None | Outer batch size for group_ranking |
| `--tournament-group-size` | 10 | Students per LLM call in group_ranking |
| `--tournament-shuffle-seed` | 42 | Shuffle seed for group_ranking |
| `--tournament-sample-k` | None | Opponents per student in legacy sampled mode |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_BACKEND` | `elevenlabs` | STT backend: `elevenlabs` or `whisper` |
| `ELEVENLABS_API_KEY` | — | ElevenLabs API key (required for default STT) |
| `GEMINI_API_KEY` | — | Google Gemini key (optional) |
