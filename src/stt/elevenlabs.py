import mimetypes
import os
import re
import subprocess
import tempfile

import requests

from src.stt.base import SpeechToText, TranscriptionResult

# ── Feature flag ──────────────────────────────────────────────────────────────
# Set to True to split audio into chunks before transcribing.
# Flip to False to restore single-file upload behavior.
_USE_CHUNKING = False
_CHUNK_COUNT = 2

# Punctuation that normally ends a sentence.
_SENTENCE_ENDINGS = {".", "!", "?", "…"}

_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


# ── ffmpeg helpers (use imageio-ffmpeg bundle, no system install needed) ──────

def _get_ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _get_audio_duration(path: str) -> float:
    """
    Return audio duration in seconds by parsing ffmpeg -i stderr output.
    ffmpeg always exits non-zero when no output file is given — that is expected.
    """
    ffmpeg = _get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", path],
        capture_output=True, text=True,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise ValueError(f"Could not parse duration from ffmpeg output for {path}")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def _extract_audio(video_path: str) -> str:
    """
    Extract mono 16 kHz WAV from a video file using the imageio-ffmpeg bundle.
    Returns path to a temporary WAV file. Caller must delete it.
    """
    ffmpeg = _get_ffmpeg_exe()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-y",
            tmp.name,
        ],
        check=True, timeout=300,
    )
    return tmp.name


def _split_audio(audio_path: str, n_chunks: int) -> list:
    """
    Split audio_path into n_chunks roughly equal sequential WAV files.
    Returns a list of temp file paths. Caller must delete them.
    """
    ffmpeg = _get_ffmpeg_exe()
    total = _get_audio_duration(audio_path)
    chunk_dur = total / n_chunks
    print(f"[DIAG] Total duration  : {total:.2f}s")
    print(f"[DIAG] Chunk duration  : {chunk_dur:.2f}s each ({n_chunks} chunks)")

    paths = []
    for i in range(n_chunks):
        start = i * chunk_dur
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk{i}.wav")
        tmp.close()
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-i", audio_path,
                "-ss", f"{start:.3f}",
                "-t",  f"{chunk_dur:.3f}",
                "-y",
                tmp.name,
            ],
            check=True, timeout=180,
        )
        size = os.path.getsize(tmp.name)
        print(f"[DIAG] Chunk {i}          : {tmp.name}  ({size / 1_048_576:.2f} MB)")
        paths.append(tmp.name)

    return paths


# ── Transcript helpers ────────────────────────────────────────────────────────

def _rebuild_transcript_from_words(words: list) -> str:
    """
    Reconstruct transcript from the words array returned by ElevenLabs Scribe.

    ElevenLabs Scribe word objects can have a `type` field with values:
      "word", "spacing", "punctuation"
    When that field is present, spacing tokens already contain the whitespace,
    so we just concatenate all `text` values directly.
    When `type` is absent, we join words with a single space and attach
    punctuation tokens without a preceding space.
    """
    if not words:
        return ""

    has_type = any("type" in w for w in words)

    if has_type:
        # Direct concatenation: spacing tokens carry the whitespace.
        return "".join(w.get("text", "") for w in words)

    # Fallback: space-join words, attach punctuation without leading space.
    _PUNCT = set(".,!?;:\"'…)-")
    parts = []
    for w in words:
        token = w.get("text") or w.get("word", "")
        if not token:
            continue
        if parts and token[0] not in _PUNCT:
            parts.append(" ")
        parts.append(token)
    return "".join(parts)


def _looks_truncated(transcript: str) -> bool:
    stripped = transcript.rstrip(' "\'')
    return bool(stripped) and stripped[-1] not in _SENTENCE_ENDINGS


# ── Core per-file transcription (shared by single-file and chunked paths) ─────

def _transcribe_file(path: str, api_key: str, label: str = "") -> str:
    """
    Upload one audio/video file to ElevenLabs Scribe and return the best
    available transcript string. Prints [DIAG] lines prefixed with label.
    """
    tag = f"[{label}] " if label else ""

    filename = os.path.basename(path)
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    size = os.path.getsize(path)
    print(f"[DIAG] {tag}Upload file    : {filename}  ({size / 1_048_576:.2f} MB, {mime_type})")

    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": api_key}

    with open(path, "rb") as f:
        files = {"file": (filename, f, mime_type)}
        data = {"model_id": "scribe_v2", "primary_language": "heb"}
        response = requests.post(
            url, headers=headers, files=files, data=data,
            timeout=600,
        )

    print(f"[DIAG] {tag}Status         : {response.status_code}")

    if response.status_code != 200:
        raise Exception(
            f"STT API call failed ({tag.strip()}) status "
            f"{response.status_code}: {response.text}"
        )

    payload = response.json()
    words = payload.get("words") or []
    text_field = payload.get("text", "")

    last_word = words[-1] if words else None
    print(f"[DIAG] {tag}text len       : {len(text_field)} chars")
    print(f"[DIAG] {tag}words count    : {len(words)}")
    print(f"[DIAG] {tag}last word      : {last_word}")

    if words:
        transcript = _rebuild_transcript_from_words(words)
        print(f"[DIAG] {tag}source         : words (primary)")
    else:
        transcript = text_field
        print(f"[DIAG] {tag}source         : text (fallback — words missing)")

    print(f"[DIAG] {tag}final len      : {len(transcript)} chars")
    if _looks_truncated(transcript):
        print(f"[WARN] {tag}ends abruptly  : ...{repr(transcript[-80:])}")

    return transcript


# ── Main STT class ────────────────────────────────────────────────────────────

class ElevenLabsSTT(SpeechToText):

    def __init__(self, api_key: str):
        self.api_key = api_key

    def transcribe(self, audio_path: str) -> TranscriptionResult:

        # ── Validate input ────────────────────────────────────────────────
        abs_path = os.path.abspath(audio_path)
        if not os.path.exists(abs_path) or not os.access(abs_path, os.R_OK):
            raise FileNotFoundError(f"Audio file not accessible: {abs_path}")

        orig_size = os.path.getsize(abs_path)
        ext = os.path.splitext(abs_path)[1].lower()
        print(f"[DIAG] Input path      : {abs_path}")
        print(f"[DIAG] Input size      : {orig_size / 1_048_576:.2f} MB")
        print(f"[DIAG] Input extension : {ext}")

        # ── Extract audio from video if needed ────────────────────────────
        audio_tmp = None
        if ext in _VIDEO_SUFFIXES:
            print(f"[DIAG] Extracting audio via imageio-ffmpeg...")
            audio_tmp = _extract_audio(abs_path)
            send_path = audio_tmp
            print(f"[DIAG] Extracted audio : {send_path}  "
                  f"({os.path.getsize(send_path) / 1_048_576:.2f} MB)")
        else:
            send_path = abs_path

        # ── Transcribe ────────────────────────────────────────────────────
        chunk_tmps = []
        try:
            if _USE_CHUNKING:
                transcript = self._transcribe_chunked(send_path)
            else:
                transcript = _transcribe_file(send_path, self.api_key, label="full")
        finally:
            if audio_tmp and os.path.exists(audio_tmp):
                os.remove(audio_tmp)
            for p in chunk_tmps:
                if os.path.exists(p):
                    os.remove(p)

        return TranscriptionResult(transcript=transcript, confidence=None)

    def _transcribe_chunked(self, audio_path: str) -> str:
        """Split audio into _CHUNK_COUNT chunks, transcribe each, merge."""
        chunk_paths = _split_audio(audio_path, _CHUNK_COUNT)
        transcripts = []

        try:
            for i, chunk_path in enumerate(chunk_paths):
                print(f"\n[DIAG] ── Chunk {i} ──────────────────────────────────────")
                t = _transcribe_file(chunk_path, self.api_key, label=f"chunk{i}")
                transcripts.append(t)
        finally:
            for p in chunk_paths:
                if os.path.exists(p):
                    os.remove(p)

        # ── Merge ─────────────────────────────────────────────────────────
        merged = " ".join(t.strip() for t in transcripts if t.strip())

        print(f"\n[DIAG] ── Merge results ─────────────────────────────────────")
        for i, t in enumerate(transcripts):
            print(f"[DIAG] Chunk {i} len       : {len(t)} chars")
        print(f"[DIAG] Merged len         : {len(merged)} chars")
        print(f"[DIAG] End of chunk 0     : ...{repr(transcripts[0][-100:])}")
        print(f"[DIAG] Start of chunk 1   : {repr(transcripts[1][:100])}...")
        print(f"[DIAG] End of merged      : ...{repr(merged[-100:])}")

        if _looks_truncated(merged):
            print(f"[WARN] Merged transcript still ends abruptly.")

        return merged
