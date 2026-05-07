"""
Standalone test: send the full extracted WAV to ElevenLabs (no chunking).
Does NOT modify or import the main pipeline code.
Run from the project root:
    python test_elevenlabs_full_file.py /tmp/exam_video.mp4
"""
import mimetypes
import os
import re
import subprocess
import sys
import tempfile

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["ELEVENLABS_API_KEY"]


def get_ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(video_path: str) -> str:
    ffmpeg = get_ffmpeg_exe()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error",
         "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-y", tmp.name],
        check=True, timeout=300,
    )
    return tmp.name


def get_duration(path: str) -> float:
    ffmpeg = get_ffmpeg_exe()
    result = subprocess.run([ffmpeg, "-hide_banner", "-i", path],
                            capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not m:
        return -1.0
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


def rebuild_from_words(words: list) -> str:
    if not words:
        return ""
    has_type = any("type" in w for w in words)
    if has_type:
        return "".join(w.get("text", "") for w in words)
    PUNCT = set(".,!?;:\"'…)-")
    parts = []
    for w in words:
        token = w.get("text") or w.get("word", "")
        if not token:
            continue
        if parts and token[0] not in PUNCT:
            parts.append(" ")
        parts.append(token)
    return "".join(parts)


def transcribe_full(audio_path: str) -> str:
    filename = os.path.basename(audio_path)
    mime_type, _ = mimetypes.guess_type(audio_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    size = os.path.getsize(audio_path)
    duration = get_duration(audio_path)
    print(f"[DIAG] File            : {filename}")
    print(f"[DIAG] Size            : {size / 1_048_576:.2f} MB")
    print(f"[DIAG] Duration        : {duration:.2f}s")
    print(f"[DIAG] MIME            : {mime_type}")
    print(f"[DIAG] Sending to ElevenLabs (no chunking)...")

    with open(audio_path, "rb") as f:
        response = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": API_KEY},
            files={"file": (filename, f, mime_type)},
            data={"model_id": "scribe_v2", "primary_language": "heb"},
            timeout=600,
        )

    print(f"[DIAG] Status          : {response.status_code}")
    if response.status_code != 200:
        print(f"[ERROR] {response.text}")
        return ""

    payload = response.json()
    words = payload.get("words") or []
    text_field = payload.get("text", "")
    last_word = words[-1] if words else None

    print(f"[DIAG] text len        : {len(text_field)} chars")
    print(f"[DIAG] words count     : {len(words)}")
    print(f"[DIAG] last word       : {last_word}")
    if last_word:
        last_end = last_word.get("end", -1)
        print(f"[DIAG] transcribed up to: {last_end:.2f}s / {duration:.2f}s")
        print(f"[DIAG] missing tail    : {duration - last_end:.2f}s")

    transcript = rebuild_from_words(words) if words else text_field
    print(f"[DIAG] final len       : {len(transcript)} chars")
    print(f"\n--- TRANSCRIPT ---\n{transcript}\n------------------")
    return transcript


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/exam_video.mp4"
    ext = os.path.splitext(input_path)[1].lower()
    audio_tmp = None

    video_suffixes = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    if ext in video_suffixes:
        print(f"[DIAG] Extracting audio from video...")
        audio_tmp = extract_audio(input_path)
        send_path = audio_tmp
    else:
        send_path = input_path

    try:
        transcribe_full(send_path)
    finally:
        if audio_tmp and os.path.exists(audio_tmp):
            os.remove(audio_tmp)
