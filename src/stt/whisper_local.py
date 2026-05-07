import os
import subprocess
import tempfile
from typing import Optional


from src.stt.base import SpeechToText, TranscriptionResult

_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _get_ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _extract_audio(video_path: str) -> str:
    """Extract mono 16 kHz WAV from a video. Returns temp path; caller must delete."""
    ffmpeg = _get_ffmpeg_exe()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-y", tmp.name,
        ],
        check=True, timeout=300,
    )
    return tmp.name


class WhisperLocalSTT(SpeechToText):

    def __init__(
        self,
        model_size: str = "large-v3",
        device: Optional[str] = None,
        compute_type: str = "float16",
    ):
        from faster_whisper import WhisperModel
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        effective_compute_type = compute_type if device == "cuda" else "int8"

        print(f"[DIAG] Whisper model   : {model_size}")
        print(f"[DIAG] Device          : {device}  compute_type={effective_compute_type}")

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=effective_compute_type,
        )

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        abs_path = os.path.abspath(audio_path)
        if not os.path.exists(abs_path) or not os.access(abs_path, os.R_OK):
            raise FileNotFoundError(f"Audio file not accessible: {abs_path}")

        ext = os.path.splitext(abs_path)[1].lower()
        size = os.path.getsize(abs_path)
        print(f"[DIAG] Input path      : {abs_path}")
        print(f"[DIAG] Input size      : {size / 1_048_576:.2f} MB")
        print(f"[DIAG] Input extension : {ext}")

        audio_tmp = None
        if ext in _VIDEO_SUFFIXES:
            print(f"[DIAG] Extracting audio via imageio-ffmpeg...")
            audio_tmp = _extract_audio(abs_path)
            send_path = audio_tmp
            print(f"[DIAG] Extracted audio : {send_path}  "
                  f"({os.path.getsize(send_path) / 1_048_576:.2f} MB)")
        else:
            send_path = abs_path

        try:
            segments, info = self.model.transcribe(
                send_path,
                language="he",      # Hebrew; set to None for auto-detect
                beam_size=5,
            )

            print(f"[DIAG] Detected lang   : {info.language}  "
                  f"prob={info.language_probability:.2f}")

            parts = []
            n_segments = 0
            for seg in segments:
                parts.append(seg.text.strip())
                n_segments += 1

            transcript = " ".join(p for p in parts if p)

        finally:
            if audio_tmp and os.path.exists(audio_tmp):
                os.remove(audio_tmp)

        print(f"[DIAG] Segments        : {n_segments}")
        print(f"[DIAG] Transcript len  : {len(transcript)} chars")

        return TranscriptionResult(transcript=transcript, confidence=None)
