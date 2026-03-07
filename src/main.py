import argparse
import os

from src.audio.loader import download_gdrive_audio, is_gdrive_url
from src.orchestration.single_turn import SingleTurnExam


def main():
    parser = argparse.ArgumentParser(description="Interactive Oral Exam")
    parser.add_argument("audio", help="Local path or Google Drive share link to the audio file")
    parser.add_argument("question", help="The exam question that was asked")
    args = parser.parse_args()

    tmp_path = None
    if is_gdrive_url(args.audio):
        print("Downloading audio from Google Drive...")
        tmp_path = download_gdrive_audio(args.audio)
        audio_path = tmp_path
    else:
        audio_path = args.audio

    try:
        exam = SingleTurnExam()
        result = exam.run(audio_path, args.question)
        print(f"Score:    {result.score}/100")
        print(f"Label:    {result.label}")
        print(f"Feedback: {result.feedback}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    main()
