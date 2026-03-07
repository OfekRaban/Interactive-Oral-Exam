import re
import tempfile
import requests


def _extract_gdrive_id(url: str) -> str:
    """Extract file ID from various Google Drive URL formats."""
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",   # .../file/d/{ID}/view
        r"[?&]id=([a-zA-Z0-9_-]+)",     # ...?id={ID} or &id={ID}
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract file ID from Google Drive URL: {url}")


def download_gdrive_audio(url: str) -> str:
    """
    Download an audio file from a Google Drive share link.
    Returns the path to a temporary local file.
    The caller is responsible for deleting the file after use.
    """
    file_id = _extract_gdrive_id(url)
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    session = requests.Session()
    response = session.get(download_url, stream=True)

    # Large files trigger a virus-scan confirmation page
    if "content-disposition" not in response.headers:
        # Extract confirmation token and retry
        token_match = re.search(r'name="confirm"\s+value="([^"]+)"', response.text)
        if not token_match:
            # Newer Google Drive format uses a different confirmation URL
            confirm_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
            response = session.get(confirm_url, stream=True)
        else:
            token = token_match.group(1)
            response = session.get(
                download_url, params={"confirm": token}, stream=True
            )

    if response.status_code != 200:
        raise Exception(
            f"Failed to download from Google Drive (status {response.status_code})"
        )

    suffix = _guess_suffix(response.headers.get("content-disposition", ""))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    for chunk in response.iter_content(chunk_size=8192):
        tmp.write(chunk)
    tmp.close()

    return tmp.name


def _guess_suffix(content_disposition: str) -> str:
    """Guess file extension from Content-Disposition header, default to .mp3."""
    match = re.search(r'filename="?([^";]+)"?', content_disposition)
    if match:
        name = match.group(1).strip()
        dot = name.rfind(".")
        if dot != -1:
            return name[dot:]
    return ".mp3"


def is_gdrive_url(path: str) -> bool:
    return "drive.google.com" in path or "drive.usercontent.google.com" in path
