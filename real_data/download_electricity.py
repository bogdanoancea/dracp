from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_URLS = [
    "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip",
]


def build_session(retries: int) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "DRACP-research-pipeline/0.1", "Accept-Encoding": "identity"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def download_with_resume(session: requests.Session, url: str, destination: Path, attempts: int, chunk_size: int) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, attempts + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            print(f"Download attempt {attempt}/{attempts}: {url}\nExisting partial bytes: {existing:,}")
            with session.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                if response.status_code == 416 and partial.exists():
                    partial.replace(destination)
                    return
                response.raise_for_status()
                range_honoured = response.status_code == 206
                if existing and not range_honoured:
                    print("Server ignored Range; restarting this download.")
                    partial.unlink(missing_ok=True)
                    existing = 0
                mode = "ab" if existing and range_honoured else "wb"
                expected_new = response.headers.get("Content-Length")
                expected_new = int(expected_new) if expected_new else None
                downloaded_new = 0
                with partial.open(mode) as fh:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fh.write(chunk)
                            downloaded_new += len(chunk)
                if expected_new is not None and downloaded_new != expected_new:
                    raise IOError(f"Incomplete response: received {downloaded_new:,} of {expected_new:,} bytes.")
            partial.replace(destination)
            print(f"Downloaded {destination.stat().st_size:,} bytes.")
            return
        except (requests.RequestException, OSError) as exc:
            print(f"Attempt {attempt} failed: {exc}")
            if attempt == attempts:
                raise
            wait = min(60, 2 ** attempt)
            print(f"Retrying in {wait} seconds...")
            time.sleep(wait)


def find_data_file(output_dir: Path) -> Path | None:
    matches = list(output_dir.rglob("LD2011_2014.txt"))
    return matches[0] if matches else None


def main(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_data = find_data_file(output_dir)
    if existing_data and existing_data.stat().st_size > 0 and not args.force:
        print(f"Dataset already exists: {existing_data}")
        return

    archive = output_dir / "electricityloaddiagrams20112014.zip"
    if args.force:
        archive.unlink(missing_ok=True)
        archive.with_suffix(archive.suffix + ".part").unlink(missing_ok=True)

    urls = [args.url] if args.url else DEFAULT_URLS
    session = build_session(args.http_retries)
    last_error = None

    for url in urls:
        try:
            if not valid_zip(archive):
                archive.unlink(missing_ok=True)
                download_with_resume(session, url, archive, args.attempts, args.chunk_size_mb * 1024 * 1024)
            if not valid_zip(archive):
                raise zipfile.BadZipFile(f"Downloaded archive failed ZIP validation: {archive}")
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(output_dir)
            data_file = find_data_file(output_dir)
            if data_file is None:
                raise FileNotFoundError("Archive extracted, but LD2011_2014.txt was not found.")
            print(f"Dataset ready: {data_file}")
            return
        except Exception as exc:
            last_error = exc
            print(f"URL failed: {url}\nReason: {exc}")
            archive.unlink(missing_ok=True)
            archive.with_suffix(archive.suffix + ".part").unlink(missing_ok=True)

    raise RuntimeError(
        "All official download URLs failed. Download manually from the UCI dataset page and place LD2011_2014.txt under "
        f"{output_dir}."
    ) from last_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw/electricity")
    parser.add_argument("--url", default=None)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--http-retries", type=int, default=5)
    parser.add_argument("--chunk-size-mb", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())
