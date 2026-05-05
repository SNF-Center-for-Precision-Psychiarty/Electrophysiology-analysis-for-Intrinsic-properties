"""
Download NWB files from DANDI Archive
======================================
Uses the DANDI REST API directly (no dandi-cli dependency).

Usage:
    python download_dandi.py                     # Interactive mode
    python download_dandi.py --dandiset 000636   # Download specific dandiset
    python download_dandi.py --dandiset 000636 --max-files 5   # Limit files
    python download_dandi.py --dandiset 000636 --subject sub-731978186  # Filter by subject
"""

import argparse
import os
import sys
import time
import requests
from pathlib import Path


DANDI_API = "https://api.dandiarchive.org/api"


def get_dandiset_info(dandiset_id: str) -> dict:
    """Fetch dandiset metadata."""
    r = requests.get(f"{DANDI_API}/dandisets/{dandiset_id}/")
    r.raise_for_status()
    return r.json()


def list_assets(dandiset_id: str, version: str = "draft", page_size: int = 100):
    """Yield all assets in a dandiset (handles pagination)."""
    url = f"{DANDI_API}/dandisets/{dandiset_id}/versions/{version}/assets/"
    params = {"page_size": page_size}

    while url:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        yield from data.get("results", [])
        url = data.get("next")
        params = {}  # next URL already includes params


def get_download_url(dandiset_id: str, asset_id: str, version: str = "draft") -> str:
    """Get the S3 download URL for an asset."""
    r = requests.get(
        f"{DANDI_API}/dandisets/{dandiset_id}/versions/{version}/assets/{asset_id}/download/",
        allow_redirects=False,
    )
    if r.status_code in (301, 302):
        return r.headers["Location"]
    r.raise_for_status()
    return r.url


def download_file(url: str, dest: Path, expected_size: int = 0):
    """Download a file with progress display."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded and correct size
    if dest.exists() and expected_size > 0 and dest.stat().st_size == expected_size:
        print(f"  [skip] Already downloaded: {dest.name}")
        return

    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", expected_size))

    downloaded = 0
    start = time.time()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192 * 16):  # 128 KB chunks
            f.write(chunk)
            downloaded += len(chunk)
            elapsed = time.time() - start
            speed = downloaded / max(elapsed, 0.01)
            pct = (downloaded / total * 100) if total else 0
            mb_done = downloaded / 1e6
            mb_total = total / 1e6
            print(
                f"\r  [{pct:5.1f}%] {mb_done:.1f}/{mb_total:.1f} MB  "
                f"({speed/1e6:.1f} MB/s)    ",
                end="",
                flush=True,
            )
    print()  # newline after progress


def main():
    parser = argparse.ArgumentParser(description="Download NWB files from DANDI Archive")
    parser.add_argument("--dandiset", type=str, help="Dandiset ID (e.g., 000636)")
    parser.add_argument("--version", type=str, default="draft", help="Version (default: draft)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: ./dandi_<id>)")
    parser.add_argument("--max-files", type=int, default=0, help="Max files to download (0 = all)")
    parser.add_argument("--subject", type=str, default=None, help="Filter by subject ID (e.g., sub-731978186)")
    parser.add_argument("--list-only", action="store_true", help="List assets without downloading")
    args = parser.parse_args()

    # Interactive mode if no dandiset provided
    if not args.dandiset:
        args.dandiset = input("Enter DANDI dataset ID (e.g., 000636): ").strip()
        if not args.dandiset:
            print("No dataset ID provided. Exiting.")
            sys.exit(1)

    dandiset_id = args.dandiset.lstrip("0") and args.dandiset  # keep leading zeros
    print(f"\n{'='*60}")
    print(f"DANDI Downloader — Dandiset {dandiset_id}")
    print(f"{'='*60}")

    # Fetch info
    try:
        info = get_dandiset_info(dandiset_id)
        name = info.get("draft_version", {}).get("name", "Unknown")
        print(f"Name: {name}")
    except requests.HTTPError as e:
        print(f"ERROR: Could not find dandiset {dandiset_id} ({e})")
        sys.exit(1)

    # Collect assets
    print(f"\nFetching asset list...")
    assets = []
    for asset in list_assets(dandiset_id, args.version):
        path = asset.get("path", "")
        # Filter by subject if specified
        if args.subject and not path.startswith(args.subject):
            continue
        # Only NWB files
        if path.endswith(".nwb"):
            assets.append(asset)

    print(f"Found {len(assets)} NWB file(s)")

    if not assets:
        print("No matching NWB files found.")
        sys.exit(0)

    # Show preview
    print(f"\n{'Path':<65} {'Size':>10}")
    print("-" * 77)
    total_size = 0
    show_assets = assets[: min(20, len(assets))]
    for a in show_assets:
        sz = a.get("size", 0)
        total_size += sz
        mb = sz / 1e6
        print(f"  {a['path']:<63} {mb:>7.1f} MB")
    remaining = len(assets) - len(show_assets)
    if remaining > 0:
        # Sum remaining sizes
        for a in assets[len(show_assets):]:
            total_size += a.get("size", 0)
        print(f"  ... and {remaining} more files")

    total_gb = total_size / 1e9
    print(f"\nTotal size: {total_gb:.2f} GB")

    if args.list_only:
        sys.exit(0)

    # Apply max-files limit
    if args.max_files > 0:
        assets = assets[: args.max_files]
        print(f"\nLimited to first {args.max_files} file(s)")

    # Confirm download
    if not args.max_files and len(assets) > 5:
        confirm = input(f"\nDownload {len(assets)} files ({total_gb:.2f} GB)? (y/n): ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            sys.exit(0)

    # Output directory
    out_dir = Path(args.output) if args.output else Path(f"dandi_{dandiset_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading to: {out_dir.resolve()}\n")

    # Download
    success = 0
    failed = 0
    for i, asset in enumerate(assets, 1):
        path = asset["path"]
        size = asset.get("size", 0)
        asset_id = asset["asset_id"]
        dest = out_dir / path

        print(f"[{i}/{len(assets)}] {path} ({size/1e6:.1f} MB)")

        try:
            dl_url = get_download_url(dandiset_id, asset_id, args.version)
            download_file(dl_url, dest, expected_size=size)
            success += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Done! {success} downloaded, {failed} failed")
    print(f"Location: {out_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
