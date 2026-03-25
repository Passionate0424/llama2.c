import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "stories260K"
HEADER_DIR = ROOT / "embedded" / "stories260K"
DOWNLOAD_ROOT = "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K"
DOWNLOADS = ("stories260K.pt", "tok512.model")
Q80_BIN = "stories260K_q80.bin"
TOKENIZER_BIN = "tok512.bin"
Q80_GROUP_SIZE = "32"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embedded 260K runq assets.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to run export.py and tokenizer.py.",
    )
    parser.add_argument(
        "--xxd",
        default=None,
        help="Optional path to xxd.exe. Defaults to PATH or a few known Git for Windows locations.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-generate all artifacts.",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, target: Path, force: bool):
    if target.exists() and not force:
        print(f"using cached {target.name}")
        return
    print(f"downloading {url}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def run_checked(cmd, cwd: Path):
    printable = " ".join(f'"{part}"' if " " in part else part for part in cmd)
    print(f"running: {printable}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def find_xxd(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    which = shutil.which("xxd")
    if which:
        candidates.append(Path(which))
    candidates.extend(
        [
            Path(r"D:\Git\usr\bin\xxd.exe"),
            Path(r"C:\Program Files\Git\usr\bin\xxd.exe"),
            Path(r"C:\Program Files (x86)\Git\usr\bin\xxd.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "xxd was not found. Install Git for Windows or pass --xxd <path-to-xxd.exe>."
    )


def header_guard(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", name).upper()
    return f"LLAMA2C_EMBEDDED_{stem}_H"


def wrap_xxd_output(header_name: str, xxd_output: str) -> str:
    body = xxd_output.replace("unsigned char ", "static const unsigned char ", 1)
    body = body.replace("unsigned int ", "static const unsigned int ", 1)
    guard = header_guard(header_name)
    return (
        f"#ifndef {guard}\n"
        f"#define {guard}\n\n"
        f"{body}\n"
        f"#endif\n"
    )


def emit_header(xxd: Path, source_dir: Path, source_name: str, header_name: str):
    result = subprocess.run(
        [str(xxd), "-i", source_name],
        cwd=str(source_dir),
        check=True,
        capture_output=True,
        text=True,
    )
    header_text = wrap_xxd_output(header_name, result.stdout.strip())
    header_path = HEADER_DIR / header_name
    header_path.write_text(header_text, encoding="ascii", newline="\n")
    print(f"wrote {header_path.relative_to(ROOT)}")


def main():
    args = parse_args()
    ensure_dir(ARTIFACT_DIR)
    ensure_dir(HEADER_DIR)

    xxd = find_xxd(args.xxd)
    python = Path(args.python)
    if not python.exists():
        raise FileNotFoundError(f"python interpreter not found: {python}")

    for filename in DOWNLOADS:
        download_file(f"{DOWNLOAD_ROOT}/{filename}", ARTIFACT_DIR / filename, args.force)

    q80_path = ARTIFACT_DIR / Q80_BIN
    if args.force and q80_path.exists():
        q80_path.unlink()
    if not q80_path.exists():
        run_checked(
            [
                str(python),
                str(ROOT / "export.py"),
                str(q80_path),
                "--version",
                "2",
                "--group-size",
                Q80_GROUP_SIZE,
                "--checkpoint",
                str(ARTIFACT_DIR / "stories260K.pt"),
            ],
            ROOT,
        )

    tokenizer_bin_path = ARTIFACT_DIR / TOKENIZER_BIN
    if args.force and tokenizer_bin_path.exists():
        tokenizer_bin_path.unlink()
    if not tokenizer_bin_path.exists():
        run_checked(
            [
                str(python),
                str(ROOT / "tokenizer.py"),
                "--tokenizer-model",
                str(ARTIFACT_DIR / "tok512.model"),
            ],
            ROOT,
        )
        generated = ARTIFACT_DIR / TOKENIZER_BIN
        if not generated.exists():
            raise FileNotFoundError(f"tokenizer.py did not create {generated}")

    emit_header(xxd, ARTIFACT_DIR, Q80_BIN, "stories260K_q80.h")
    emit_header(xxd, ARTIFACT_DIR, TOKENIZER_BIN, "tok512.h")

    print("embedded assets are ready")
    print(f"  headers: {HEADER_DIR.relative_to(ROOT)}")
    print(f"  artifacts: {ARTIFACT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
