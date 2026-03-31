import argparse
import os
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]


def write_xxd_style_header(input_path: Path, output_path: Path, symbol_name: str) -> None:
    data = input_path.read_bytes()
    lines = [f"unsigned char {symbol_name}[] = {{\n"]
    for start in range(0, len(data), 12):
        chunk = data[start : start + 12]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk))
        if start + 12 < len(data):
            lines.append(",\n")
        else:
            lines.append("\n")
    lines.append("};\n")
    lines.append(f"unsigned int {symbol_name}_len = {len(data)};\n")
    output_path.write_text("".join(lines), encoding="ascii", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 OpenLA500 风格的部署头文件资产。")
    parser.add_argument("--model-bin", required=True, help="输入模型二进制，例如 stories260K_q80.bin")
    parser.add_argument("--tokenizer-bin", required=True, help="输入 tokenizer 二进制，例如 tok512.bin")
    parser.add_argument("--output-dir", default=str(REPO_DIR / "deploy" / "assets" / "stories260K_qat_best"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_xxd_style_header(Path(args.model_bin), output_dir / "stories_data.h", "stories260K_q80_bin")
    write_xxd_style_header(Path(args.tokenizer_bin), output_dir / "tok512.h", "tok512_bin")
    print(f"写出部署资产到: {output_dir}")


if __name__ == "__main__":
    main()
