#!/usr/bin/env python3
"""VisionGuard: Project Report Compiler.

Compiles report/project_report.tex into report/project_report.pdf using pdflatex.
"""

from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    report_dir = Path(__file__).resolve().parent.parent / "report"
    tex_file = report_dir / "project_report.tex"
    pdf_file = report_dir / "project_report.pdf"

    if not tex_file.is_file():
        print(f"[ERROR] LaTeX source file not found: {tex_file}", file=sys.stderr)
        return 1

    print("=" * 78)
    print("  Compiling Academic Project Report (report/project_report.pdf)")
    print("=" * 78)

    pdflatex_bin = shutil.which("pdflatex")
    if pdflatex_bin:
        print(f"[*] Found pdflatex at: {pdflatex_bin}")
        try:
            # Run pdflatex twice for citations and references
            for pass_num in (1, 2):
                print(f"[*] Executing LaTeX pass {pass_num}...")
                res = subprocess.run(
                    [
                        pdflatex_bin,
                        "-interaction=nonstopmode",
                        "-output-directory",
                        str(report_dir),
                        str(tex_file),
                    ],
                    capture_output=True,
                    text=True,
                )
                if res.returncode != 0 and not pdf_file.is_file():
                    print(f"[ERROR] pdflatex failed:\n{res.stdout[-1500:]}", file=sys.stderr)
                    return 1

            # Clean temporary build files
            for ext in (".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk"):
                for tmp in report_dir.glob(f"*{ext}"):
                    try:
                        tmp.unlink()
                    except OSError:
                        pass

            print(f"\n[SUCCESS] Compiled academic report: {pdf_file} ({pdf_file.stat().st_size / 1024:.1f} KB)")
            return 0
        except Exception as e:
            print(f"[ERROR] Compilation error: {e}", file=sys.stderr)
            return 1
    else:
        print("[ERROR] pdflatex compiler not found in PATH.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
