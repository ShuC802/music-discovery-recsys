"""Remove local machine details from notebook outputs before publishing.

Drops stderr streams (library warnings, which print local install paths) and any
remaining output line that contains a home-directory path.

    uv run python scripts/scrub_notebooks.py
"""
import json
import re
from pathlib import Path

PATH_PATTERN = re.compile(r"([A-Za-z]:\\\\?Users\\\\?|/Users/|/home/|/sessions/|AppData)", re.IGNORECASE)


def scrub(path: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    removed = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        kept = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                removed += 1
                continue
            for key in ("text",):
                if key in out:
                    lines = out[key] if isinstance(out[key], list) else out[key].splitlines(True)
                    clean = [ln for ln in lines if not PATH_PATTERN.search(ln)]
                    removed += len(lines) - len(clean)
                    out[key] = clean
            data = out.get("data", {})
            if "text/plain" in data:
                lines = data["text/plain"] if isinstance(data["text/plain"], list) else data["text/plain"].splitlines(True)
                clean = [ln for ln in lines if not PATH_PATTERN.search(ln)]
                removed += len(lines) - len(clean)
                data["text/plain"] = clean
            kept.append(out)
        cell["outputs"] = kept
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return removed


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for nb_path in sorted(root.glob("notebooks/**/*.ipynb")):
        n = scrub(nb_path)
        if n:
            print(f"{nb_path.relative_to(root)}: removed {n} output item(s)")
