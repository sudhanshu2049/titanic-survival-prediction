"""Extract embedded PNG outputs from a Jupyter notebook into `results/`.

Usage (from repo root):
    python tools\extract_notebook_images.py modelprojectibm.ipynb

It writes files named `results/notebook_cell-<i>_out-<j>.png` for each image found.
"""
import sys
import json
from pathlib import Path
import base64


def extract_images(nb_path: Path, out_dir: Path) -> int:
    nb = json.loads(nb_path.read_text(encoding='utf8'))
    images_saved = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    for ci, cell in enumerate(nb.get('cells', [])):
        for oi, output in enumerate(cell.get('outputs', [])):
            data = output.get('data') or {}
            img_b64 = data.get('image/png')
            if img_b64:
                # Some notebooks store image/png as list of strings
                if isinstance(img_b64, list):
                    img_b64 = ''.join(img_b64)
                img_data = base64.b64decode(img_b64)
                fname = out_dir / f'notebook_cell-{ci}_out-{oi}.png'
                with open(fname, 'wb') as f:
                    f.write(img_data)
                print('Wrote', fname)
                images_saved += 1
    return images_saved


def main():
    if len(sys.argv) < 2:
        print('Usage: python tools\\extract_notebook_images.py <notebook.ipynb>')
        raise SystemExit(1)
    nb_path = Path(sys.argv[1]).resolve()
    if not nb_path.exists():
        print('Notebook not found:', nb_path)
        raise SystemExit(2)
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / 'results'
    count = extract_images(nb_path, out_dir)
    print(f'Extracted {count} image(s) to {out_dir}')


if __name__ == '__main__':
    main()
