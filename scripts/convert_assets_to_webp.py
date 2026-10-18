"""Convert raster images under assets/ to lossless WebP and update content links.

Run from any directory: python scripts/convert_assets_to_webp.py [--dry-run].
SVG and WebP files are left alone; each original is deleted only after its WebP verifies.
"""

import argparse
from pathlib import Path
from urllib.parse import quote

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
CONTENT = ROOT / "content"
RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"}


def _convert(source: Path) -> Path:
	"""Write a verified lossless WebP next to the source and remove the source."""
	target = source.with_suffix(".webp")
	if target.exists():
		raise FileExistsError(f"Refusing to overwrite {target.relative_to(ROOT)}")
	with Image.open(source) as image:
		image.save(target, "WEBP", lossless=True, method=6)
	with Image.open(target) as check:
		check.verify()
	source.unlink()
	return target


def _rewrite_links(renames: dict[str, str]) -> list[Path]:
	"""Replace raw and URL-encoded asset paths in content Markdown."""
	pairs = []
	for old, new in renames.items():
		pairs.append((old, new))
		if quote(old) != old:
			pairs.append((quote(old), quote(new)))
	changed = []
	for document in CONTENT.rglob("*.md"):
		text = document.read_text(encoding="utf-8")
		updated = text
		for old, new in pairs:
			updated = updated.replace(old, new)
		if updated != text:
			document.write_text(updated, encoding="utf-8")
			changed.append(document)
	return changed


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--dry-run", action="store_true", help="list files without converting")
	args = parser.parse_args()

	sources = sorted(p for p in ASSETS.rglob("*") if p.is_file() and p.suffix.lower() in RASTER_SUFFIXES)
	if not sources:
		print("No raster images to convert.")
		return

	renames = {}
	for source in sources:
		relative = source.relative_to(ROOT)
		if args.dry_run:
			print(f"would convert {relative}")
			continue
		size = source.stat().st_size
		target = _convert(source)
		print(f"{relative}: {size / 1e6:.2f} MB -> {target.stat().st_size / 1e6:.2f} MB")
		renames[relative.as_posix()] = target.relative_to(ROOT).as_posix()

	for document in _rewrite_links(renames):
		print(f"updated links in {document.relative_to(ROOT)}")


if __name__ == "__main__":
	main()
