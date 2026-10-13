#!/usr/bin/env python3
"""Generic M6 ZIP packer: includes logs, CSV/JSON and failure diagnostics, never weight files."""
import argparse, pathlib, zipfile
p=argparse.ArgumentParser();p.add_argument('--run-dir',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args();root=a.run_dir.resolve();a.output.parent.mkdir(parents=True,exist_ok=True)
EXCLUDED={'.pt','.pth','.ckpt','.safetensors','.bin','.zip'}
with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for f in sorted(root.rglob('*')):
        if not f.is_file() or f.suffix.lower() in EXCLUDED: continue
        if f.stat().st_size > 100_000_000:
            print('Skipped too-large result:',f);continue
        z.write(f,arcname=f.relative_to(root).as_posix())
print('Packaged:',a.output)
