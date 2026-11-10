"""Show (1) mtime and (2) unified diff of external/SKVQ/quant_parser.py vs
the git HEAD blob SKVQ/quant_parser.py."""
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
F = ROOT / "external" / "SKVQ" / "quant_parser.py"

st = os.stat(F)
print("mtime:", st.st_mtime_size if False else st.st_size, "bytes,", end=" ")
import time
print(time.strftime("%F %T", time.localtime(st.st_mtime)))

head = subprocess.run(
    ["git", "show", "HEAD:SKVQ/quant_parser.py"],
    cwd=ROOT, capture_output=True, text=True,
).stdout.splitlines(keepends=True)
disk = F.read_text().splitlines(keepends=True)

import difflib
sys_diff = list(difflib.unified_diff(
    head, disk, fromfile="HEAD:SKVQ/quant_parser.py",
    tofile="external/SKVQ/quant_parser.py", n=3))
print("".join(sys_diff) if sys_diff else "no textual diff (mode/line-ending?)")
