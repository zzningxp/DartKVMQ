"""Restore external/SKVQ/quant_parser.py to the pristine official version
(git HEAD blob SKVQ/quant_parser.py), undoing the import rewrite."""
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
F = ROOT / "external" / "SKVQ" / "quant_parser.py"

head = subprocess.run(
    ["git", "show", "HEAD:SKVQ/quant_parser.py"],
    cwd=ROOT, capture_output=True, text=True, check=True,
).stdout
F.write_text(head)
sha = subprocess.run(
    ["git", "hash-object", str(F)], capture_output=True, text=True, check=True
).stdout.strip()
head_sha = subprocess.run(
    ["git", "rev-parse", "HEAD:SKVQ/quant_parser.py"],
    capture_output=True, text=True, check=True,
).stdout.strip()
print("restored, hash match:", sha == head_sha)
