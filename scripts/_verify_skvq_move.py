"""Verify that external/SKVQ is a byte-identical copy of HEAD's SKVQ/ tree.

Compares git blob SHAs (git ls-tree HEAD SKVQ) with git hash-object of every
file under external/SKVQ. Run from repo root.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXT = ROOT / "external" / "SKVQ"


def run(*args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout


head = {}
for line in run("git", "ls-tree", "-r", "HEAD", "--", "SKVQ").splitlines():
    meta, path = line.split("\t")
    sha = meta.split()[2]
    head[path.split("/", 1)[1]] = sha

disk = {}
for p in EXT.rglob("*"):
    if p.is_file():
        rel = str(p.relative_to(EXT))
        disk[rel] = run("git", "hash-object", str(p)).strip()

missing = sorted(set(head) - set(disk))
extra = sorted(set(disk) - set(head))
diff = sorted(k for k in set(head) & set(disk) if head[k] != disk[k])

print(f"HEAD SKVQ/ files : {len(head)}")
print(f"external/SKVQ    : {len(disk)}")
print(f"missing on disk  : {missing}")
print(f"extra on disk    : {extra}")
print(f"content differs  : {diff}")
ok = not missing and not extra and not diff
print("VERDICT:", "identical copy" if ok else "MISMATCH — see above")
sys.exit(0 if ok else 1)
