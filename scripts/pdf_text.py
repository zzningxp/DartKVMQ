"""Extract text from a paper PDF. Usage:
  python scripts/pdf_text.py <pdf> <start> <end> [out.txt]
Prints to stdout if no out path given.
"""
import sys

from pypdf import PdfReader

pdf, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = sys.argv[4] if len(sys.argv) > 4 else None
r = PdfReader(pdf)
chunks = []
for i in range(start - 1, min(end, len(r.pages))):
    chunks.append(f"\n===== PAGE {i + 1} =====\n" + (r.pages[i].extract_text() or ""))
text = "\n".join(chunks)
if out:
    with open(out, "w") as f:
        f.write(text)
    print(f"wrote {len(text)} chars to {out}")
else:
    print(text)
