# Vendored IFEval instruction checker (OFFICIAL, verbatim)

Source: https://github.com/google-research/google-research/tree/master/instruction_following_eval
(`instruction_following_eval/`), fetched 2026-11-10 via
`raw.githubusercontent.com` (commit on `master` at fetch time; see
`external/ifeval_official/` for the untouched originals).

Only modification vs upstream (provable by
`diff external/ifeval_official/<f>.py src/dartkv/tasks/ifeval_checker/<f>.py`):
the three intra-package import lines were rewritten from
`from instruction_following_eval import X` to `from . import X` so the files
work as the `dartkv.tasks.ifeval_checker` package. No logic touched — scoring
equivalence with the official checker is by construction.

Files:
- `instructions.py` — the 25 instruction types + `check_following` logic
- `instructions_registry.py` — INSTRUCTION_DICT id -> class
- `instructions_util.py` — word/sentence counting, language codes
- `evaluation_lib.py` — InputExample + strict/loose `test_instruction_following_*`

Deps: absl-py, langdetect, nltk (punkt + punkt_tab in ~/nltk_data),
immutabledict. Apache 2.0 (headers preserved).
