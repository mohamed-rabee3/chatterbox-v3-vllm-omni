"""Report every vendored file that diverges from the pinned upstream copy.

A divergence is only legitimate when the file carries at least one `# PORT:`
comment explaining it. Anything else is accidental drift and fails.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

UPSTREAM = Path("/workspace/repos/chatterbox/src/chatterbox/models")
VENDOR = Path(
    "/workspace/repos/vllm-omni/vllm_omni/model_executor/models/chatterbox_mtl_v3/vendor"
)


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    failures: list[str] = []
    changed: list[str] = []
    missing: list[str] = []

    for up in sorted(UPSTREAM.rglob("*.py")):
        rel = up.relative_to(UPSTREAM)
        ven = VENDOR / rel
        if not ven.exists():
            missing.append(str(rel))
            continue
        if digest(up) == digest(ven):
            continue
        changed.append(str(rel))
        if "# PORT:" not in ven.read_text(encoding="utf-8"):
            failures.append(str(rel))

    print(f"vendored files compared: {len(list(UPSTREAM.rglob('*.py')))}")
    print(f"identical: {len(list(UPSTREAM.rglob('*.py'))) - len(changed) - len(missing)}")
    print(f"intentionally modified (carry '# PORT:'): {len(changed) - len(failures)}")
    for c in changed:
        if c not in failures:
            print(f"    ~ {c}")
    if missing:
        print(f"MISSING from vendor: {missing}")
    if failures:
        print("UNEXPLAINED DRIFT (no '# PORT:' marker):")
        for f in failures:
            print(f"    ! {f}")
    return 1 if (failures or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
