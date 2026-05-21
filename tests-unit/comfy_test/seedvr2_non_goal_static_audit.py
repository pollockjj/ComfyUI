import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_FILES = {
    "comfy/ldm/seedvr/model.py",
    "comfy/ldm/modules/attention.py",
    "comfy/sample.py",
    "comfy/samplers.py",
}


def main():
    diff = subprocess.check_output(
        ["git", "-C", str(ROOT), "diff", "--name-only", "issue_101"],
        text=True,
    ).splitlines()
    changed_forbidden = sorted(FORBIDDEN_FILES.intersection(diff))
    if changed_forbidden:
        raise SystemExit(f"forbidden non-goal files changed: {changed_forbidden}")
    print("PASS seedvr2 non-goal static audit")


if __name__ == "__main__":
    main()
