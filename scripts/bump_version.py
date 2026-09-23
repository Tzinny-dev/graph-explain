#!/usr/bin/env python3
"""Bump graph-explain versions and finalize CHANGELOG.md releases."""
from __future__ import annotations
import argparse, re, subprocess, sys
from datetime import date
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
INIT_PY = ROOT / "src" / "graph_explain" / "__init__.py"
CHANGELOG = ROOT / "CHANGELOG.md"

def _re(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.MULTILINE)

_VERSION_TOML = _re('^version\\s*=\\s*[\\"\']([^\\"\']+)[\\"\']')
_VERSION_INIT = _re('^__version__\\s*=\\s*[\\"\']([^\\"\']+)[\\"\']')
_SECTION = _re('^## \\[([^\\]]+)\\](?:\\s*-\\s*([^\\n]*))?')
_UNRELEASED = _re('^## \\[Unreleased\\]')

class Version(tuple):
    """Semver X.Y.Z with bump helpers."""
    __slots__ = ()
    def __new__(cls, major, minor, patch):
        return super().__new__(cls, (int(major), int(minor), int(patch)))
    @property
    def major(self) -> int: return self[0]
    @property
    def minor(self) -> int: return self[1]
    @property
    def patch(self) -> int: return self[2]
    def __str__(self) -> str: return f"{self.major}.{self.minor}.{self.patch}"
    def __repr__(self) -> str: return f"Version({self.major}, {self.minor}, {self.patch})"
    def __eq__(self, o) -> bool:
        if not isinstance(o, Version): return NotImplemented
        return (self.major, self.minor, self.patch) == (o.major, o.minor, o.patch)
    def __hash__(self) -> int: return hash((self.major, self.minor, self.patch))
    @classmethod
    def parse(cls, s: str) -> Version:
        parts = s.strip().split(".")
        if len(parts) != 3: raise ValueError(f"expected semver X.Y.Z, got {s!r}")
        try: return cls(*[int(p.strip()) for p in parts])
        except ValueError as exc: raise ValueError(f"expected semver X.Y.Z, got {s!r}") from exc
    def bump_patch(self) -> Version: return Version(self.major, self.minor, self.patch + 1)
    def bump_minor(self) -> Version: return Version(self.major, self.minor + 1, 0)
    def bump_major(self) -> Version: return Version(self.major + 1, 0, 0)

def read_current() -> Version:
    t = PYPROJECT.read_text(encoding="utf-8")
    m = _VERSION_TOML.search(t)
    if not m: raise RuntimeError("could not read version from pyproject.toml")
    return Version.parse(m.group(1))

def changelog_sections() -> dict[str, str]:
    t = CHANGELOG.read_text(encoding="utf-8")
    secs = {}; matches = list(_SECTION.finditer(t))
    for i, m in enumerate(matches):
        key = m.group(1); end = matches[i + 1].start() if i + 1 < len(matches) else len(t)
        secs[key] = t[m.end():end].strip()
    return secs

def latest_version_in_changelog() -> Version | None:
    sections = changelog_sections(); cand = []
    for key in sections:
        if key.lower() == "unreleased": continue
        try: cand.append(Version.parse(key))
        except ValueError: continue
    return max(cand, key=lambda v: (v.major, v.minor, v.patch)) if cand else None

def has_unreleased_section() -> bool:
    return bool(_UNRELEASED.search(CHANGELOG.read_text(encoding="utf-8")))

def git_log_between(start_ref: str, end_ref: str = "HEAD") -> list[str]:
    proc = subprocess.run(["git", "log", f"{start_ref}..{end_ref}", "--pretty=format:%s"], cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0: raise RuntimeError(f"git log failed: {proc.stderr.strip()}")
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]

def git_tag_exists(tag: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", tag], cwd=ROOT, capture_output=True).returncode == 0

def latest_semver_tag() -> str | None:
    proc = subprocess.run(["git", "tag", "-l", "--sort=-version:refname", "v*"], cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0: return None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line: continue
        try: Version.parse(line[1:]); return line
        except ValueError: continue
    return None

def is_breaking(subject: str) -> bool:
    s = subject.lower(); return s.startswith("breaking:") or "breaking change:" in s

def is_feature(subject: str) -> bool:
    s = subject.lower().lstrip()
    return s.startswith("feat!") or s.startswith("feat(") or s.startswith("feat:")

def compute_next(*, branch_sense: bool = True) -> Version:
    if not branch_sense: return read_current().bump_patch()
    tag = latest_semver_tag()
    if tag is None: return Version(0, 1, 0)
    commits = git_log_between(tag, "HEAD")
    has_breaking = any(is_breaking(c) for c in commits)
    has_feature = any(is_feature(c) for c in commits)
    cur = read_current()
    if has_breaking: return cur.bump_major()
    if has_feature: return cur.bump_minor()
    return cur.bump_patch()

def bump_files(v: Version) -> list[Path]:
    updated = []
    for path, regex, replacement in [
        (PYPROJECT, _VERSION_TOML, f"version = \"" + str(v) + "\""),
        (INIT_PY, _VERSION_INIT, f"__version__ = \"" + str(v) + "\""),
    ]:
        text = path.read_text(encoding="utf-8")
        new, n = regex.subn(replacement, text, count=1)
        if n != 1: raise RuntimeError(f"failed to update {path}")
        path.write_text(new, encoding="utf-8"); updated.append(path)
    return updated

def finalize_changelog(v: Version) -> Path:
    if not has_unreleased_section(): raise RuntimeError("CHANGELOG.md has no [Unreleased] section")
    sections = changelog_sections()
    unreleased_body = sections.pop("Unreleased", "").strip()
    if not unreleased_body: raise RuntimeError("[Unreleased] section is empty")
    header = f"## [{v}] - {date.today().isoformat()}"
    text = CHANGELOG.read_text(encoding="utf-8")
    m = _UNRELEASED.search(text); assert m
    # *intro* = everything before the [Unreleased] header (changelog title,
    # intro paragraph, etc.).  In Keep-a-Changelog layout [Unreleased] sits
    # at the very top, right after the intro.
    intro = text[:m.start()].rstrip()
    after = text[m.end():].lstrip("\n")
    # *body_text* = content that belongs to the released version (what was
    # under [Unreleased]).  *rest* = the already-released sections below.
    next_header = re.search(r"^## \[", after, re.MULTILINE)
    if next_header:
        body_text = after[:next_header.start()].rstrip()
        rest = after[next_header.start():].rstrip()
    else:
        body_text = after.rstrip(); rest = ""
    # Re-assemble: intro → new [Unreleased] (empty) → released section → rest
    new_text = f"{intro}\n\n## [Unreleased]\n"
    if body_text:
        new_text += f"\n{body_text}\n"
    new_text += f"\n## [{v}] - {date.today().isoformat()}\n\n{rest}\n"
    new_text = new_text.strip() + "\n"
    CHANGELOG.write_text(new_text, encoding="utf-8")
    return CHANGELOG

def release_body(v: Version) -> str:
    body = changelog_sections().get(str(v), "").strip()
    if not body:
        repo = __import__("os").environ.get("GITHUB_REPOSITORY", "Tzinny-dev/graph-explain")
        return f"Release v{v}\n\nSee [CHANGELOG.md](https://github.com/{repo}/blob/main/CHANGELOG.md)."
    return body

def git_commit(message: str, files: list[Path] | None = None) -> subprocess.CompletedProcess[str]:
    cmd = ["git", "commit", "-m", message]
    if files: cmd.extend(str(p) for p in files)
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)

def git_tag(version: Version) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "tag", "-a", f"v{version}", "-m", f"Release v{version}"], cwd=ROOT, capture_output=True, text=True)

def cmd(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bump_version.py")
    ap.add_argument("action", nargs="?", choices=["query-version", "bump", "release-body"])
    ap.add_argument("--version", type=Version.parse)
    ap.add_argument("--bump", action="store_true")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--no-branch-sense", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.action == "bump" or args.commit:
        args.bump = True
    v = args.version if args.version else compute_next(branch_sense=not args.no_branch_sense)
    if args.action == "query-version": print(read_current()); return 0
    if args.action == "release-body": print(release_body(v)); return 0
    if args.bump:
        if args.dry_run:
            print(f"[DRY RUN] would bump to {v}"); print("[DRY RUN] files: pyproject.toml, __init__.py, CHANGELOG.md")
        else:
            files = bump_files(v); finalize_changelog(v); files.append(CHANGELOG)
            print(f"bumped to {v}"); print(f"edited: " + ", ".join(str(p) for p in files))
            if args.commit:
                c = git_commit(f"chore: bump to v{v}", files)
                if c.returncode != 0: print(c.stderr, file=sys.stderr); return 1
                t = git_tag(v)
                if t.returncode != 0: print(t.stderr, file=sys.stderr); return 1
                print(f"committed and tagged v{v}")
    return 0

if __name__ == "__main__": raise SystemExit(cmd())
