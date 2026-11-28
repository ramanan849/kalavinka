"""The READMEs point only at things that exist, and the uv environments are complete.

The main README is the replication guide, so a renamed script or module would silently break it. These checks
read the READMEs as text:

* every repository path they mention (``kalavinka/...``, ``saregama/...``, ``envs/...``, ``docs/...``,
  ``p1_project/...``) exists;
* every module run with ``python -m kalavinka....`` / ``saregama....`` exists as a file or package;
* every relative Markdown link resolves (the ``#anchor`` part is ignored);
* both ``envs/<env>/pyproject.toml`` parse, pin torch from a PyTorch index, and have a ``uv.lock`` next to them.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = [ROOT / "README.md", ROOT / "kalavinka" / "README.md", ROOT / "kalavinka" / "ops" / "README.md",
        ROOT / "envs" / "README.md"]
PATH_RE = re.compile(r"(?<![\w/$.~-])((?:kalavinka|saregama|envs|docs|p1_project)/[\w./-]*)")
MODULE_RE = re.compile(r"-m ((?:kalavinka|saregama)(?:\.\w+)+)")
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
ENVS = ("annotate", "soulx")


def _text(path):
    """A README's text."""
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_mentioned_paths_exist(doc):
    """Every repository path written in the README exists (directories may end with ``/``)."""
    missing = sorted({m.rstrip(".") for m in PATH_RE.findall(_text(doc))
                      if not (ROOT / m.rstrip(".")).exists()})
    assert not missing, f"{doc.name} names missing paths: {missing}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_mentioned_modules_exist(doc):
    """Every ``python -m <module>`` in the README is a module or package of this repository."""
    missing = []
    for mod in set(MODULE_RE.findall(_text(doc))):
        base = ROOT.joinpath(*mod.split("."))
        if not (base.with_suffix(".py").exists() or (base / "__init__.py").exists()):
            missing.append(mod)
    assert not missing, f"{doc.name} runs missing modules: {sorted(missing)}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(doc):
    """Every relative Markdown link (not http, not a pure ``#anchor``) points at an existing file or folder."""
    bad = []
    for target in LINK_RE.findall(_text(doc)):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        if not (doc.parent / target.split("#", 1)[0]).exists():
            bad.append(target)
    assert not bad, f"{doc.name} has broken links: {bad}"


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib needs Python 3.11")
@pytest.mark.parametrize("env", ENVS)
def test_uv_envs_are_complete(env):
    """The env's pyproject parses, takes torch from an explicit PyTorch index, and is locked."""
    import tomllib
    folder = ROOT / "envs" / env
    cfg = tomllib.loads((folder / "pyproject.toml").read_text(encoding="utf-8"))
    deps = cfg["project"]["dependencies"]
    assert any(d.startswith("torch==") for d in deps), "torch must be pinned"
    assert cfg["tool"]["uv"]["sources"]["torch"]["index"].startswith("pytorch-")
    assert all(i.get("explicit") for i in cfg["tool"]["uv"]["index"])
    lock = (folder / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "torch"' in lock
    assert (ROOT / "envs" / "setup.sh").exists()
