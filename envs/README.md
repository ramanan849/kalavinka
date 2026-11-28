# envs: the two Python environments

Kalavinka needs two environments, because SoulX-Singer pins old versions (Python 3.10, PyTorch 2.2.0,
transformers 4.41.2) while the annotation pipeline needs newer ones (Python 3.12, PyTorch 2.8, Qwen3-ASR). Each is a
small uv project with a `uv.lock`, so you get exactly the versions our reported runs used.

| Env | Folder | Installed to | Python / PyTorch | Used for |
|---|---|---|---|---|
| annotate | `envs/annotate/` | `$PROJECT/.venv-p1` | 3.12 / 2.8.0 (CUDA 12.8) | annotation pipeline (steps 4 to 6), scoring (step 10), tests |
| soulx | `envs/soulx/` | `$PROJECT/.venv-sx` | 3.10 / 2.2.0 (CUDA 12.1) | SoulX phone table, training, contour models, rendering, app zips (steps 7 to 11) |

## Install

```
curl -LsSf https://astral.sh/uv/install.sh | sh      # once, if you do not have uv
envs/setup.sh                                        # both envs (or: envs/setup.sh annotate | soulx)
```

`$PROJECT` is the repo's parent folder (set `SAREGAMA_PROJECT` to use another one). The annotate env installs this
repo as an editable package (`saregama`, `kalavinka` importable). The soulx env does not install it (`saregama`
needs Python 3.11 or newer); `setup.sh` writes a `.pth` file that puts the repo and `$PROJECT/SoulX-Singer` on its
path. The locks are for Linux with an NVIDIA GPU. To try the model on a Mac, use the app zip instead (main README).

## Updating a lock

Edit the pins in `envs/<env>/pyproject.toml`, then `cd envs/<env> && uv lock`. `constraint-dependencies` holds
every other package at the version we ran; remove an entry there to let uv choose.

The older script `kalavinka/ops/setup_envs.sh` builds the same two envs with loose pins; it is kept for reference.
