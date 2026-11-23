"""Stage 2: SoulX-Singer fine-tuning for Hindi (adapter, phone embeddings, data, trainer).

Runs in the SoulX venv (``.venv-sx``: py3.10, torch 2.2, transformers 4.41.2).
The repo and the SoulX clone are on that venv's path through a ``.pth`` file
(``kalavinka/ops/setup_envs.sh``), so ``saregama.phoneset`` and ``soulxsinger``
both import.
"""
