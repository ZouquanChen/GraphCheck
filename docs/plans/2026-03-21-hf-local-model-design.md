# HuggingFace Local Model Override Design

**Goal:** Keep the original project model names and defaults, while allowing a user-provided local HuggingFace model directory to override the mapped repo id.

## Current Behavior

- `train.py` and `inference.py` always overwrite `args.llm_model_path` from `model.model_path[args.llm_model_name]`.
- This prevents using a downloaded local HuggingFace directory even though `src/config.py` already exposes `--llm_model_path`.
- `model/graphcheck.py` expects a standard HuggingFace `from_pretrained(...)` path or repo id.

## Design

- Keep `llm_model_name` restricted to the original project keys in `model/__init__.py`.
- Add a single resolver helper in `model/__init__.py` that returns:
  - `args.llm_model_path` when the user passes a non-empty local override path
  - otherwise the original mapped HuggingFace repo id from `model_path`
- Update both `train.py` and `inference.py` to use the shared resolver instead of duplicating direct assignment.
- Update CLI help text to make it clear that `--llm_model_path` is an optional local HuggingFace directory override.

## Non-Goals

- No support for GGUF, LM Studio, or llama.cpp loaders.
- No expansion of the allowed model names beyond the original project set.
- No change to training or inference logic beyond model path resolution.

## Validation

- Add a small unit test suite for path resolution.
- Verify the project still resolves the original repo ids when no local path is provided.
- Verify a provided local path wins over the default mapping.
