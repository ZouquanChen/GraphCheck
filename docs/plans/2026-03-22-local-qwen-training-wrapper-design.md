# Local Qwen Training Wrapper Design

**Goal:** Make `run_training.py` default to a locally downloaded Qwen 2.5 7B model so the user can launch training without repeating model flags on every run.

**Context**

- `run_training.py` already prepares `dataset/extracted_KG`, runs `graph_build.py` when needed, and then calls `train.py`.
- Today it forwards model arguments only when the user provides them manually.
- The user has a local model at `E:\HFModels\Qwen2.5-7B-Instruct` and wants that path baked into the wrapper.

**Recommended Approach**

- Keep the change scoped to `run_training.py`.
- Add default constants for `qwen_7b` and the local model path.
- Inject those flags only when the caller did not explicitly pass `--llm_model_name` or `--llm_model_path`.

**Why This Approach**

- It preserves the wrapper's convenience goal without changing global defaults used by `train.py`, `inference.py`, or other entrypoints.
- It keeps command-line overrides working, so the script still supports experiments with other models.
- It avoids surprising side effects in `src/config.py`, where changing defaults would affect the whole project.

**Data Flow**

1. Parse wrapper-specific arguments and collect passthrough training arguments.
2. Ensure `dataset/extracted_KG/<dataset>/<dataset>.pkl` exists.
3. Run preprocessing when graph artifacts are missing.
4. Build the `train.py` command.
5. If no explicit model flags were provided, append:
   - `--llm_model_name qwen_7b`
   - `--llm_model_path E:\HFModels\Qwen2.5-7B-Instruct`
6. Execute training.

**Validation**

- Add a unit test that verifies default local Qwen arguments are injected.
- Keep the existing test that confirms explicit CLI model arguments are preserved.
- Run the targeted unit tests and a dry-run smoke test of `run_training.py`.
