# Local Qwen Training Wrapper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update `run_training.py` so it defaults to the local Qwen 2.5 7B model at `E:\HFModels\Qwen2.5-7B-Instruct` while still allowing explicit CLI overrides.

**Architecture:** Keep the behavior isolated to the wrapper script. The wrapper will add default `--llm_model_name` and `--llm_model_path` flags only when they are absent from passthrough arguments, leaving `train.py` and other entrypoints unchanged.

**Tech Stack:** Python, `argparse`, `unittest`, subprocess-based CLI wrapper

---

### Task 1: Add a failing test for default model injection

**Files:**
- Modify: `tests/test_run_training.py`
- Test: `tests/test_run_training.py`

**Step 1: Write the failing test**

```python
def test_build_train_command_uses_local_qwen_defaults_when_unspecified(self):
    from run_training import build_train_command

    root = Path(r"E:/demo/GraphCheck")
    command = build_train_command(root, "MiniCheck_Train", ["--project", "GraphCheck_Debug"])

    self.assertEqual(
        command,
        [
            sys.executable,
            str(root / "train.py"),
            "--train_dataset",
            "MiniCheck_Train",
            "--llm_model_name",
            "qwen_7b",
            "--llm_model_path",
            r"E:\HFModels\Qwen2.5-7B-Instruct",
            "--project",
            "GraphCheck_Debug",
        ],
    )
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_run_training.RunTrainingTests.test_build_train_command_uses_local_qwen_defaults_when_unspecified -v`
Expected: FAIL because `build_train_command()` does not yet inject default model flags.

**Step 3: Write minimal implementation**

Update `run_training.py` to:

```python
DEFAULT_LLM_MODEL_NAME = "qwen_7b"
DEFAULT_LLM_MODEL_PATH = r"E:\HFModels\Qwen2.5-7B-Instruct"
```

and append those args inside `build_train_command()` only when the caller did not pass `--llm_model_name` or `--llm_model_path`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_run_training.RunTrainingTests.test_build_train_command_uses_local_qwen_defaults_when_unspecified -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_run_training.py run_training.py
git commit -m "feat: default training wrapper to local qwen model"
```

### Task 2: Preserve explicit CLI overrides

**Files:**
- Modify: `tests/test_run_training.py`
- Modify: `run_training.py`
- Test: `tests/test_run_training.py`

**Step 1: Keep the existing override test as the guardrail**

```python
def test_build_train_command_forwards_extra_args(self):
    ...
```

**Step 2: Run test to verify override behavior still holds**

Run: `python -m unittest tests.test_run_training.RunTrainingTests.test_build_train_command_forwards_extra_args -v`
Expected: PASS after the implementation change.

**Step 3: Adjust implementation only if needed**

If the test fails, detect explicit `--llm_model_name` and `--llm_model_path` in passthrough args and avoid injecting defaults for the flags the user already set.

**Step 4: Run both model command tests together**

Run: `python -m unittest tests.test_run_training -v`
Expected: PASS with all tests green.

**Step 5: Commit**

```bash
git add tests/test_run_training.py run_training.py
git commit -m "test: preserve model overrides in training wrapper"
```

### Task 3: Verify wrapper behavior end-to-end

**Files:**
- Modify: `run_training.py` (only if smoke test exposes an issue)

**Step 1: Run syntax verification**

Run: `python -m py_compile run_training.py`
Expected: exit code 0

**Step 2: Run a dry-run smoke test**

Run: `python run_training.py --train_dataset MiniCheck_Train --prepare-only --dry-run`
Expected: printed `train.py` command includes `--llm_model_name qwen_7b` and `--llm_model_path E:\HFModels\Qwen2.5-7B-Instruct` when training is launched; prepare-only path should still complete cleanly.

**Step 3: Run a full dry-run train command**

Run: `python run_training.py --train_dataset MiniCheck_Train --dry-run --project GraphCheck_Qwen_Debug`
Expected: printed command targets `train.py`, includes the local Qwen defaults, and preserves the custom project name.

**Step 4: Fix only issues revealed by verification**

Keep changes limited to argument construction and messaging.

**Step 5: Commit**

```bash
git add run_training.py tests/test_run_training.py
git commit -m "chore: verify local qwen training wrapper flow"
```
