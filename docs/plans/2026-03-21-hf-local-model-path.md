# HuggingFace Local Model Override Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let the project keep its original model names while supporting an optional local HuggingFace model directory override.

**Architecture:** Centralize model-path resolution in one helper inside `model/__init__.py`, then call that helper from both training and inference entrypoints. Cover the resolver with a tiny unit test suite so the behavior stays stable.

**Tech Stack:** Python, unittest, transformers, existing GraphCheck CLI entrypoints.

---

### Task 1: Add resolver tests

**Files:**
- Create: `tests/test_model_path_resolution.py`
- Modify: none
- Test: `tests/test_model_path_resolution.py`

**Step 1: Write the failing test**

```python
import unittest

from model import resolve_llm_model_path


class ResolveModelPathTests(unittest.TestCase):
    def test_returns_default_repo_id_when_override_missing(self):
        self.assertEqual(
            resolve_llm_model_path("qwen_7b", ""),
            "Qwen/Qwen2.5-7B-Instruct",
        )

    def test_prefers_local_override_when_provided(self):
        local_path = r"C:\models\Qwen2.5-7B-Instruct"
        self.assertEqual(resolve_llm_model_path("qwen_7b", local_path), local_path)
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_model_path_resolution -v`
Expected: FAIL because `resolve_llm_model_path` does not exist yet.

**Step 3: Write minimal implementation**

Add a helper to `model/__init__.py`:

```python
def resolve_llm_model_path(llm_model_name, llm_model_path=""):
    if llm_model_path and llm_model_path.strip():
        return llm_model_path
    return model_path[llm_model_name]
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_model_path_resolution -v`
Expected: PASS.

### Task 2: Use shared resolver in entrypoints

**Files:**
- Modify: `train.py`
- Modify: `inference.py`
- Modify: `src/config.py`
- Test: `tests/test_model_path_resolution.py`

**Step 1: Write the failing test**

Add a whitespace case to the resolver test file:

```python
def test_ignores_blank_override_and_uses_default(self):
    self.assertEqual(
        resolve_llm_model_path("llama_8b", "   "),
        "meta-llama/Meta-Llama-3-8B-Instruct",
    )
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_model_path_resolution -v`
Expected: FAIL if blank-path handling is missing.

**Step 3: Write minimal implementation**

- Replace direct assignments in `train.py` and `inference.py` with `resolve_llm_model_path(...)`
- Update `src/config.py` help text for `--llm_model_path`

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_model_path_resolution -v`
Expected: PASS.

### Task 3: Smoke-verify the CLI imports

**Files:**
- Modify: none
- Test: `train.py`, `inference.py`

**Step 1: Run smoke checks**

Run: `python -c "from train import main; print('train smoke ok')"`

Run: `python -c "from inference import main; print('inference smoke ok')"`

Expected: both imports succeed.

**Step 2: Review changed files**

Run: `git diff -- tests/test_model_path_resolution.py model/__init__.py train.py inference.py src/config.py`

Expected: only path-resolution related changes.
