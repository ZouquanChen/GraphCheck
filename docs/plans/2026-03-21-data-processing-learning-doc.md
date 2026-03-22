# Data Processing Learning Doc Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write a new Chinese learning document that explains GraphCheck's data-processing pipeline with code snippets and annotations.

**Architecture:** The document stays separate from the existing roadmap and follows the actual runtime data flow. Each section anchors to one or two source files so the reader can bounce between the note and the codebase without losing context.

**Tech Stack:** Markdown, Python source excerpts, pandas, PyTorch Geometric

---

### Task 1: Create the document skeleton

**Files:**
- Create: `learning/数据处理学习笔记.md`
- Reference: `learning/学习路线.md`

**Step 1: Draft the heading structure**

Create sections for overview, raw data format, preprocessing pipeline, split generation, `KGDataset`, `collate_fn`, training entry, and hands-on checks.

**Step 2: Verify the structure is data-flow-first**

Check that the section order matches runtime order: raw data -> preprocessing -> split -> load -> batch -> train.

**Step 3: Save the initial markdown file**

Write the headings and short intro text so later edits only need to fill in content.

### Task 2: Add code-backed explanations

**Files:**
- Modify: `learning/数据处理学习笔记.md`
- Reference: `graph_build.py`
- Reference: `dataset/utils/dataset.py`
- Reference: `dataset/utils/collate.py`
- Reference: `train.py`

**Step 1: Extract minimal code snippets**

Select short excerpts for `textualize_graph()`, split creation, `KGDataset.__getitem__`, `get_idx_split()`, `collate_fn`, and `DataLoader` setup.

**Step 2: Add plain Chinese annotations under each snippet**

Explain what the inputs are, what files are produced, and why each step matters for later training.

**Step 3: Add one or two concrete file paths per stage**

Point the reader to directories such as `dataset/extracted_KG/<dataset>/graphs/claim/` and `dataset/extracted_KG/<dataset>/split/`.

### Task 3: Add practical learning checks

**Files:**
- Modify: `learning/数据处理学习笔记.md`

**Step 1: Add a minimal inspection example**

Include a short `pandas.read_pickle(...)` example for understanding raw dataset rows.

**Step 2: Add a graph object inspection example**

Include a short `torch.load(...)` example so the reader can inspect one generated graph.

**Step 3: Add a split inspection example**

Include a short example showing how to read `train_indices.txt` or call `KGDataset.get_idx_split()`.

### Task 4: Verify accuracy against source files

**Files:**
- Verify: `learning/数据处理学习笔记.md`
- Reference: `graph_build.py`
- Reference: `dataset/utils/dataset.py`
- Reference: `dataset/utils/collate.py`
- Reference: `train.py`

**Step 1: Re-read the markdown**

Confirm every described step exists in the current code and no future behavior is invented.

**Step 2: Check split description**

Confirm the document states the actual `60/20/20` random split and notes the fixed `random_state=42`.

**Step 3: Check terminology consistency**

Make sure the document consistently uses `doc`, `claim`, `KG`, `graph`, `split`, and `batch` in the same way the code does.

**Step 4: Commit**

If the user later requests a commit, stage:

```bash
git add docs/plans/2026-03-21-data-processing-learning-doc-design.md docs/plans/2026-03-21-data-processing-learning-doc.md learning/数据处理学习笔记.md
git commit -m "docs: add data processing learning note"
```
