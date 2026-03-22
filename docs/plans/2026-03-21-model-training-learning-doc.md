# Model And Training Learning Doc Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write a new Chinese learning document that explains GraphCheck's model structure and training method from `GraphCheck` internals through the training loop.

**Architecture:** The note stays separate from the data-processing note and follows the runtime path inside the code. It starts from model composition, then follows graph encoding and graph injection into the frozen LLM, and finally explains optimization, checkpointing, and evaluation behavior.

**Tech Stack:** Markdown, Python source excerpts, PyTorch, HuggingFace Transformers, PyTorch Geometric

---

### Task 1: Create the document skeleton

**Files:**
- Create: `learning/模型结构与训练方法学习笔记.md`
- Reference: `learning/数据处理学习笔记.md`

**Step 1: Draft the heading structure**

Create sections for overall intuition, model composition, GNN encoding, graph-to-LLM injection, loss construction, training loop, caveats, and hands-on checks.

**Step 2: Verify the structure is model-first**

Check that the section order starts with `GraphCheck` internals before `train.py`, so the reader sees what is being trained before how it is trained.

**Step 3: Save the initial markdown file**

Write the headings and a short intro that explicitly links this note to the earlier data-processing note.

### Task 2: Add code-backed model explanations

**Files:**
- Modify: `learning/模型结构与训练方法学习笔记.md`
- Reference: `model/graphcheck.py`
- Reference: `model/gnn.py`

**Step 1: Extract minimal constructor excerpts**

Select short excerpts showing tokenizer/LLM loading, LLM freezing, GNN creation, and projector creation.

**Step 2: Explain graph encoding**

Add an excerpt for `encode_graphs()` and explain node-level encoding, graph-level pooling, and the two returned graph vectors.

**Step 3: Explain graph injection into the LLM**

Add an excerpt from `forward()` showing `claim_embedding`, `doc_embedding`, and `torch.cat([bos_embeds, claim_embedding, doc_embedding, inputs_embeds], dim=0)`.

### Task 3: Add training-method explanations

**Files:**
- Modify: `learning/模型结构与训练方法学习笔记.md`
- Reference: `train.py`
- Reference: `src/utils.py`
- Reference: `src/ckpt.py`

**Step 1: Explain the supervision target**

Show how the label text becomes `support` or `unsupport`, how label tokens are appended, and how `IGNORE_INDEX` masks non-label positions.

**Step 2: Explain the optimization scope**

Describe that the optimizer only receives parameters with `requires_grad=True`, so the frozen LLM is excluded.

**Step 3: Explain validation, early stopping, checkpointing, and final balanced accuracy evaluation**

Cover the epoch loop, best checkpoint saving, reload of the best model, and regex-based label extraction in `get_accuracy(...)`.

### Task 4: Document important current-code caveats

**Files:**
- Modify: `learning/模型结构与训练方法学习笔记.md`
- Reference: `src/config.py`
- Reference: `train.py`

**Step 1: Check whether configuration fields are used**

Confirm and note that `llm_num_virtual_tokens` exists in config but is not used in the current forward path.

**Step 2: Check how `grad_steps` behaves in practice**

Confirm from `train.py` that the current code still calls `optimizer.zero_grad()` and `optimizer.step()` every batch, so `grad_steps` does not implement standard gradient accumulation.

**Step 3: Phrase caveats as code-reading notes, not code changes**

Keep the tone educational and descriptive, and do not suggest that the document itself modifies behavior.

### Task 5: Verify accuracy against source files

**Files:**
- Verify: `learning/模型结构与训练方法学习笔记.md`
- Reference: `model/graphcheck.py`
- Reference: `model/gnn.py`
- Reference: `train.py`
- Reference: `src/utils.py`
- Reference: `src/ckpt.py`

**Step 1: Re-read the markdown**

Confirm every described behavior exists in the current code and no inferred mechanism is presented as fact.

**Step 2: Check terminology consistency**

Make sure the document consistently distinguishes node embeddings, graph embeddings, token embeddings, labels, and predictions.

**Step 3: Check caveat wording**

Make sure notes about unused config fields or non-standard training behavior are framed as observations from the current implementation.

**Step 4: Commit**

If the user later requests a commit, stage:

```bash
git add docs/plans/2026-03-21-model-training-learning-doc-design.md docs/plans/2026-03-21-model-training-learning-doc.md learning/模型结构与训练方法学习笔记.md
git commit -m "docs: add model and training learning note"
```
