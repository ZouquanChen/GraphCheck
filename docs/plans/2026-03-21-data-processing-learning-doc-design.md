# GraphCheck Data Processing Learning Doc Design

**Goal:** Create a new Chinese learning document that starts with GraphCheck's data-processing pipeline, using a data-flow-first structure with code snippets and explanatory annotations.

**Audience:** A reader learning the project from scratch who needs to understand how raw dataset rows become trainable graph batches.

**Approach:**
- Create a standalone document at `learning/数据处理学习笔记.md` instead of expanding the broader roadmap.
- Organize the content by data flow: raw `.pkl` data -> graph preprocessing in `graph_build.py` -> split generation -> `KGDataset` loading -> `collate_fn` batching -> `train.py` usage.
- Use short, high-signal code excerpts rather than copying full files, and place plain Chinese explanations directly under each excerpt.

**Sections:**
1. Why the data-processing stage matters in GraphCheck
2. Raw dataset fields and what each column means
3. `graph_build.py`: `textualize_graph()`, `step_one()`, `step_two()`, `generate_split()`
4. How train/val/test splits are generated and reused
5. `dataset/utils/dataset.py`: how `KGDataset` reconstructs one sample
6. `dataset/utils/collate.py`: how PyG graphs become a batch
7. `train.py`: how the split and `DataLoader` connect to training
8. Minimal hands-on checks for the reader

**Non-goals:**
- Do not cover model internals (`model/graphcheck.py`, GNN, projector) in depth yet.
- Do not rewrite the existing `learning/学习路线.md`.
- Do not change project code.
