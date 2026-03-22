# GraphCheck Model And Training Learning Doc Design

**Goal:** Create a new standalone Chinese learning document that explains GraphCheck's model structure and training method with a model-structure-first narrative.

**Audience:** A reader who has already understood the data pipeline and now wants to understand how graphs are encoded, injected into the LLM, and optimized during training.

**Approach:**
- Create a standalone document at `learning/模型结构与训练方法学习笔记.md` instead of extending `learning/数据处理学习笔记.md`.
- Organize the content around the actual model path in code: `GraphCheck.__init__()` -> `model/gnn.py` -> `encode_graphs()` -> `forward()` -> `train.py` -> checkpointing and evaluation.
- Use short source excerpts plus direct Chinese explanation, and explicitly connect each section back to the previous data-processing note.

**Sections:**
1. Why GraphCheck is neither a pure LLM nor a pure GNN
2. `GraphCheck` overall structure: frozen LLM, `graph_encoder`, and `projector`
3. `model/gnn.py` and `encode_graphs()`: how two graph batches become two graph vectors
4. How graph vectors are injected into the LLM input embedding sequence
5. How `forward()` turns generation into `support` / `unsupport` training
6. How `train.py` optimizes the trainable modules and validates the model
7. Important implementation details and caveats visible in the current code
8. Minimal hands-on checks for the reader

**Non-goals:**
- Do not re-explain the full preprocessing pipeline already covered in `learning/数据处理学习笔记.md`.
- Do not modify project code.
- Do not expand into a full inference-only deep dive beyond what is needed to explain training and evaluation.
