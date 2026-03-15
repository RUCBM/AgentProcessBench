# AgentProcessBench Annotation Platform

A deployable local/intranet data annotation platform (integrated frontend + backend) for step-level labeling of the 4 JSONL files under `annotation_platform/raw_trajectories/annotation_file_diverse_queries/` (`gaia_dev.jsonl`, `hotpotqa.jsonl`, `tau2.jsonl`, `bfcl.jsonl`):

- Label each "assistant step" (positive/negative/uncertain)
- Label the final result separately (positive/negative/uncertain)
- Persist annotations to SQLite + append-only JSONL exports for downstream training/statistics

For labeling guidelines, see: `annotation_platform/ANNOTATION_GUIDE_v1.md`

## 1) Start

Run from the repository root:

```bash
python annotation_platform/server.py --host 0.0.0.0 --port 8000
```

Open in your browser:

- `http://localhost:8000/`

## 1.1) Frontend Shortcuts

- Each assistant message has `+ / 0 / -` buttons on the right. Click to label that step (the selected button is highlighted).
- The top "Final Result Label" also uses buttons for `+1/0/-1` (highlighted when selected).
- On first open, a "Start Annotation" page prompts for username (you can also change it in the top-right Annotator field).
- Keyboard (for the currently focused assistant step):
  - `j/k` or `↑/↓`: switch between assistant steps
  - `1/0/-`: assign `+1/0/-1` to the current step

## 2) Data Paths

Default input:

- `annotation_platform/raw_trajectories/annotation_file_diverse_queries/*.jsonl`

Default storage:

- SQLite: `annotation_platform/annotation_results/annotations.sqlite3`
- Export JSONL: `annotation_platform/annotation_results/exports/<dataset>__<username>.jsonl` (each line includes a `username` field)

You can override paths with arguments:

```bash
python annotation_platform/server.py --annotation_dir annotation_platform/raw_trajectories/annotation_file_diverse_queries --data_dir annotation_platform/annotation_results
```

## 3) Labeling Guidelines (Recommended)

- `+1`: The assistant's judgment/action/output in this step is correct.
- `-1`: The assistant's judgment/action/output in this step is incorrect. (Use `-1` conservatively: only when this step leads to a wrong conclusion and significantly affects the subsequent process.)
- `0`: The step is exploratory, does not significantly affect the main task, or has no clear benefit/harm. (Since `-1` can strongly impact LLM Agent RL training, exploratory steps that gather supplemental but weakly related information should be labeled `0`.)

## 4) Export

Each time an annotation is saved, the platform appends one JSON line to the corresponding export file (append-only).

## 5) LLM Auto-Annotation (Offline Script)

Run from the repository root (reads `OPENAI_BASE_URL` / `OPENAI_API_KEY` from environment variables, or override via arguments):

- Without `ground_truth`/`reward_info` (recommended for formal process scoring):
  - `python -m annotation_platform.llm_annotate annotation_platform/raw_trajectories/annotation_file_diverse_queries/gaia_dev.jsonl --model <model> --mode blind`
- With `ground_truth`/`reward_info` (recommended only for reference annotation/data alignment):
  - `python -m annotation_platform.llm_annotate annotation_platform/raw_trajectories/annotation_file_diverse_queries/gaia_dev.jsonl --model <model> --mode reference`

Default output path: `annotation_platform/annotation_results/exports/<dataset>__<annotator>.jsonl`, with an extra `explanations` field added to each line for manual review.
