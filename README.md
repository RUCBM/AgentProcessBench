# AgentProcessBench: Diagnosing Step-Level Process Quality in Tool-Using Agents

<p align="center">
  <a href="https://rucbm.github.io/AgentProcessBench-Homepage/"><img alt="Homepage" src="https://img.shields.io/badge/%F0%9F%8C%8D%20Homepage-2E8B57?style=for-the-badge" /></a>
  <a href="https://huggingface.co/datasets/LulaCola/AgentProcessBench"><img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-F59E0B?style=for-the-badge" /></a>
  <a href="https://arxiv.org/abs/2603.14465"><img alt="Paper" src="https://img.shields.io/badge/%F0%9F%93%91%20Paper-2563EB?style=for-the-badge" /></a>
  <a href="https://github.com/RUCBM/AgentProcessBench/blob/main/README.md"><img alt="DOCUMENT" src="https://img.shields.io/badge/%F0%9F%93%96%20DOCUMENT-8B5CF6?style=for-the-badge" /></a>
</p>

![Method](figs/Method.png)

AgentProcessBench is a benchmark for **process-level evaluation of agent trajectories**.
Each trajectory contains multi-turn messages and tool interactions, and the target is to predict step-wise process labels.

## 👀 Overview

AgentProcessBench contains `1000` trajectories (`4` datasets × `250` samples) from `hotpotqa`, `gaia_dev`, `bfcl`, and `tau2`.
It evaluates whether a model can make reliable step-level process judgments under a unified protocol.
To support this benchmark, we built a dedicated data annotation platform in `annotation_platform/`.

The figure below reports cross-setting comparisons, showing relative strengths and weaknesses across datasets.

![Comparison](figs/Comparison.png)

The next figure summarizes overall performance, giving a compact view of aggregate step-level effectiveness.

![OverallPerformance](figs/OverallPerformance.png)

## 📑 Quick Start

### Data Access

- Local benchmark data: `data/AgentProcessBench/`

### Run Evaluation

Full benchmark:

```bash
cd /path/to/AgentProcessBench
export OPENAI_BASE_URL="your_api_url"
export OPENAI_API_KEY="your_api_key"
bash eval/eval.sh --model deepseek-chat --concurrency 8
```

Subset example:

```bash
bash eval/eval.sh --model deepseek-chat --datasets hotpotqa --start 0 --end 50 --concurrency 8
```

### Evaluation Outputs

All outputs are written under `eval/yourresults/`:

- predictions: `eval/yourresults/<run_name>/*.jsonl`
- raw judge logs: `eval/yourresults/_raw/<run_name>/*.jsonl`
- score table: `eval/yourresults/<run_name>/score.txt`

Printed metrics include:

- per-dataset: `step_micro_acc`, `firsterroracc`
- overall (AVG): `step_micro_acc`, `firsterroracc`
