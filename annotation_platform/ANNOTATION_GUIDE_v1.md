# AgentProcessBench Annotation Guide (v1.1)

This guide is for manual annotation in **AgentProcessBench (Process Reward / step-level supervision for tool-use scenarios)**. This benchmark focuses on the **process quality of Tool-Use Agents in multi-step interactions**, with emphasis on:

- Whether tool selection and parameters are reasonable
- Whether tool outputs are correctly understood and used
- Whether system and task constraints are followed (policy / format / state)
- Whether realistic **"make mistakes -> correct them -> continue"** behavior is allowed

Core judgment criterion:

> Does this step make the task "closer to being correctly completed"? (Not just "looks like effort".)

---

## 0. "Step" Definition Consistent with the Annotation Platform (Important)

To avoid ambiguity, the step definition in this guide is **strictly aligned with the current `annotation_platform` implementation** (each scorable unit you see on the web page):

- **One Step = one `role="assistant"` message (assistant message)**
- The platform only allows you to score those assistant messages with `+ / 0 / -`
- `role="user"` and `role="tool"` are **not scored**, but they are key evidence for correctness

> Note: even if one assistant message contains multiple `tool_calls`, in the current platform you still assign **only one label** (the "net contribution" of the whole message).

---

## 1. Annotation Goals and Outputs

Given a full trajectory (user task, tool definitions, model outputs, tool returns, etc.), your tasks are:

1) **Label each assistant message (+1 / 0 / -1)**  
2) **Assign one separate label to the final result (+1 / 0 / -1)**

Hard platform constraints (otherwise it cannot be saved):

- **Every assistant message must have a label**
- **The final result label must be clicked/touched**

Outputs are written to SQLite and exported to JSONL (append-only) for downstream training/statistics.

---

## 2. Label System (Three-way, Goal-oriented)

### 2.1 Positive (+1)

This step is **correct, reasonable, and materially moves the task forward**.

Common cases:
- Chose the **right tool** with appropriate timing
- Tool parameters satisfy schema and task constraints (fields/types/formats/ranges/prerequisites)
- Correctly understood and used tool returns (citing evidence, updating state, revising plan)
- Necessary clarification (high risk of error or inability to proceed without it)
- **Effective correction**: identifies and fixes earlier errors, bringing the trajectory back on track
- **Cross-validation**: if "repetition" is purposeful verification/cross-checking and it truly adds key evidence or fixes earlier mistakes (e.g., verifying the same fact with another source, reproducing an error to locate the cause), mark it as Positive.

Judgment rule:
> Keeping this step makes correct task completion easier.

---

### 2.2 Negative (-1)

This step **introduces errors, deviates from the goal, or harms the chance of task completion**.

Common cases:
- Clearly wrong tool choice, or parameter errors causing invalid calls/overreach/schema violations
- **Misreading tool outputs**, or "imagining" they say what is needed
- **Fabrication/unsupported assumptions**: certainty claims without evidence, fake citations, fake tool results
- **Redundancy**: obvious repetition with no new information (same tool + same params/same query, no strategy change), or repeated actions despite sufficient evidence; wasting steps, missing key actions, or repeatedly advancing on wrong premises
- Continuing on a wrong premise ("compounding errors") with no verification/rollback/correction
- Violating system/task constraints (policy/format/state machine; executing directly when confirmation is required)
- Obvious off-topic drift that loses the main task

Judgment rule:
> This step makes the task harder, and may even drive the trajectory toward failure.

---

### 2.3 Neutral (0)

This step has **unclear impact, minimal contribution, or insufficient evidence to judge positive vs. negative**.

Common cases:
- Reasonable exploration/attempts but no useful output yet (e.g., one search miss)
- Tool failure due to environment/external causes (404, permission, network, missing file, etc.) and not obviously caused by bad parameters
- Content is mostly correct but low-information or slightly redundant, without clear harm

Decision principles:
- Prefer **Neutral** when uncertain  
- But "making things up / hard guessing / misreading returns without evidence" is usually **Negative**

---

## 3. Definition of "Final Result Label" (Relation to Step Labels)

The "Final Result Label" at the top of the platform evaluates the quality of the **final deliverable/final answer**. It is not the same as the label of the last assistant message.

Recommended criteria:

- **Final result +1**: final answer matches ground truth (or can be judged correct from trajectory evidence), and meets required format/constraints
- **Final result -1**: final answer is wrong, format is non-compliant, or conclusion conflicts with evidence
- **Final result 0**: insufficient evidence to determine correctness, or the task is open-ended and trajectory evidence is insufficient

How to handle common conflicts:
- Most process steps are reasonable but final answer is wrong: many steps can be +1 / 0, but **final result should be -1**
- Process includes trial and error, but errors are corrected and final answer is correct: intermediate steps can include -1, then return to +1 after correction, and **final result is +1**

---

## 4. Error Propagation and the "Correctability" Principle (Very Important)

### 4.1 Error Propagation Rule

If one step introduces incorrect facts/state, and later steps **continue entirely based on that wrong premise** with **no** verification/rollback/correction:

- Later steps should be labeled **Negative (-1)**

### 4.2 Correction Exceptions (Allowed to "Return to the Right Track")

If later steps clearly do any of the following, you do not need to directly apply error propagation and mark Negative:

- Point out previous errors (facts/parameters/tool choice/constraint understanding)
- Re-search, realign evidence, roll back state, or switch tools/parameters for verification
- Fix based on tool error messages

Annotation suggestions:
- **Effective correction (successfully corrected)** -> Positive
- **Correction attempted but information still insufficient** -> Neutral
- **Superficial correction, or still actually wrong** -> Negative

---

## 5. Tool Failures / Exceptions (Unified Criteria)

When a tool call fails (404/permission/no result/timeout/parse failure, etc.):

- **First reasonable failed attempt**: usually Neutral
- **Adjusting strategy based on failure information (change query/tool/evidence source/fallback plan)**: Positive
- **Repeating the same error (same params/same query/no change)**: shifts from Neutral toward Negative (depending on repetition and damage)
- **Making up answers or faking evidence after failure**: Negative

---

## 6. Annotation Focus by Data Source

### 6.1 HotpotQA (Text QA + Retrieval)

Focus:
- Whether retrieval queries are on-topic and target the correct entities
- Whether conclusions are based on retrieved evidence (closed evidence chain)
- Whether the final answer matches ground truth and required format (e.g., `<answer>...</answer>`)

---

### 6.2 GAIA-Text Only (Multi-step Retrieval / Reading)

Common tools: `search`, `fetch_url`, `read_file` (follow each sample's tool documentation)

Focus:
- Whether the problem space is narrowed step by step and key evidence is captured
- Whether page structures/long documents are handled correctly (locating question-relevant passages)
- Whether the final answer satisfies format/constraints (number/unit/case-sensitivity/specific wrapper tags, etc.)

---

### 6.3 tau^2-bench (Real Business Process Simulation)

Primary focus: **compliance and process** (usually reflected in system messages or tool documentation)

Key checks:
- Whether actions follow policy (what can/cannot be done)
- Whether user intent is confirmed when needed (especially irreversible actions, refunds/compensation, cancellation/rebooking, etc.)
- Whether escalation/hand-off is reasonable (when it is truly required by the task -> Positive; when it is used to avoid solving -> Negative)

---

### 6.4 BFCL multi-turn (Function Composition / Stateful Environment)

Focus:
- Whether tool prerequisites are satisfied (ordering, state machine)
- Whether parameter names/types/ranges are correct
- Whether corrections are made based on error messages

---
## 7. Interpreting Ground Truth / Reward Info Fields (By Dataset)

The **Ground Truth / Reward Info** panel on the left provides "verifiable reference information" for annotators.  
These fields are not meant to replace human judgment; they help you **quickly understand task expectations, check final outcomes, and locate possible violations/failure causes**.

---

### General Usage Principles (Read First)

- **Human annotation first**: your judgment always overrides automatic reward.
- **Final result judgment**:
  - Prioritize `ground_truth`
  - `reward_info.reward` is only an automatic evaluation hint; it may be unstable, may be 0, and may miss errors.
- **Step-level annotation**:
  - Always center on: "Does this step advance correct task completion?"
  - `ground_truth / reward_info` is mainly used to judge:
    - Whether the final result is correct
    - Whether explicit constraints/process/policy are violated

⚠️ **Note**:  
- `reward = 0` != "definitely wrong"  
- `reward = 1` != "definitely correct"  

---

## 7. Interpreting Ground Truth / Reward Info Fields (By Dataset)

### 7.1 HotpotQA

#### Field Description

- **`ground_truth.target`**
  - Type: `List[str]`
  - Meaning: acceptable final answer set (usually one)
  - Use: determine whether the final answer is correct

- **`reward_info.reward`**
  - Hint result from the automatic judge
  - `reward = 0` often means "uncertain / cannot determine"

#### Annotation Suggestions

- **Final result**
  - Final answer not in `target` -> final result is usually `-1`
  - In `target` -> final result can be `+1`
- **Step-level**
  - Whether retrieval is on-topic and key evidence is found
  - Whether the model "makes things up without evidence"
- Even if the final answer is wrong, reasonable retrieval steps can still be `+1`

---

### 7.2 GAIA-Text Only

#### Field Description

- **`ground_truth.answer`**
  - The unique correct final answer (string or numeric string)

- **`ground_truth.Steps / Tools / Number of steps`**
  - Reference solution path description
  - ⚠️ Not a hard constraint; the model does not need to replicate it

- **`reward_info.reward`**
  - Automatic evaluation hint, for reference only

#### Annotation Suggestions

- **Final result**
  - Matches `ground_truth.answer` -> usually `+1`
  - Does not match -> usually `-1`
- **Process**
  - Focus on whether the evidence chain is reasonable
  - Tool outputs may be imperfect; the model should iterate across rounds based on returns. Responding directly based on a faulty tool result should be labeled `-1`
- Do not mark as wrong solely because the model's path differs from the reference path

---

### 7.3 tau^2-bench (airline / retail / telecom)

#### Field Description

- **`ground_truth.nl_assertions`**
  - Key business constraints expressed in natural language
  - For example:
    - "Should refuse to cancel the flight"
    - "Must not promise a refund"
    - "Must explain the reason to the user"

- **`ground_truth.actions`**
  - Execution steps in the reference answer

- **`communicate_info`**
  - At certain key points, what information/position the assistant should explicitly communicate to the user.


- **`reward_info.reward_breakdown`**
  - Itemized results from automatic checks (for quickly locating issues)

#### Annotation Suggestions

- If assistant behavior clearly violates `nl_assertions`
  -> related steps and final result tend toward `-1`
- If assertions are followed and the process advances reasonably
  -> steps and final result can be `+1`
- tau^2 tasks do not always have a single "answer sentence"; compliant process itself is success

---

### 7.4 BFCL multi-turn

#### Field Description

- **`ground_truth`**
  - Expected multi-turn target sequence
  - Each turn includes several key tool-call strings
  - Order and key actions matter, but exact wording is not required

- **`reward_info.info`**
  - Verifier output (e.g., `valid`, `error_type`, `expected_tool_names`, etc.)
  - Used to locate failure causes

#### Annotation Suggestions

- **Final result**
  - Insufficient turns, missing key actions, early termination -> usually `-1`
- **Step-level**
  - Obvious tool-name/parameter errors -> `-1`
  - Correcting based on errors and returning to the right path -> correction step can be `+1`
- Reasonable exploratory errors can be `0`

---

## 8. Labeled Examples (Assistant-message Granularity)

Notes:
- `S1/S2/...` in the tables each correspond to **one `role="assistant"` message**
- `role="tool"` returns are not scored, but you must use them to judge neighboring steps

### A. HotpotQA Example 1 (Correct Chain)

**Task**: Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?  
**Target answer**: No

| Step | Behavior Summary | Label | Reason |
|---|---|---|---|
| S1 | `search` for Laleli Mosque location | Positive | Obtains key location info |
| S2 | `search` for Esma Sultan Mansion location | Positive | Gets evidence for the second entity |
| S3 | Output `<answer>No</answer>` | Positive | Conclusion is correct and evidence-based |

### B. HotpotQA Example 2 (Wrong Due to Evidence Mismatch)

| Step | Behavior Summary | Label | Reason |
|---|---|---|---|
| S1 | `search`, result points to wrong university | Neutral | Exploratory but off-target |
| S2 | `search` again, still no entity correction | Neutral | Repetitive and no correction |
| S3 | Output wrong answer (inconsistent with evidence) | Negative | Conclusion conflicts with evidence |

### C. GAIA Example (Corrected After Failure, Final Correct)

| Step | Behavior Summary | Label | Reason |
|---|---|---|---|
| S1 | `search` key entity/clue | Positive | Correct retrieval direction |
| S2 | `fetch_url` returns 404 | Neutral | Reasonable attempt but failed (not obviously random input) |
| S3 | Switch to another retrieval/evidence source | Positive | Effective correction |
| S4 | `fetch_url`/`read_file` gets key evidence | Positive | Critical information obtained |
| S5 | Output final answer (format-compliant) | Positive | Consistent with ground truth |

### D. GAIA Example (Reasonable Process, Wrong Conclusion)

| Step | Behavior Summary | Label | Reason |
|---|---|---|---|
| S1-S4 | Multiple retrievals and document evidence localization | Positive | Process advances reasonably |
| S5 | Output wrong final answer | Negative | Inconsistent with ground truth |

### E. tau^2-bench Example (Compliant Refusal + Reasonable Escalation)

| Step | Behavior Summary | Label | Reason |
|---|---|---|---|
| S1 | Query order/status | Positive | Gets key facts |
| S2 | Explain policy constraints/compliance boundaries | Positive | Compliance explanation |
| S3 | Escalate/hand off according to process | Positive | Reasonable escalation |

### F. tau^2-bench Example (Off-topic + Fabrication)

| Step | Behavior Summary | Label | Reason |
|---|---|---|---|
| S1-S3 | Initially advances around the main task | Positive | Advances core task |
| S4-S6 | Deviates from main task and does not return to process | Negative | Main task lost |
| S7 | Fabricated/confused compensation or policy | Negative | Inconsistent with constraints |
| S8 | Directly escalates to avoid problem-solving | Negative | Non-compliant handling/no correction |

### G. BFCL Example (Parameter Error Then Correction)

| Step | Behavior Summary | Label | Reason |
|---|---|---|---|
| S1 | Locate resource/read document | Positive | Correct targeting |
| S2 | Tool call parameter error -> error returned | Negative | Violates schema/prerequisites |
| S3 | Fix parameters and retry based on error | Positive | Effective correction |
| S4 | Subsequent calls complete the task | Positive | Advances and completes |

### H. BFCL Example (Ignoring Prerequisites + Extra Action)

| Step | Behavior Summary | Label | Reason |
|---|---|---|---|
| S1 | Correct required tool call | Positive | Advances main task |
| S2 | Ignores prerequisites and calls directly -> error/invalid | Negative | Prerequisite error |
| S3 | Satisfies prerequisites and retries | Positive | Correction |
| S4 | Extra action unrelated to task | Negative | Damages state/off-topic |



## 9. Annotation Workflow Suggestions

1. First read the user task end-to-end and clarify the "final deliverable"
2. Judge each step against corresponding tool returns
3. Use "whether it advances task completion" as the only core criterion
4. When uncertain, prefer **Neutral**
5. Avoid introducing external knowledge; judge only from the trajectory itself

---

**This guide is the official AgentProcessBench annotation specification v1.1**  
