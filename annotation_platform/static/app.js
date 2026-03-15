function $(id) {
  return document.getElementById(id);
}

const state = {
  datasets: [],
  dataset: null,
  annotator: "",
  rememberAnnotator: false,
  item: null,
  llmReferences: null,
  stepLabels: {},
  finalLabel: null,
  finalLabelTouched: false,
  comment: "",
  focusedAssistantIdx: null,
  toolsFilter: "",
};

function _readRememberAnnotator() {
  return localStorage.getItem("apb_remember_annotator") === "true";
}

function _setRememberAnnotator(v) {
  state.rememberAnnotator = Boolean(v);
  localStorage.setItem("apb_remember_annotator", state.rememberAnnotator ? "true" : "false");
  const cb = $("rememberAnnotator");
  if (cb) cb.checked = state.rememberAnnotator;
}

function isValidLabel(v) {
  return v === 1 || v === 0 || v === -1;
}

function normalizeLabel(v) {
  if (v === null || v === undefined) return null;
  if (v === 1 || v === 0 || v === -1) return v;
  const n = parseInt(String(v).trim(), 10);
  return n === 1 || n === 0 || n === -1 ? n : null;
}

function labelBadgeClass(v) {
  const n = normalizeLabel(v);
  if (n === 1) return "hlGreen";
  if (n === 0) return "hlAmber";
  if (n === -1) return "hlRed";
  return "muted";
}

function labelText(v) {
  const n = normalizeLabel(v);
  if (n === 1) return "+1";
  if (n === 0) return "0";
  if (n === -1) return "-1";
  return "—";
}

function showLogin() {
  $("loginOverlay").dataset.open = "true";
  const saved = localStorage.getItem("apb_annotator") || "";
  _setRememberAnnotator(_readRememberAnnotator());
  $("loginName").value = (state.rememberAnnotator ? saved : "") || "";
  setTimeout(() => $("loginName").focus(), 0);
}

function hideLogin() {
  $("loginOverlay").dataset.open = "false";
}

function setStatus(text) {
  $("progressText").textContent = text;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function stringify(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function _maybePrettyJsonString(s) {
  const t = String(s).trim();
  if (t.length < 2) return null;
  const looksJson =
    (t.startsWith("{") && t.endsWith("}")) ||
    (t.startsWith("[") && t.endsWith("]"));
  if (!looksJson) return null;
  if (t.length > 200_000) return null;
  try {
    return JSON.stringify(JSON.parse(t), null, 2);
  } catch {
    return null;
  }
}

function _decodeEscapesOnce(s) {
  const raw = String(s);
  let out = "";
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i];
    if (ch !== "\\") {
      out += ch;
      continue;
    }
    const next = raw[i + 1];
    if (next === undefined) {
      out += "\\";
      continue;
    }
    if (next === "n") {
      out += "\n";
      i++;
      continue;
    }
    if (next === "r") {
      out += "\n";
      i++;
      continue;
    }
    if (next === "t") {
      out += "\t";
      i++;
      continue;
    }
    if (next === "\\") {
      out += "\\";
      i++;
      continue;
    }
    if (next === '"') {
      out += '"';
      i++;
      continue;
    }
    if (next === "'") {
      out += "'";
      i++;
      continue;
    }
    if (next === "u") {
      const hex = raw.slice(i + 2, i + 6);
      if (/^[0-9a-fA-F]{4}$/.test(hex)) {
        out += String.fromCharCode(parseInt(hex, 16));
        i += 5;
        continue;
      }
    }
    if (next === "x") {
      const hex = raw.slice(i + 2, i + 4);
      if (/^[0-9a-fA-F]{2}$/.test(hex)) {
        out += String.fromCharCode(parseInt(hex, 16));
        i += 3;
        continue;
      }
    }
    out += "\\";
  }
  return out;
}

function polishEscapedText(s) {
  const raw = String(s);
  if (raw.length > 200_000) return raw;
  if (!raw.includes("\\")) return raw;

  let cur = raw;
  for (let pass = 0; pass < 2; pass++) {
    const next = _decodeEscapesOnce(cur);
    if (next === cur) break;
    cur = next;
  }
  return cur.replaceAll("\r\n", "\n").replaceAll("\r", "\n");
}

function unescapeJsonStringLiterals(jsonText) {
  const raw = String(jsonText);
  if (raw.length > 200_000) return raw;
  let out = "";
  let inString = false;
  let escaping = false;
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i];
    if (!inString) {
      out += ch;
      if (ch === '"') inString = true;
      continue;
    }
    if (escaping) {
      escaping = false;
      if (ch === "n") out += "\n";
      else if (ch === "r") out += "\n";
      else if (ch === "t") out += "\t";
      else if (ch === '"') out += '"';
      else if (ch === "\\") out += "\\";
      else if (ch === "/") out += "/";
      else if (ch === "u") {
        const hex = raw.slice(i + 1, i + 5);
        if (/^[0-9a-fA-F]{4}$/.test(hex)) {
          out += String.fromCharCode(parseInt(hex, 16));
          i += 4;
        } else {
          out += "u";
        }
      } else {
        out += ch;
      }
      continue;
    }
    if (ch === "\\") {
      escaping = true;
      continue;
    }
    out += ch;
    if (ch === '"') inString = false;
  }
  return out.replaceAll("\r\n", "\n").replaceAll("\r", "\n");
}

function stringifyToolContent(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") {
    const pretty = _maybePrettyJsonString(value);
    if (pretty !== null) return unescapeJsonStringLiterals(pretty);
    return polishEscapedText(value);
  }
  try {
    return unescapeJsonStringLiterals(JSON.stringify(value, null, 2));
  } catch {
    return String(value);
  }
}

async function copyToClipboard(text) {
  const payload = String(text ?? "");
  if (!payload) return false;

  if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
    try {
      await navigator.clipboard.writeText(payload);
      return true;
    } catch {
      // fallthrough
    }
  }

  const ta = document.createElement("textarea");
  ta.value = payload;
  ta.setAttribute("readonly", "true");
  ta.style.position = "fixed";
  ta.style.top = "-1000px";
  ta.style.left = "-1000px";
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  try {
    return document.execCommand("copy");
  } catch {
    return false;
  } finally {
    document.body.removeChild(ta);
  }
}

function makeCopyButton(getText) {
  const btn = document.createElement("button");
  btn.className = "copyBtn";
  btn.textContent = "Copy";
  btn.type = "button";
  btn.addEventListener("click", async (e) => {
    e.preventDefault();
    e.stopPropagation();
    const text = typeof getText === "function" ? getText() : "";
    const ok = await copyToClipboard(text);
    if (!ok) return;
    btn.dataset.copied = "true";
    const prev = btn.textContent;
    btn.textContent = "Copied";
    setTimeout(() => {
      btn.dataset.copied = "false";
      btn.textContent = prev || "Copy";
    }, 700);
  });
  return btn;
}

function attachCornerCopyButton(host, getText) {
  if (!host) return;
  if (host.querySelector(":scope > .copyBtn.copyCorner")) return;
  const btn = makeCopyButton(getText);
  btn.classList.add("copyCorner");
  host.appendChild(btn);
}

function renderJsonWithHighlights(value, highlightClassForKey) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return escapeHtml(value);

  const renderValue = (v, indent) => {
    if (v === null) return "null";
    if (v === undefined) return "null";
    if (typeof v === "string") return escapeHtml(JSON.stringify(v));
    if (typeof v === "number" || typeof v === "boolean") return escapeHtml(String(v));
    if (Array.isArray(v)) {
      if (v.length === 0) return "[]";
      const nextIndent = indent + 2;
      const parts = ["["];
      v.forEach((item, idx) => {
        const comma = idx === v.length - 1 ? "" : ",";
        parts.push(`\n${" ".repeat(nextIndent)}${renderValue(item, nextIndent)}${comma}`);
      });
      parts.push(`\n${" ".repeat(indent)}]`);
      return parts.join("");
    }
    if (typeof v === "object") {
      const obj = v;
      const keys = Object.keys(obj);
      if (keys.length === 0) return "{}";
      const nextIndent = indent + 2;
      const parts = ["{"];
      keys.forEach((key, idx) => {
        const comma = idx === keys.length - 1 ? "" : ",";
        const keyHtml = escapeHtml(JSON.stringify(key));
        const valueHtml = renderValue(obj[key], nextIndent);
        const cls = highlightClassForKey ? highlightClassForKey(key, obj[key]) : null;
        const wrapped = cls ? `<span class="${cls}">${valueHtml}</span>` : valueHtml;
        parts.push(`\n${" ".repeat(nextIndent)}${keyHtml}: ${wrapped}${comma}`);
      });
      parts.push(`\n${" ".repeat(indent)}}`);
      return parts.join("");
    }
    try {
      return escapeHtml(JSON.stringify(v));
    } catch {
      return escapeHtml(String(v));
    }
  };

  return renderValue(value, 0);
}

function renderGroundTruthHtml(groundTruth) {
  const hlKeys = new Set(["answer", "answer_text", "possible_answer", "target"]);
  return renderJsonWithHighlights(groundTruth, (key) => (hlKeys.has(key) ? "hlGreen" : null));
}

function renderRewardInfoHtml(rewardInfo) {
  const hlKeys = new Set(["answer", "answer_text", "possible_answer", "target"]);
  return renderJsonWithHighlights(rewardInfo, (key, v) => {
    if (key === "reward" && typeof v === "number") return v > 0 ? "hlGreen" : v < 0 ? "hlRed" : "hlAmber";
    return hlKeys.has(key) ? "hlGreen" : null;
  });
}

async function apiGet(path) {
  const res = await fetch(path, { method: "GET" });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function apiPost(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function renderFinalPills() {
  document.querySelectorAll("[data-final]").forEach((btn) => {
    const v = parseInt(btn.getAttribute("data-final"), 10);
    btn.dataset.active = v === state.finalLabel ? "true" : "false";
  });
}

function previewText(value, limit = 160) {
  const t = stringify(value).replaceAll(/\s+/g, " ").trim();
  if (t.length <= limit) return t;
  return t.slice(0, limit - 1) + "…";
}

function normalizeToolDef(tool) {
  if (!tool || typeof tool !== "object") return null;

  if (tool.type === "function" && tool.function && typeof tool.function === "object") {
    return {
      type: "function",
      name: String(tool.function.name || ""),
      description: tool.function.description || "",
      parameters: tool.function.parameters,
      raw: tool,
    };
  }

  if (tool.name || tool.description || tool.parameters) {
    return {
      type: tool.type || "unknown",
      name: String(tool.name || ""),
      description: tool.description || "",
      parameters: tool.parameters,
      raw: tool,
    };
  }

  return {
    type: tool.type || "unknown",
    name: String(tool.name || ""),
    description: "",
    parameters: null,
    raw: tool,
  };
}

function toolMatchesFilter(toolDef, filterText) {
  const q = String(filterText || "").trim().toLowerCase();
  if (!q) return true;
  const hay = `${toolDef.name}\n${stringify(toolDef.description)}\n${stringify(toolDef.parameters)}\n${stringify(
    toolDef.raw
  )}`.toLowerCase();
  return hay.includes(q);
}

function renderToolsDocs() {
  const root = $("toolsDocs");
  if (!root) return;
  root.innerHTML = "";

  if (!state.item) {
    root.textContent = "No sample loaded.";
    return;
  }

  const tools = state.item.tools;
  if (!Array.isArray(tools) || tools.length === 0) {
    root.textContent = "This sample does not include a `tools` field.";
    return;
  }

  const normalized = tools.map(normalizeToolDef).filter(Boolean);
  const filtered = normalized.filter((t) => toolMatchesFilter(t, state.toolsFilter));

  if (filtered.length === 0) {
    root.textContent = "No matching tools (try clearing the filter).";
    return;
  }

  filtered.forEach((toolDef) => {
    const details = document.createElement("details");
    details.className = "toolDoc";

    const summary = document.createElement("summary");
    const sumWrap = document.createElement("div");
    sumWrap.className = "toolSummary";

    const name = document.createElement("span");
    name.className = "toolName";
    name.textContent = toolDef.name || "(unnamed)";
    sumWrap.appendChild(name);

    const desc = document.createElement("span");
    desc.className = "toolDesc";
    desc.textContent = previewText(toolDef.description || stringify(toolDef.raw), 220);
    sumWrap.appendChild(desc);

    const copyBtn = makeCopyButton(() => stringifyToolContent(toolDef.raw));
    sumWrap.appendChild(copyBtn);

    summary.appendChild(sumWrap);
    details.appendChild(summary);

    const body = document.createElement("div");
    body.className = "toolDocBody";

    const descPre = document.createElement("pre");
    descPre.className = "pre";
    descPre.textContent = String(toolDef.description || "").trim() || "(no description)";
    body.appendChild(descPre);

    const schemaPre = document.createElement("pre");
    schemaPre.className = "pre";
    schemaPre.textContent = stringifyToolContent(toolDef.parameters || toolDef.raw);
    body.appendChild(schemaPre);

    details.appendChild(body);
    root.appendChild(details);
  });
}

function renderMessages() {
  const root = $("messages");
  root.innerHTML = "";
  if (!state.item) return;

  const assistantSet = new Set(state.item.assistant_message_indices || []);
  const messages = state.item.messages || [];
  const finalAssistantIdx = state.item.final_assistant_message_index;
  const focused = state.focusedAssistantIdx;

  messages.forEach((msg, idx) => {
    const role = msg.role || "unknown";
    const wrap = document.createElement("div");
    wrap.className = "msg";
    wrap.id = `msg-${idx}`;
    if (focused === idx) wrap.dataset.focused = "true";

    let label = assistantSet.has(idx) ? (state.stepLabels[String(idx)] ?? null) : null;
    if (
      state.finalLabelTouched &&
      isValidLabel(state.finalLabel) &&
      (label === null || label === undefined) &&
      role === "assistant" &&
      idx === finalAssistantIdx
    ) {
      label = state.finalLabel;
    }
    if (label !== null && label !== undefined) wrap.dataset.label = String(label);

    const header = document.createElement("div");
    header.className = "msgHeader";

    const badge = document.createElement("div");
    badge.className = `badge role-${role}`;
    badge.textContent = `${idx} · ${role}`;
    header.appendChild(badge);

    if (role === "assistant" && idx === finalAssistantIdx) {
      const fin = document.createElement("div");
      fin.className = "badge";
      fin.textContent = "final";
      header.appendChild(fin);
    }

    if (role === "tool") {
      const name = document.createElement("div");
      name.className = "badge";
      name.textContent = msg.name ? `tool: ${msg.name}` : "tool";
      header.appendChild(name);
    }

    const right = document.createElement("div");
    right.className = "msgRightControls";
    const copyBtn = makeCopyButton(() =>
      stringifyToolContent({
        index: idx,
        role,
        name: msg.name,
        content: msg.content,
        tool_calls: msg.tool_calls,
      })
    );
    right.appendChild(copyBtn);

    if (assistantSet.has(idx)) {
      const controls = document.createElement("div");
      controls.className = "stepControls";
      controls.appendChild(makeStepBtn(idx, 1, "+", state.stepLabels[String(idx)] === 1));
      controls.appendChild(makeStepBtn(idx, 0, "0", state.stepLabels[String(idx)] === 0));
      controls.appendChild(makeStepBtn(idx, -1, "-", state.stepLabels[String(idx)] === -1));
      right.appendChild(controls);
    }

    header.appendChild(right);

    const body = document.createElement("div");
    body.className = "msgBody";

    if (role === "tool" || role === "system") {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.className = "mono small muted";
      summary.textContent = previewText(msg.content);
      const pre = document.createElement("pre");
      pre.className = "pre";
      pre.textContent = stringifyToolContent(msg.content);
      details.appendChild(summary);
      details.appendChild(pre);
      body.appendChild(details);
    } else {
      const pre = document.createElement("pre");
      pre.className = "pre";
      pre.textContent = stringify(msg.content);
      body.appendChild(pre);
    }

    if (msg.tool_calls) {
      const toolBox = document.createElement("div");
      toolBox.className = "toolsBox";
      const title = document.createElement("div");
      title.innerHTML = `<span class="k">tool_calls</span>`;
      toolBox.appendChild(title);
      attachCornerCopyButton(toolBox, () => stringifyToolContent(msg.tool_calls));
      const tpre = document.createElement("pre");
      tpre.className = "pre";
      tpre.textContent = stringify(msg.tool_calls);
      toolBox.appendChild(tpre);
      body.appendChild(toolBox);
    }

    wrap.appendChild(header);
    wrap.appendChild(body);

    if (assistantSet.has(idx) && state.llmReferences && state.llmReferences.models) {
      const models = state.llmReferences.models || {};
      const order = state.llmReferences.order || Object.keys(models).sort();
      const details = document.createElement("details");
      details.className = "llmRefsDetails";
      details.open = false;

      const summary = document.createElement("summary");
      summary.className = "llmRefsSummary mono small muted";
      summary.textContent = `LLM Reference Annotations (${order.length} models)`;
      summary.addEventListener("click", (e) => {
        e.stopPropagation();
      });
      details.addEventListener("click", (e) => {
        e.stopPropagation();
      });
      details.appendChild(summary);

      const refWrap = document.createElement("div");
      refWrap.className = "llmRefs";

      order.forEach((modelKey) => {
        const rec = models[modelKey];
        if (!rec) return;
        const stepLabels = rec.step_labels || {};
        const explanations = rec.explanations || {};
        const stepsExpl = explanations.steps || {};
        const v = stepLabels[String(idx)];
        const expl = stepsExpl[String(idx)];

        const row = document.createElement("div");
        row.className = "llmRefRow";

        const modelEl = document.createElement("span");
        modelEl.className = "llmRefModel mono";
        modelEl.textContent = String(modelKey);
        row.appendChild(modelEl);

        const score = document.createElement("span");
        score.className = `llmRefScore ${labelBadgeClass(v)}`;
        score.textContent = labelText(v);
        row.appendChild(score);

        const explEl = document.createElement("span");
        explEl.className = "llmRefExpl";
        explEl.textContent = expl === null || expl === undefined || String(expl).trim() === "" ? "" : String(expl);
        row.appendChild(explEl);

        refWrap.appendChild(row);
      });

      if (idx === finalAssistantIdx) {
        const modelsFinal = state.llmReferences.models || {};
        const orderFinal = state.llmReferences.order || Object.keys(modelsFinal).sort();
        const fin = document.createElement("div");
        fin.className = "llmFinalRefs";
        const title = document.createElement("div");
        title.className = "llmFinalTitle mono muted small";
        title.textContent = "LLM final_label (reference):";
        fin.appendChild(title);

        orderFinal.forEach((modelKey) => {
          const rec = modelsFinal[modelKey];
          if (!rec) return;
          const v = rec.final_label;
          const expl = rec.explanations && rec.explanations.final;
          const row = document.createElement("div");
          row.className = "llmRefRow";

          const modelEl = document.createElement("span");
          modelEl.className = "llmRefModel mono";
          modelEl.textContent = String(modelKey);
          row.appendChild(modelEl);

          const score = document.createElement("span");
          score.className = `llmRefScore ${labelBadgeClass(v)}`;
          score.textContent = labelText(v);
          row.appendChild(score);

          const explEl = document.createElement("span");
          explEl.className = "llmRefExpl";
          explEl.textContent = expl === null || expl === undefined || String(expl).trim() === "" ? "" : String(expl);
          row.appendChild(explEl);

          fin.appendChild(row);
        });

        refWrap.appendChild(fin);
      }

      if (refWrap.childNodes.length > 0) {
        details.appendChild(refWrap);
        body.appendChild(details);
      }
    }

    if (assistantSet.has(idx)) {
      wrap.addEventListener("click", () => {
        state.focusedAssistantIdx = idx;
        renderStepsNav();
        renderMessages();
      });
    }
    root.appendChild(wrap);
  });
}

function makeStepBtn(idx, v, text, active) {
  const b = document.createElement("button");
  b.textContent = text;
  b.dataset.v = String(v);
  b.dataset.active = active ? "true" : "false";
  b.onclick = () => {
    const k = String(idx);
    if (state.stepLabels[k] === v) {
      delete state.stepLabels[k];
    } else {
      state.stepLabels[k] = v;
    }
    state.focusedAssistantIdx = idx;
    renderStepsNav();
    renderMessages();
  };
  return b;
}

function renderStepsNav() {
  const root = $("stepsNav");
  root.innerHTML = "";
  if (!state.item) return;
  const indices = state.item.assistant_message_indices || [];
  indices.forEach((idx) => {
    const chip = document.createElement("button");
    chip.className = "stepChip";
    chip.textContent = `a@${idx}`;
    const label = state.stepLabels[String(idx)];
    if (label === 1 || label === 0 || label === -1) chip.dataset.label = String(label);
    if (state.focusedAssistantIdx === idx) chip.dataset.focused = "true";
    chip.addEventListener("click", () => {
      state.focusedAssistantIdx = idx;
      renderStepsNav();
      renderMessages();
      const el = document.getElementById(`msg-${idx}`);
      if (el) el.scrollIntoView({ block: "center" });
    });
    root.appendChild(chip);
  });
}

function setItem(payload) {
  state.item = payload;
  state.llmReferences = payload.llm_references || null;
  state.stepLabels = (payload.existing_annotation && payload.existing_annotation.step_labels) || {};
  if (payload.existing_annotation) {
    const touched = payload.existing_annotation.final_label_touched ?? true;
    state.finalLabelTouched = Boolean(touched);
    state.finalLabel = state.finalLabelTouched ? payload.existing_annotation.final_label ?? 0 : null;
  } else {
    state.finalLabelTouched = false;
    state.finalLabel = null;
  }
  state.comment = (payload.existing_annotation && payload.existing_annotation.comment) || "";
  state.focusedAssistantIdx = (payload.assistant_message_indices && payload.assistant_message_indices[0]) || null;

  $("sampleMeta").textContent = stringify({
    dataset: payload.dataset,
    index_in_dataset: payload.index_in_dataset,
    record_id: payload.record_id,
    data_source: payload.data_source,
    query_index: payload.query_index,
    sample_index: payload.sample_index,
  });
  $("currentIndex").textContent = payload.index_in_dataset ?? "-";
  $("question").textContent = stringify(payload.question || "");
  $("taskDescription").textContent = stringify(payload.task_description || "");
  $("groundTruth").innerHTML = renderGroundTruthHtml(payload.ground_truth || "");
  $("rewardInfo").innerHTML = renderRewardInfoHtml(payload.reward_info || "");
  $("comment").value = state.comment;
  const tf = $("toolsFilter");
  if (tf) tf.value = state.toolsFilter;

  const rh = payload.reward_info && payload.reward_info.reward !== undefined ? payload.reward_info.reward : null;
  if (rh === null) {
    $("rewardHint").textContent = "";
    $("rewardHint").style.color = "";
  } else {
    $("rewardHint").textContent = `Reward hint (model/rule): ${String(rh)} (reference only)`;
    if (typeof rh === "number") {
      $("rewardHint").style.color = rh > 0 ? "var(--green)" : rh < 0 ? "var(--red)" : "var(--amber)";
    } else {
      $("rewardHint").style.color = "var(--muted)";
    }
  }

  renderFinalPills();
  renderStepsNav();
  renderMessages();
  renderToolsDocs();
  $("jumpIndex").value = String(payload.index_in_dataset ?? "");
}

async function refreshProgress() {
  if (!state.dataset || !state.annotator) {
    setStatus("-");
    return;
  }
  const p = await apiGet(`/api/progress?dataset=${encodeURIComponent(state.dataset)}&annotator=${encodeURIComponent(state.annotator)}`);
  setStatus(`${p.done} done · ${p.skipped} skipped · ${p.total} total`);
}

async function loadDatasets() {
  const data = await apiGet("/api/datasets");
  state.datasets = data.datasets || [];

  const sel = $("datasetSelect");
  sel.innerHTML = "";
  state.datasets.forEach((d) => {
    const opt = document.createElement("option");
    opt.value = d.name;
    opt.textContent = `${d.name} (${d.size})`;
    sel.appendChild(opt);
  });

  const saved = localStorage.getItem("apb_dataset");
  if (saved && state.datasets.some((d) => d.name === saved)) {
    sel.value = saved;
  }
  state.dataset = sel.value;
}

async function loadNext() {
  if (!state.dataset) return;
  if (!state.annotator) {
    showLogin();
    return;
  }
  const payload = await apiGet(
    `/api/next?dataset=${encodeURIComponent(state.dataset)}&annotator=${encodeURIComponent(state.annotator)}`
  );
  if (payload.done) {
    alert("This dataset is complete (or no samples are available).");
    return;
  }
  setItem(payload);
  await refreshProgress();
}

async function loadByIndex(index) {
  if (!state.dataset) return;
  const payload = await apiGet(
    `/api/item?dataset=${encodeURIComponent(state.dataset)}&index=${encodeURIComponent(String(index))}&annotator=${encodeURIComponent(
      state.annotator || ""
    )}`
  );
  setItem(payload);
  await refreshProgress();
}

function collectPayload(status) {
  const comment = $("comment").value || "";
  return {
    dataset: state.dataset,
    record_id: state.item.record_id,
    annotator: state.annotator,
    username: state.annotator,
    index_in_dataset: state.item.index_in_dataset,
    data_source: state.item.data_source,
    query_index: state.item.query_index,
    sample_index: state.item.sample_index,
    step_labels: state.stepLabels,
    final_label: isValidLabel(state.finalLabel) ? state.finalLabel : 0,
    final_label_touched: state.finalLabelTouched,
    status,
    comment,
  };
}

function validateDoneOrAlert() {
  if (!state.item) return false;
  const indices = state.item.assistant_message_indices || [];
  const missing = indices.filter((idx) => !isValidLabel(state.stepLabels[String(idx)]));
  if (missing.length > 0) {
    const head = missing.slice(0, 12).map((x) => `a@${x}`).join(", ");
    const more = missing.length > 12 ? ` …(+${missing.length - 12})` : "";
    alert(`Unlabeled assistant steps: ${head}${more}`);
    const first = missing[0];
    state.focusedAssistantIdx = first;
    renderStepsNav();
    renderMessages();
    const el = document.getElementById(`msg-${first}`);
    if (el) el.scrollIntoView({ block: "center" });
    return false;
  }
  if (!state.finalLabelTouched || !isValidLabel(state.finalLabel)) {
    alert('Please label the "Final Outcome" (+ / 0 / -) before saving.');
    return false;
  }
  return true;
}

async function save(status) {
  if (!state.item) {
    alert("No sample loaded.");
    return false;
  }
  if (status !== "skipped" && !validateDoneOrAlert()) return false;
  await apiPost("/api/annotation", collectPayload(status));
  await refreshProgress();
  return true;
}

function initEvents() {
  $("datasetSelect").addEventListener("change", async (e) => {
    state.dataset = e.target.value;
    localStorage.setItem("apb_dataset", state.dataset);
    await refreshProgress();
    await loadNext();
  });

  $("annotatorInput").addEventListener("change", async (e) => {
    state.annotator = (e.target.value || "").trim();
    if (state.rememberAnnotator && state.annotator) {
      localStorage.setItem("apb_annotator", state.annotator);
    } else {
      localStorage.removeItem("apb_annotator");
    }
    await refreshProgress();
    if (state.dataset && state.annotator) await loadNext();
  });

  $("loginStartBtn").addEventListener("click", async () => {
    const name = ($("loginName").value || "").trim();
    if (!name) return;
    _setRememberAnnotator(Boolean($("rememberAnnotator")?.checked));
    state.annotator = name;
    $("annotatorInput").value = name;
    if (state.rememberAnnotator) {
      localStorage.setItem("apb_annotator", state.annotator);
    } else {
      localStorage.removeItem("apb_annotator");
    }
    hideLogin();
    await refreshProgress();
    await loadNext();
  });

  $("loginName").addEventListener("keydown", async (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      $("loginStartBtn").click();
    }
  });

  document.querySelectorAll("[data-final]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const v = parseInt(btn.getAttribute("data-final"), 10);
      if (state.finalLabelTouched && state.finalLabel === v) {
        state.finalLabel = null;
        state.finalLabelTouched = false;
      } else {
        state.finalLabel = v;
        state.finalLabelTouched = true;
      }
      renderFinalPills();
      renderMessages();
    });
  });

  $("saveBtn").addEventListener("click", async () => {
    await save("in_progress");
  });

  $("saveNextBtn").addEventListener("click", async () => {
    const ok = await save("done");
    if (ok) await loadNext();
  });

  $("skipBtn").addEventListener("click", async () => {
    if (!confirm("Skip this sample?")) return;
    await save("skipped");
    await loadNext();
  });

  $("jumpBtn").addEventListener("click", async () => {
    const v = parseInt(($("jumpIndex").value || "0").trim(), 10);
    if (Number.isNaN(v)) return;
    await loadByIndex(v);
  });

  $("jumpIndex").addEventListener("keydown", async (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      $("jumpBtn").click();
    }
  });

  $("toolsFilter")?.addEventListener("input", (e) => {
    state.toolsFilter = (e.target.value || "").trim();
    renderToolsDocs();
  });

  document.addEventListener("keydown", (e) => {
    if (!state.item) return;
    const indices = state.item.assistant_message_indices || [];
    if (indices.length === 0) return;

    const focused = state.focusedAssistantIdx ?? indices[0];
    const pos = indices.indexOf(focused);
    const nextFocus = (delta) => {
      const p = Math.max(0, Math.min(indices.length - 1, (pos >= 0 ? pos : 0) + delta));
      state.focusedAssistantIdx = indices[p];
      renderStepsNav();
      renderMessages();
      const el = document.getElementById(`msg-${state.focusedAssistantIdx}`);
      if (el) el.scrollIntoView({ block: "center" });
    };

    if (e.key === "ArrowDown" || e.key === "j") {
      nextFocus(1);
      return;
    }
    if (e.key === "ArrowUp" || e.key === "k") {
      nextFocus(-1);
      return;
    }
    if (e.key === "1" || e.key === "+") {
      const k = String(focused);
      if (state.stepLabels[k] === 1) {
        delete state.stepLabels[k];
      } else {
        state.stepLabels[k] = 1;
      }
      state.focusedAssistantIdx = focused;
      renderStepsNav();
      renderMessages();
      return;
    }
    if (e.key === "0") {
      const k = String(focused);
      if (state.stepLabels[k] === 0) {
        delete state.stepLabels[k];
      } else {
        state.stepLabels[k] = 0;
      }
      state.focusedAssistantIdx = focused;
      renderStepsNav();
      renderMessages();
      return;
    }
    if (e.key === "-") {
      const k = String(focused);
      if (state.stepLabels[k] === -1) {
        delete state.stepLabels[k];
      } else {
        state.stepLabels[k] = -1;
      }
      state.focusedAssistantIdx = focused;
      renderStepsNav();
      renderMessages();
      return;
    }
  });
}

function initStaticCopyButtons() {
  attachCornerCopyButton($("sampleMeta")?.closest(".card"), () => $("sampleMeta")?.textContent || "");

  attachCornerCopyButton($("question")?.closest(".card"), () => {
    const q = $("question")?.textContent || "";
    const t = $("taskDescription")?.textContent || "";
    return [q, t].filter(Boolean).join("\n\n");
  });

  attachCornerCopyButton($("toolsPanel"), () => stringifyToolContent(state.item?.tools || ""));

  attachCornerCopyButton($("rewardHint")?.closest(".card"), () =>
    stringifyToolContent({
      final_label: state.finalLabelTouched && isValidLabel(state.finalLabel) ? state.finalLabel : null,
      final_label_touched: Boolean(state.finalLabelTouched),
    })
  );

  attachCornerCopyButton($("comment")?.closest(".card"), () => $("comment")?.value || "");

  attachCornerCopyButton($("groundTruth")?.closest(".card"), () =>
    stringifyToolContent({
      ground_truth: state.item?.ground_truth,
      reward_info: state.item?.reward_info,
    })
  );

  attachCornerCopyButton($("messages")?.closest(".card"), () => stringifyToolContent(state.item?.messages || ""));
}

async function main() {
  await loadDatasets();
  initEvents();
  initStaticCopyButtons();
  await refreshProgress();
  showLogin();
}

main().catch((e) => {
  console.error(e);
  alert(String(e));
});
