import { pipeline, env, AutoTokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/+esm";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1/+esm";
import * as pdfjsLib from "https://cdn.jsdelivr.net/npm/pdfjs-dist@4/build/pdf.min.mjs";
import mammoth from "https://esm.sh/mammoth@1.8.0";

pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4/build/pdf.worker.min.mjs";

// onnxruntime-web needs its own wasm runtime binaries (the execution engine
// itself, a few MB — not model weights) fetched from somewhere. This is the
// same kind of asset transformers.js already fetches internally for TMR's
// pipeline() path; it's not a new category of external dependency, just an
// explicit one now that we're calling onnxruntime-web directly for desklib.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1/dist/";

// ---- config -----------------------------------------------------------
const MIN_WORDS_FOR_RELIABLE = 40;

const MODELS = {
  tmr: {
    folder: "tmr-ai-text-detector",
    label: "Fast — TMR (RoBERTa-base)",
    architecture: "pipeline",       // standard 2-class softmax via transformers.js pipeline()
    sizeMB: 126,
    dtype: "int8",
    maxTokens: 400,
  },
  desklib: {
    folder: "desklib-ai-text-detector",
    label: "Thorough — Desklib (DeBERTa-v3-large, fp16)",
    architecture: "raw-sigmoid",    // custom single-logit head, run via raw ONNX session
    onnxFile: "model_fp16.onnx",
    sizeMB: 830,                    // measured — matches the actual exported file
    dtype: "fp16",
    maxTokens: 700,
  },
};

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = "./models/";

let loadedModel = null; // { key, architecture, tokenizer, classifier?, aiLabelKey?, session? }

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const dropzone = $("dropzone");
const fileInput = $("fileInput");
const textInput = $("textInput");
const analyzeBtn = $("analyzeBtn");

function setStatus(msg) { statusEl.textContent = msg; }
function wordCount(t) { return t.split(/\s+/).filter(Boolean).length; }

// ---- device/connection capability estimate -------------------------------
function estimateCapability(modelKey) {
  const m = MODELS[modelKey];
  const lines = [];

  const conn = navigator.connection;
  if (conn && conn.downlink) {
    const seconds = (m.sizeMB * 8) / conn.downlink;
    lines.push(`Estimated download at your current connection: ≈${formatDuration(seconds)}`);
  } else {
    lines.push(`Estimated download: ≈${formatDuration((m.sizeMB * 8) / 10)} on slow connections, ≈${formatDuration((m.sizeMB * 8) / 50)} on typical broadband (exact speed not available in this browser)`);
  }

  const estimatedRamNeedGB = (m.sizeMB * 2) / 1024;
  if (navigator.deviceMemory) {
    lines.push(`Estimated memory needed: ≈${estimatedRamNeedGB.toFixed(1)}GB (your device reports ≈${navigator.deviceMemory}GB total)`);
    if (navigator.deviceMemory < estimatedRamNeedGB * 1.5) {
      lines.push(`⚠ This may be tight on your current device — the tab could slow down or be closed by the browser.`);
    }
  } else {
    lines.push(`Estimated memory needed: ≈${estimatedRamNeedGB.toFixed(1)}GB (can't detect your device's total RAM in this browser — Safari and Firefox don't expose it)`);
  }

  if (navigator.gpu) {
    lines.push(`WebGPU is available — actual VRAM can't be read from the browser, so if the model fails to load on GPU it'll automatically fall back to CPU.`);
  }
  if (modelKey === "desklib") {
    lines.push(`This model runs one chunk at a time (no batching), so larger documents take noticeably longer than the fast model.`);
  }

  return lines;
}

function formatDuration(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  return `${Math.round(seconds / 60)} min`;
}

// ---- file text extraction ----------------------------------------------
function extractRtfText(rtf) {
  return rtf
    .replace(/\\'[0-9a-fA-F]{2}/g, "")
    .replace(/\{\*?\\[^{}]+\}|[{}]|\\[A-Za-z]+-?\d* ?/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

async function extractPdfText(arrayBuffer) {
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  let text = "";
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    text += content.items.map((item) => item.str).join(" ") + "\n\n";
  }
  return text.trim();
}

async function extractDocxText(arrayBuffer) {
  const { value } = await mammoth.extractRawText({ arrayBuffer });
  return value.trim();
}

async function extractText(file) {
  const name = file.name.toLowerCase();
  if (name.endsWith(".doc")) {
    throw new Error("Legacy .doc isn't supported in-browser. Save as .docx (or paste the text) and try again.");
  }
  if (name.endsWith(".pdf")) return extractPdfText(await file.arrayBuffer());
  if (name.endsWith(".docx")) return extractDocxText(await file.arrayBuffer());
  if (name.endsWith(".rtf")) return extractRtfText(await file.text());
  if (name.endsWith(".txt") || name.endsWith(".md") || file.type.startsWith("text/")) return file.text();
  throw new Error(`Unsupported file type: ${file.name}`);
}

// ---- drag/drop + file input ---------------------------------------------
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", async (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  const f = e.dataTransfer.files[0];
  if (f) await loadFile(f);
});
fileInput.addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (f) await loadFile(f);
});

async function loadFile(file) {
  setStatus(`Reading ${file.name}…`);
  try {
    const text = await extractText(file);
    textInput.value = text;
    $("dzLabel").textContent = `Loaded: ${file.name} (${wordCount(text)} words)`;
    setStatus("");
  } catch (err) {
    setStatus(`Could not read ${file.name}: ${err.message}`);
  }
}

$("clearBtn").addEventListener("click", () => {
  textInput.value = "";
  fileInput.value = "";
  $("dzLabel").textContent = "Drop a file here, or click to choose one (.txt, .md, .pdf, .docx, .rtf)";
  $("resultsPanel").classList.add("hidden");
});

// ---- model loading --------------------------------------------------------
async function ensureModel() {
  const key = $("modelSelect").value;
  if (loadedModel && loadedModel.key === key) return loadedModel;

  const cfg = MODELS[key];
  const deviceChoice = $("deviceSelect").value;
  const webgpuOK = !!navigator.gpu;
  const device = deviceChoice === "auto" ? (webgpuOK ? "webgpu" : "wasm") : deviceChoice;

  if (cfg.architecture === "raw-sigmoid") {
    setStatus(`Loading ${cfg.label} on ${device}…`);
    const tokenizer = await AutoTokenizer.from_pretrained(cfg.folder);

    let session;
    const modelUrl = `./models/${cfg.folder}/onnx/${cfg.onnxFile}`;
    try {
      session = await ort.InferenceSession.create(modelUrl, { executionProviders: [device] });
    } catch (err) {
      setStatus(`Failed to load on ${device} (${err.message}). Retrying on CPU (WASM)…`);
      session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"] });
    }

    loadedModel = { key, architecture: "raw-sigmoid", tokenizer, session };
    setStatus(`Model ready. (${cfg.label})`);
    return loadedModel;
  }

  // Standard pipeline() path — TMR, or any future 2-class softmax model.
  setStatus(`Loading ${cfg.label} (dtype=${cfg.dtype}) on ${device}…`);
  let classifier;
  try {
    classifier = await pipeline("text-classification", cfg.folder, { device, dtype: cfg.dtype });
  } catch (err) {
    setStatus(`Failed to load on ${device} (${err.message}). Retrying on CPU (WASM)…`);
    classifier = await pipeline("text-classification", cfg.folder, { device: "wasm", dtype: cfg.dtype });
  }

  const id2label = classifier.model.config.id2label || {};
  const entries = Object.entries(id2label);
  const aiEntry = entries.find(([, v]) => /ai|machine|generated|gpt|llm|fake|synthetic/i.test(v));
  const aiLabelKey = aiEntry ? aiEntry[1] : (entries[1] ? entries[1][1] : entries[0]?.[1]);

  loadedModel = { key, architecture: "pipeline", tokenizer: classifier.tokenizer, classifier, aiLabelKey };
  setStatus(`Model ready on ${device}. AI label resolved to "${aiLabelKey}".`);
  return loadedModel;
}

// ---- segmentation & chunking ---------------------------------------------
function splitSentences(text) {
  if (typeof Intl !== "undefined" && Intl.Segmenter) {
    const seg = new Intl.Segmenter("en", { granularity: "sentence" });
    return Array.from(seg.segment(text), (s) => ({ text: s.segment, start: s.index }));
  }
  const out = [];
  const re = /[^.!?]+[.!?]+(\s+|$)|[^.!?]+$/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m[0].trim().length) out.push({ text: m[0], start: m.index });
  }
  return out;
}

function estimateTokens(tokenizer, text) {
  try {
    if (typeof tokenizer.encode === "function") return tokenizer.encode(text).length;
  } catch (e) {}
  return Math.ceil(wordCount(text) * 1.3);
}

async function buildChunks(text, tokenizer, maxTokens) {
  const sentences = splitSentences(text).filter((s) => s.text.trim().length > 0);
  const chunks = [];
  for (const s of sentences) {
    const tokLen = estimateTokens(tokenizer, s.text);
    if (tokLen <= maxTokens) {
      chunks.push({ text: s.text, start: s.start, end: s.start + s.text.length, tokenCount: tokLen });
    } else {
      const words = s.text.split(/(\s+)/);
      let buf = "", bufStart = s.start, cursor = s.start;
      for (const w of words) {
        buf += w;
        if (wordCount(buf) >= 150) {
          chunks.push({ text: buf, start: bufStart, end: cursor + w.length, tokenCount: null });
          bufStart = cursor + w.length; buf = "";
        }
        cursor += w.length;
      }
      if (buf.trim()) chunks.push({ text: buf, start: bufStart, end: cursor, tokenCount: null });
    }
  }
  return chunks;
}

// ---- inference ------------------------------------------------------------
function toInt64Tensor(data, dims) {
  const big = data instanceof BigInt64Array
    ? data
    : BigInt64Array.from(Array.from(data), (x) => BigInt(x));
  return new ort.Tensor("int64", big, dims);
}

async function runDesklibChunk(loadedModel, text, maxTokens) {
  const enc = await loadedModel.tokenizer(text, { padding: true, truncation: true, max_length: maxTokens });
  const feeds = {
    input_ids: toInt64Tensor(enc.input_ids.data, enc.input_ids.dims),
    attention_mask: toInt64Tensor(enc.attention_mask.data, enc.attention_mask.dims),
  };
  const out = await loadedModel.session.run(feeds);
  const logit = Number(out.logits.data[0]);
  return 1 / (1 + Math.exp(-logit)); // single sigmoid logit -> AI probability
}

async function classifyChunks(loadedModel, chunks, maxTokens) {
  const results = [];

  if (loadedModel.architecture === "raw-sigmoid") {
    // No batching here — desklib's custom head takes one text at a time.
    // Chunks are already larger (700 tok) and fewer than the fast model's,
    // so the batching win would be smaller anyway; simplicity wins here.
    for (let i = 0; i < chunks.length; i++) {
      const aiScore = await runDesklibChunk(loadedModel, chunks[i].text, maxTokens);
      results.push({ ...chunks[i], aiProbability: aiScore });
      setStatus(`Classified ${i + 1} / ${chunks.length} chunk(s)…`);
    }
    return results;
  }

  const BATCH = 8;
  const numLabels = Object.keys(loadedModel.classifier.model.config.id2label || { 0: 0, 1: 1 }).length;
  for (let i = 0; i < chunks.length; i += BATCH) {
    const batch = chunks.slice(i, i + BATCH);
    const texts = batch.map((c) => c.text);
    const out = await loadedModel.classifier(texts, { top_k: numLabels });
    batch.forEach((c, idx) => {
      const scores = out[idx];
      let aiScore;
      if (Array.isArray(scores)) {
        const found = scores.find((s) => s.label === loadedModel.aiLabelKey);
        aiScore = found ? found.score : (scores[0].label === loadedModel.aiLabelKey ? scores[0].score : 1 - scores[0].score);
      } else {
        aiScore = scores.label === loadedModel.aiLabelKey ? scores.score : 1 - scores.score;
      }
      results.push({ ...c, aiProbability: aiScore });
    });
    setStatus(`Classified ${Math.min(i + BATCH, chunks.length)} / ${chunks.length} chunk(s)…`);
  }
  return results;
}

function aggregate(results, method) {
  if (method === "sentence") {
    return results.reduce((a, r) => a + r.aiProbability, 0) / results.length;
  }
  let num = 0, den = 0;
  for (const r of results) {
    const w = r.tokenCount ?? wordCount(r.text);
    num += r.aiProbability * w;
    den += w;
  }
  return den ? num / den : 0;
}

function band(p) {
  const pct = p * 100;
  if (pct < 20) return { label: "Low AI likelihood", color: "var(--green)" };
  if (pct < 40) return { label: "Some AI-like characteristics", color: "var(--green)" };
  if (pct < 60) return { label: "Uncertain", color: "var(--amber)" };
  if (pct < 80) return { label: "Elevated AI likelihood", color: "var(--red)" };
  return { label: "High AI likelihood", color: "var(--red)" };
}

function heatColor(p) {
  const hue = 130 - 130 * p;
  return `hsla(${hue}, 65%, 45%, ${0.15 + 0.5 * p})`;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function buildHeatHtml(fullText, sortedResults) {
  let html = "", cursor = 0;
  for (const r of sortedResults) {
    if (r.start > cursor) html += escapeHtml(fullText.slice(cursor, r.start));
    html += `<span style="background:${heatColor(r.aiProbability)}" title="AI probability: ${Math.round(r.aiProbability * 100)}%">${escapeHtml(r.text)}</span>`;
    cursor = r.end;
  }
  if (cursor < fullText.length) html += escapeHtml(fullText.slice(cursor));
  return html;
}

// ---- render ----------------------------------------------------------------
function renderResults(fullText, results, overall, method) {
  $("resultsPanel").classList.remove("hidden");
  const pct = Math.round(overall * 100);
  $("aiBar").style.width = pct + "%";
  $("aiPct").textContent = pct + "%";
  $("humanPct").textContent = (100 - pct) + "%";
  const b = band(overall);
  $("bandLabel").textContent = b.label;
  $("bandLabel").style.background = b.color;
  $("bandLabel").style.color = "#0f1115";
  $("confidenceNote").textContent = `${results.length} chunk(s) · ${method === "token" ? "token-weighted" : "sentence-weighted"} aggregation · heuristic bands, not independently validated`;

  const sorted = [...results].sort((a, b2) => a.start - b2.start);
  $("heatDiv").innerHTML = buildHeatHtml(fullText, sorted);

  const tbody = $("chunkTableBody");
  tbody.innerHTML = "";
  sorted.forEach((r, i) => {
    const p = Math.round(r.aiProbability * 100);
    const bd = band(r.aiProbability);
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i + 1}</td><td>${escapeHtml(r.text.slice(0, 80))}${r.text.length > 80 ? "…" : ""}</td><td>${r.tokenCount ?? "≈" + wordCount(r.text)}</td><td>${p}%</td><td>${bd.label}</td>`;
    tbody.appendChild(tr);
  });

  window.__lastResult = { overall, method, pct, band: b.label, generatedAt: new Date().toISOString(), fullText, results: sorted };
}

// ---- export / report --------------------------------------------------------
function download(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

function buildStandaloneReport(data) {
  const heatHtml = buildHeatHtml(data.fullText, data.results);
  const rows = data.results.map((r, i) => {
    const p = Math.round(r.aiProbability * 100);
    return `<tr><td>${i + 1}</td><td>${escapeHtml(r.text.slice(0, 100))}${r.text.length > 100 ? "…" : ""}</td><td>${p}%</td><td>${band(r.aiProbability).label}</td></tr>`;
  }).join("");

  return `<!DOCTYPE html><html><head><meta charset="utf-8"><title>AI text detection report</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:820px;margin:32px auto;padding:0 16px;color:#1a1a1a}
h1{font-size:20px}h2{font-size:15px;margin-top:24px}
.meta{color:#666;font-size:13px}
.bar-outer{height:20px;background:#eee;border-radius:999px;overflow:hidden;border:1px solid #ddd}
.bar-ai{height:100%;background:linear-gradient(90deg,#e0a52e,#e0524b);width:${data.pct}%}
.band{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;background:#eee}
.heat{line-height:2;font-size:14px;padding:12px;background:#fafafa;border:1px solid #ddd;border-radius:8px;white-space:pre-wrap}
.heat span{padding:1px 2px;border-radius:3px}
table{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #ddd}
.disclaimer{font-size:12px;color:#555;border-left:3px solid #e0a52e;padding:8px 12px;background:#fafafa;margin-top:14px}
</style></head><body>
<h1>AI text detection report</h1>
<p class="meta">Generated ${data.generatedAt} · ${data.results.length} chunks · ${data.method}-weighted aggregation</p>
<div class="bar-outer"><div class="bar-ai"></div></div>
<p>AI likelihood: <strong>${data.pct}%</strong> · Human likelihood: <strong>${100 - data.pct}%</strong> · <span class="band">${data.band}</span></p>
<div class="disclaimer">This is a statistical model estimate, not proof of authorship. False positives are more likely on formulaic, academic, non-native-English, heavily edited, or very short text. False negatives are more likely after paraphrasing, translation, or human editing.</div>
<h2>Highlighted text</h2>
<div class="heat">${heatHtml}</div>
<h2>Chunk breakdown</h2>
<table><thead><tr><th>#</th><th>Text</th><th>AI probability</th><th>Band</th></tr></thead><tbody>${rows}</tbody></table>
</body></html>`;
}

$("exportJson").addEventListener("click", () => {
  download("ai-detection-result.json", JSON.stringify(window.__lastResult, null, 2), "application/json");
});
$("exportCsv").addEventListener("click", () => {
  const data = window.__lastResult;
  const rows = [["#", "text", "tokens", "ai_probability", "band"]];
  data.results.forEach((r, i) => rows.push([i + 1, r.text.replace(/"/g, '""'), r.tokenCount ?? "", r.aiProbability.toFixed(4), band(r.aiProbability).label]));
  const csv = rows.map((row) => row.map((v) => `"${v}"`).join(",")).join("\n");
  download("ai-detection-result.csv", csv, "text/csv");
});
$("exportHtml").addEventListener("click", () => {
  download("ai-detection-report.html", buildStandaloneReport(window.__lastResult), "text/html");
});
$("printReport").addEventListener("click", () => window.print());

// ---- model selector ---------------------------------------------------------
for (const [key, cfg] of Object.entries(MODELS)) {
  const opt = document.createElement("option");
  opt.value = key;
  opt.textContent = `${cfg.label} — ≈${cfg.sizeMB}MB`;
  $("modelSelect").appendChild(opt);
}

function renderModelInfo() {
  const lines = estimateCapability($("modelSelect").value);
  $("modelInfo").innerHTML = lines.map((l) => `<p>${l}</p>`).join("");
}
$("modelSelect").addEventListener("change", renderModelInfo);
renderModelInfo();

// ---- main flow ---------------------------------------------------------------
analyzeBtn.addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (!text) { setStatus("Paste text or attach a file first."); return; }

  const wc = wordCount(text);
  if (wc < MIN_WORDS_FOR_RELIABLE && !$("forceShort").checked) {
    $("resultsPanel").classList.remove("hidden");
    $("heatDiv").innerHTML = ""; $("chunkTableBody").innerHTML = "";
    $("aiPct").textContent = "–"; $("humanPct").textContent = "–"; $("aiBar").style.width = "0%";
    $("bandLabel").textContent = "Insufficient text";
    $("bandLabel").style.background = "var(--muted)"; $("bandLabel").style.color = "#0f1115";
    $("confidenceNote").textContent = `${wc} words found, ≈${MIN_WORDS_FOR_RELIABLE}+ recommended. Check "analyze anyway" to run regardless.`;
    return;
  }

  analyzeBtn.disabled = true;
  try {
    const model = await ensureModel();
    const cfg = MODELS[$("modelSelect").value];
    setStatus("Segmenting text…");
    const chunks = await buildChunks(text, model.tokenizer, cfg.maxTokens);
    setStatus(`Running inference on ${chunks.length} chunk(s)…`);
    const results = await classifyChunks(model, chunks, cfg.maxTokens);
    const method = $("aggSelect").value;
    const overall = aggregate(results, method);
    renderResults(text, results, overall, method);
    setStatus(`Done. ${results.length} chunk(s) analyzed.`);
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`);
  } finally {
    analyzeBtn.disabled = false;
  }
});