"use strict";
const element = id => document.getElementById(id);
const number = value => Number.isFinite(value) ? value.toLocaleString("zh-CN") : "—";
const gib = value => Number.isFinite(value) ? (value / 2 ** 30).toFixed(2) + " GiB" : "—";
const healthNames = {ok: "运行中", degraded: "部分异常", unhealthy: "后端异常", shutting_down: "正在停止"};
const set = (id, value) => { element(id).textContent = value; };
let apiKey = "";
let generation = null;
let refreshID = 0;
let previousSample = null;
let healthModels = [];
let apiModels = [];
let needsKey = false;

function navigate() {
  const page = ["models", "chat"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "status";
  for (const id of ["status", "models", "chat"]) element(id).hidden = id !== page;
  for (const link of document.querySelectorAll("nav a")) {
    if (link.hash === "#" + page) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  document.title = "MIV · " + element(page).querySelector("h1").textContent;
}
window.addEventListener("hashchange", () => { navigate(); refresh(); });
navigate();
set("base", location.origin + "/v1");

function modelRows(models) {
  const list = element("model-list");
  list.replaceChildren();
  if (!models.length) { list.textContent = "暂无可显示的模型。"; return; }
  for (const model of models) {
    const article = document.createElement("article");
    article.className = "model";
    const title = document.createElement("h3");
    title.textContent = model.id;
    const details = document.createElement("dl");
    const fields = [
      ["运行状态", healthNames[model.status] || (model.status === "error" ? "后端异常" : "—")],
      ["后端", model.backend || model.provider_id], ["上下文长度", number(model.max_context)],
      ["最大并发", number(model.max_concurrency)], ["KV 类型", model.kv_dtype],
      ["推测解码", model.spec_backend], ["驻留内存", gib(model.resident_memory_bytes)]
    ];
    if (model.lora_adapter) fields.push(["LoRA 适配器", model.lora_adapter], ["LoRA 模式", model.lora_mode]);
    for (const [label, value] of fields) {
      const pair = document.createElement("div");
      const term = document.createElement("dt");
      const definition = document.createElement("dd");
      term.textContent = label;
      definition.textContent = value || "—";
      pair.append(term, definition);
      details.append(pair);
    }
    article.append(title, details);
    if (model.error) {
      const error = document.createElement("p");
      error.className = "error";
      error.textContent = model.error;
      article.append(error);
    }
    list.append(article);
  }
}

function renderHealth(health) {
  const state = health?.status;
  set("health", healthNames[state] || "未连接");
  element("health").dataset.state = state === "ok" ? "ok" : "error";
  set("version", health?.version || "—");
  set("server-version", health?.backend || (health ? "后端状态待恢复" : "无法读取核心状态"));
  set("capacity", number(health?.max_concurrency));
  healthModels = health?.models || [];
  modelRows(healthModels);
}

function renderMetrics(text) {
  const metrics = {};
  for (const line of text.split("\n")) {
    const match = /^ninfer_metal_([a-z_]+) ([0-9.eE+-]+)$/.exec(line.trim());
    if (match) metrics[match[1]] = Number(match[2]);
  }
  set("active", number(metrics.active_requests));
  set("queued", number(metrics.queued_requests));
  set("count", number(metrics.loaded_models));
  set("memory", gib(metrics.phys_footprint_bytes));
  set("requests", number(metrics.requests_succeeded_total) + " / " + number(metrics.requests_total));
  set("input-tokens", number(metrics.prompt_tokens_total));
  set("output-tokens", number(metrics.generated_tokens_total));
  set("cache", number(metrics.prefix_cache_restored_tokens_total));
  const now = performance.now();
  const tokens = metrics.generated_tokens_total;
  let rate = "—";
  if (Number.isFinite(tokens)) {
    if (previousSample && tokens >= previousSample.tokens && now > previousSample.time) {
      rate = ((tokens - previousSample.tokens) * 1000 / (now - previousSample.time)).toFixed(1) + " tok/s";
    }
    previousSample = {tokens, time: now};
  } else {
    previousSample = null;
  }
  set("rate", rate);
}

function updateChatControls() {
  const busy = Boolean(generation);
  element("send").disabled = busy || !element("model").value || !element("prompt").value.trim();
  element("model").disabled = busy || !element("model").options.length;
  element("limit").disabled = busy;
  element("prompt").disabled = busy;
  element("cancel").hidden = !busy;
}

function renderPicker() {
  const picker = element("model");
  const ids = apiModels.map(model => model.id).filter(id => healthModels.some(model => model.id === id && model.status === "ok"));
  if (JSON.stringify(ids) !== JSON.stringify(Array.from(picker.options, o => o.value))) {
    const selected = picker.value;
    picker.replaceChildren(...ids.map(id => new Option(id, id)));
    if (ids.includes(selected)) picker.value = selected;
  }
  updateChatControls();
}

async function fetchData(path, type, allowedStatus = 0) {
  const headers = path.startsWith("/v1/") && apiKey ? {Authorization: "Bearer " + apiKey} : {};
  const response = await fetch(path, {headers, cache: "no-store", redirect: "error", signal: AbortSignal.timeout(10000)});
  if (!response.ok && response.status !== allowedStatus) {
    const error = new Error("HTTP " + response.status);
    error.status = response.status;
    throw error;
  }
  return type === "text" ? response.text() : response.json();
}

async function refresh() {
  const id = ++refreshID;
  const checkModels = !element("chat").hidden && !needsKey;
  const [health, metrics, models] = await Promise.allSettled([
    fetchData("/health", "json", 503), fetchData("/metrics", "text"), checkModels ? fetchData("/v1/models", "json") : null
  ]);
  if (id !== refreshID) return;
  renderHealth(health.value);
  renderMetrics(metrics.value || "");
  set("metrics-error", metrics.status === "rejected" ? "无法读取性能指标。" : "");
  set("updated", new Date().toLocaleTimeString("zh-CN", {hour12: false}));
  if (checkModels) {
    needsKey = models.reason?.status === 401;
    element("credentials").hidden = !needsKey;
    if (needsKey) {
      set("auth-hint", apiKey ? "密钥无效，请重新输入。" : "此服务需要 API 密钥，密钥仅在本页有效。");
      apiKey = "";
    }
    apiModels = models.value?.data || [];
    const hint = models.status === "rejected" && !needsKey ? "无法读取模型列表。" : "";
    set("picker-hint", hint || (!needsKey && !healthModels.some(model => model.status === "ok") ? "暂无可用模型。" : ""));
  }
  renderPicker();
}
element("credentials").addEventListener("submit", event => {
  event.preventDefault();
  apiKey = element("key").value.trim();
  needsKey = false;
  element("key").value = "";
  refresh();
});
element("refresh").addEventListener("click", refresh);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) { previousSample = null; refresh(); }
});
async function poll() {
  if (!document.hidden) await refresh();
  setTimeout(poll, 5000);
}
poll();

async function readCompletion(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let pending = "";
  let usage, finishReason;
  let complete = false;
  const reasoning = element("reasoning"), answer = element("answer");
  try {
    while (!complete) {
      const {value, done} = await reader.read();
      if (done) break;
      pending += value;
      const lines = pending.split("\n");
      pending = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") { complete = true; break; }
        const chunk = JSON.parse(data);
        if (chunk.error) throw new Error(chunk.error.message);
        const choice = chunk.choices?.[0];
        if (choice?.delta?.reasoning_content) {
          element("thinking").hidden = false;
          reasoning.append(choice.delta.reasoning_content);
        }
        if (choice?.delta?.content) answer.append(choice.delta.content);
        if (choice?.finish_reason) finishReason = choice.finish_reason;
        if (chunk.usage) usage = chunk.usage;
      }
    }
  } finally { reader.releaseLock(); }
  if (!complete) throw new Error("连接中断，回复未完成。");
  const detail = finishReason === "length" ? "达到输出上限" : "生成完成";
  return usage ? detail + " · 输入 " + number(usage.prompt_tokens) + " / 输出 " + number(usage.completion_tokens) + " token" : detail;
}

element("prompt").addEventListener("input", updateChatControls);
element("cancel").addEventListener("click", () => generation?.abort());
element("generate").addEventListener("submit", async event => {
  event.preventDefault();
  if (generation || !element("model").value || !element("prompt").value.trim()) return;
  generation = new AbortController();
  set("reasoning", ""); set("answer", ""); set("result-status", "正在生成…");
  element("thinking").hidden = true;
  element("reply").hidden = false;
  updateChatControls();
  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST", signal: generation.signal, redirect: "error",
      headers: {"Content-Type": "application/json", ...(apiKey ? {Authorization: "Bearer " + apiKey} : {})},
      body: JSON.stringify({model: element("model").value, messages: [{role: "user", content: element("prompt").value}],
        max_completion_tokens: Number(element("limit").value), stream: true, stream_options: {include_usage: true}})
    });
    if (!response.ok) {
      const error = await response.json().catch(() => null);
      throw new Error(error?.error?.message || "HTTP " + response.status);
    }
    set("result-status", await readCompletion(response));
  } catch (error) {
    set("result-status", generation.signal.aborted ? "已停止生成" : "请求失败 · " + error.message);
  } finally {
    generation = null;
    updateChatControls();
    refresh();
  }
});
