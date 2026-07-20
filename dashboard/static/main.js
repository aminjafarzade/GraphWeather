/* gw-dashboard SPA — a pure display layer (invariant I1).
   Every number rendered here arrived verbatim from the backend API.
   The only "logic" below is option-list construction and formatting. */
"use strict";

/* ---------- palette (validated 8-slot categorical, light/dark) ---------- */
const PALETTES = {
  light: { cat: ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"],
           ink: "#0b0b0b", ink2: "#52514e", muted: "#898781", grid: "#e1e0d9", axis: "#c3c2b7",
           surface: "#fcfcfb" },
  dark:  { cat: ["#3987e5", "#008300", "#d55181", "#c98500", "#199e70", "#d95926", "#9085e9", "#e66767"],
           ink: "#ffffff", ink2: "#c3c2b7", muted: "#898781", grid: "#2c2c2a", axis: "#383835",
           surface: "#1a1a19" },
  /* warm ivory theme; same validated categorical hues as light (the cream
     surface sits close to the light surface the palette was validated on) */
  claude: { cat: ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"],
            ink: "#191714", ink2: "#57503f", muted: "#8f8676", grid: "#e2dcc9", axis: "#cdc2a8",
            surface: "#faf9f5" },
};
const THEME_CYCLE = ["light", "dark", "claude"];
const THEME_ICON = { light: "◐ Light", dark: "◑ Dark", claude: "✳ Claude" };
const theme = () => {
  const t = document.documentElement.dataset.theme;
  return PALETTES[t] ? t : "light";
};
const pal = () => PALETTES[theme()];

const STATUS_LABEL = {
  in_progress: "training in progress",
  trained: "awaiting evaluation",
  evaluated: "evaluated",
  invalid: "invalid",
  removed: "removed",
};

/* ---------- state ---------- */
const state = {
  view: "runs",
  meta: null,
  runs: [],                    // summaries from /api/runs
  detailRunId: null,
  detailEval: null,
  detailVar: null,
  cmp: { selected: new Set(), eval: "primary", variable: null, metric: "rmse",
         lead: null, persistence: true, externals: new Set() },
  tree: { nodes: [], selected: null, mode: "view" },   // mode: view | edit | add
};

const $ = (id) => document.getElementById(id);
const fmt = (v, dig = 4) => (v === null || v === undefined) ? "—"
  : (typeof v === "number" ? (+v).toPrecision(dig).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "") : String(v));

async function getJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${url} -> ${resp.status}`);
  return resp.json();
}
async function postJSON(url, body) {
  const resp = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                                  body: JSON.stringify(body) });
  if (!resp.ok) throw new Error(`${url} -> ${resp.status}`);
  return resp.json();
}

function runColor(runId) {
  const i = state.runs.findIndex((r) => r.run_id === runId);
  return i >= 0 && i < pal().cat.length ? pal().cat[i] : pal().muted;
}

/* ---------- plotly helpers ---------- */
const PLOT_CONF = { displayModeBar: false, responsive: true };
function baseLayout(xTitle, yTitle, extra = {}) {
  const p = pal();
  return Object.assign({
    margin: { l: 62, r: 14, t: 8, b: 88 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, system-ui, sans-serif", size: 12, color: p.ink2 },
    xaxis: { title: { text: xTitle }, gridcolor: p.grid, zeroline: false,
             linecolor: p.axis, tickcolor: p.axis, dtick: 1 },
    yaxis: { title: { text: yTitle }, gridcolor: p.grid, zeroline: false,
             linecolor: p.axis, tickcolor: p.axis },
    hovermode: "x unified",
    hoverlabel: { bgcolor: p.surface, bordercolor: p.axis,
                  font: { family: "Inter, sans-serif", size: 12, color: p.ink } },
    legend: { orientation: "h", y: -0.3, yanchor: "top", font: { size: 11.5 } },
  }, extra);
}
function hexAlpha(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
/* series + optional CI band, all values verbatim from the API */
function seriesTraces(leads, s, color, name, opts = {}) {
  const traces = [];
  const emphasized = !!opts.winner;
  if (s.ci_lower && s.ci_upper) {
    traces.push({ x: leads, y: s.ci_upper, mode: "lines", line: { width: 0 },
                  hoverinfo: "skip", showlegend: false });
    traces.push({ x: leads, y: s.ci_lower, mode: "lines", line: { width: 0 },
                  fill: "tonexty", fillcolor: hexAlpha(color, 0.13),
                  hoverinfo: "skip", showlegend: false });
  }
  traces.push({ x: leads, y: s.values,
                name: emphasized ? `▶ ${name}` : name,
                mode: "lines+markers",
                line: { color, width: emphasized ? 4 : 2, dash: opts.dash || "solid" },
                marker: { size: emphasized ? 7 : 5, color },
                /* exact stored values on hover */
                hovertemplate: `${name}: %{y:.6g}<extra>${emphasized ? "winner" : ""}</extra>` });
  return traces;
}
function overlayTraces(payload, winnerId = null) {
  const traces = [];
  for (const s of payload.series)
    traces.push(...seriesTraces(payload.lead_times, s, runColor(s.run_id), s.label,
                                { winner: s.run_id === winnerId }));
  for (const b of payload.baselines || [])
    traces.push({ x: payload.lead_times, y: b.values,
                  name: (b.id === winnerId ? "▶ " : "") + b.label, mode: "lines",
                  line: { color: b.external ? pal().ink2 : pal().muted,
                          width: b.id === winnerId ? 3.5 : 1.6,
                          dash: b.external ? "dashdot" : "dash" },
                  hovertemplate: `${b.label}: %{y:.6g}<extra>${b.id === winnerId ? "winner" : ""}</extra>` });
  return traces;
}

/* ---------- header / tabs / theme ---------- */
$("tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (b) setView(b.dataset.view);
});
$("backToRuns").addEventListener("click", () => setView("runs"));
function setView(name) {
  state.view = name;
  document.querySelectorAll("nav.tabs button").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll("section.view").forEach((s) =>
    s.classList.toggle("active", s.id === "view-" + (name === "detail" ? "detail" : name)));
  render();
}
function updateThemeButton() {
  $("themeToggle").textContent = THEME_ICON[theme()] || "◐ Theme";
}
$("themeToggle").addEventListener("click", () => {
  const next = THEME_CYCLE[(THEME_CYCLE.indexOf(theme()) + 1) % THEME_CYCLE.length];
  document.documentElement.dataset.theme = next;
  localStorage.setItem("gw-dash-theme", next);
  updateThemeButton();
  render();
});
updateThemeButton();

/* ---------- data refresh + SSE ---------- */
async function refreshData() {
  const [meta, runsPayload] = await Promise.all([getJSON("/api/meta"), getJSON("/api/runs")]);
  state.meta = meta;
  state.runs = runsPayload.runs;
  $("schemaVersion").textContent = meta.schema_version;
  const evaluated = state.runs.filter((r) => r.status === "evaluated");
  if (state.cmp.selected.size === 0)
    evaluated.slice(0, 4).forEach((r) => state.cmp.selected.add(r.run_id));
  for (const id of [...state.cmp.selected])
    if (!evaluated.some((r) => r.run_id === id)) state.cmp.selected.delete(id);
  if (!state.cmp.variable) state.cmp.variable = meta.variables_union[0] || null;
  const errs = (meta.problem_totals || {}).error || 0;
  const warns = (meta.problem_totals || {}).warning || 0;
  const badge = $("problemBadge");
  badge.style.display = errs + warns ? "inline-block" : "none";
  badge.classList.toggle("warn-only", errs === 0);
  badge.textContent = errs ? `${errs}E/${warns}W` : `${warns}W`;
  $("statusPill").textContent =
    `${state.runs.length} runs · sweep ${meta.scan.last_sweep ? meta.scan.last_sweep.slice(11, 19) : "…"}`;
  $("footer").textContent =
    `scanning ${meta.scan.root} every ${meta.scan.interval_s ?? "—"}s · units are declared, not derived · ` +
    `backend is the single source of truth`;
}
let refreshTimer = null;
function scheduleRefresh() {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(async () => { await refreshData(); render(); }, 400);
}
function connectSSE() {
  try {
    const es = new EventSource("/api/events");
    for (const type of ["run_added", "run_updated", "run_removed", "problem"])
      es.addEventListener(type, scheduleRefresh);
    es.onerror = () => { es.close(); setTimeout(connectSSE, 10_000); };
  } catch (e) { /* SSE unsupported: polling via manual reload only */ }
}

/* ---------- runs view ---------- */
function renderRuns() {
  const rows = state.runs.map((r) => {
    const arch = r.architecture;
    const h = r.headline;
    const firstVar = h ? Object.keys(h.variables)[0] : null;
    const evals = r.evaluations.map((e) =>
      `${e.eval_id}${e.is_primary ? "*" : ""}${e.is_valid ? "" : " (invalid)"}`).join(", ") || "—";
    return `<tr class="clickable" data-run="${r.run_id}">
      <td><div class="runcell"><span class="dot" style="background:${runColor(r.run_id)}"></span>
        <span>${r.name}<br><span class="sub">${(r.tags || []).join(" · ")}</span></span></div></td>
      <td>${r.stale
        ? '<span class="badge stalled" title="no DONE marker and no recent checkpoint activity — training crashed or was abandoned">stalled — no activity</span>'
        : `<span class="badge ${r.status}">${STATUS_LABEL[r.status] || r.status}</span>`}</td>
      <td>${fmt(arch.resolution_mode)}</td>
      <td class="num">${fmt(arch.hidden_dim)}</td>
      <td class="num">${fmt(arch.params_millions)}</td>
      <td class="num">${h ? `${h.lead_first}–${h.lead_last}` : "—"}</td>
      <td class="num">${h && firstVar ? fmt(h.variables[firstVar].rmse_last) : "—"}</td>
      <td class="num">${h && firstVar ? fmt(h.variables[firstVar].acc_last, 3) : "—"}</td>
      <td><span class="sub">${evals}</span></td>
      <td class="num">${r.problem_count.error}E/${r.problem_count.warning}W</td>
    </tr>`;
  }).join("");
  const firstVar = state.meta && state.meta.variables_union[0] || "—";
  $("runsTable").innerHTML = `<thead><tr>
    <th>Run</th><th>Status</th><th>Res</th><th>Hidden</th><th>Params (M)</th>
    <th>Leads</th><th>${firstVar} RMSE @last</th><th>${firstVar} ACC @last</th>
    <th>Evaluations</th><th>Problems</th></tr></thead><tbody>${rows}</tbody>`;
  $("runsTable").querySelectorAll("tr.clickable").forEach((tr) =>
    tr.addEventListener("click", () => {
      if (state.detailRunId !== tr.dataset.run) {
        state.detailEval = null; state.detailVar = null;
        state.mapMode = null; state.mapVar = null; state.mapDay = null;
      }
      state.detailRunId = tr.dataset.run;
      setView("detail");
    }));
}

/* ---------- detail view ---------- */
async function renderDetail() {
  const body = $("detailBody");
  if (!state.detailRunId) { body.innerHTML = '<p class="empty">No run selected.</p>'; return; }
  let record;
  try { record = await getJSON(`/api/runs/${encodeURIComponent(state.detailRunId)}`); }
  catch (e) { body.innerHTML = `<p class="empty">${e.message}</p>`; return; }

  const problemsHtml = record.problems.length ? `
    <div class="card mt"><h3>Problems for this run (verbatim)</h3>
      ${record.problems.map(problemHtml).join("")}</div>` : "";

  const archHtml = architectureHtml(record.architecture);
  const validEvals = record.evaluations.filter((e) => e.is_valid);

  if (!validEvals.length) {
    body.innerHTML = `
      <h2>${record.run_id}
        <span class="badge ${record.status}">${STATUS_LABEL[record.status]}</span></h2>
      <div class="grid3 mt">
        <div class="card"><h3>Status</h3>
          <p class="hint">${record.status === "invalid"
            ? "This run failed contract validation — see the problems below."
            : "No valid evaluation yet. Curves appear when the evaluation stage writes its metrics."}</p></div>
        ${archHtml}
      </div>${problemsHtml}`;
    return;
  }

  const evalSel = state.detailEval && validEvals.some((e) => e.eval_id === state.detailEval)
    ? state.detailEval
    : (validEvals.find((e) => e.is_primary) || validEvals[0]).eval_id;
  const ev = validEvals.find((e) => e.eval_id === evalSel) || validEvals[0];
  const varSel = state.detailVar && ev.variables.includes(state.detailVar)
    ? state.detailVar : ev.variables[0];

  body.innerHTML = `
    <h2>${record.run_id}
      <span class="badge evaluated">evaluated</span></h2>
    <div class="controls mt">
      ${validEvals.length > 1 ? `<span><label>Evaluation</label>
        <select id="detEval">${validEvals.map((e) =>
          `<option ${e.eval_id === ev.eval_id ? "selected" : ""}>${e.eval_id}</option>`).join("")}</select></span>` : ""}
      <span><label>Variable</label><select id="detVar">${ev.variables.map((v) =>
        `<option ${v === varSel ? "selected" : ""}>${v}</option>`).join("")}</select></span>
      <span class="hint">horizon ${ev.horizon} · leads ${ev.lead_times[0]}–${ev.lead_times[ev.lead_times.length - 1]}
        · checkpoint epoch ${fmt(ev.checkpoint.epoch)} · unit ${ev.units[varSel] || "?"} (declared)</span>
    </div>
    <div class="grid2">
      <div class="card"><h3>RMSE vs lead time</h3><div class="chart" id="detRmse"></div></div>
      <div class="card"><h3>ACC vs lead time</h3><div class="chart" id="detAcc"></div></div>
    </div>
    <div class="grid3 mt">
      <div>
        <div class="grid2">
          <div class="card"><h3>Rollout curve (valid loss / step)</h3><div class="chart" id="detRollout"></div></div>
          <div class="card"><h3>Activity ratio (variance)</h3><div class="chart" id="detVar"></div></div>
        </div>
        <div class="card mt"><h3>Power spectrum / spectral RMSE</h3><div class="chart tall" id="detSpec"></div></div>
        <div class="card mt"><h3>Attention metrics (per layer)</h3><div class="tablewrap" id="detAttn"></div></div>
      </div>
      ${archHtml}
    </div>
    <div class="card mt"><h3>Qualitative maps</h3><div id="detMaps"></div></div>
    ${problemsHtml}`;

  const detEval = $("detEval");
  if (detEval) detEval.onchange = () => { state.detailEval = detEval.value; renderDetail(); };
  $("detVar").onchange = () => { state.detailVar = $("detVar").value; renderDetail(); };
  await renderDetailCharts(record, ev, varSel);
}

async function renderDetailCharts(record, ev, variable) {
  for (const metric of ["rmse", "acc"]) {
    const payload = await getJSON(`/api/overlay?runs=${encodeURIComponent(record.run_id)}` +
      `&variable=${encodeURIComponent(variable)}&metric=${metric}` +
      `&eval=${encodeURIComponent(ev.eval_id)}&include=persistence`);
    Plotly.react(metric === "rmse" ? "detRmse" : "detAcc", overlayTraces(payload),
      baseLayout("Lead time (days)", metric === "rmse"
        ? `RMSE (${ev.units[variable] || "?"})` : "ACC"), PLOT_CONF);
  }
  const diag = record.diagnostics;
  const placeholder = (id, msg) => { $(id).innerHTML = `<p class="empty">${msg}</p>`; };
  if (diag && diag.rollout_curve) {
    Plotly.react("detRollout",
      [{ x: diag.rollout_curve.steps, y: diag.rollout_curve.losses, mode: "lines+markers",
         name: "valid loss", line: { color: runColor(record.run_id), width: 2 }, marker: { size: 5 },
         hovertemplate: "loss: %{y:.4f}<extra></extra>" }],
      baseLayout("Rollout step", "Validation loss"), PLOT_CONF);
  } else placeholder("detRollout", "Not recorded for this run.");
  if (diag && diag.variance_ratio) {
    const key = diag.variance_ratio[variable] ? variable : Object.keys(diag.variance_ratio)[0];
    const ys = diag.variance_ratio[key];
    Plotly.react("detVar",
      [{ x: ys.map((_, i) => i + 1), y: ys, mode: "lines+markers", name: key,
         line: { color: runColor(record.run_id), width: 2 }, marker: { size: 5 },
         hovertemplate: `${key}: %{y:.3f}<extra></extra>` }],
      baseLayout("Lead time (days)", "Forecast / truth activity"), PLOT_CONF);
  } else placeholder("detVar", "Not recorded for this run.");
  if (diag && diag.power_spectrum) {
    const key = diag.power_spectrum[variable] ? variable : Object.keys(diag.power_spectrum)[0];
    const sp = diag.power_spectrum[key];
    const p = pal();
    const seq = theme() === "dark" ? ["#3987e5", "#9ec5f4", "#cde2fb"] : ["#86b6ef", "#2a78d6", "#104281"];
    const leads = Object.keys(sp.pred_by_lead);
    const shown = leads.length > 3 ? [leads[0], leads[Math.floor(leads.length / 2)], leads[leads.length - 1]] : leads;
    Plotly.react("detSpec",
      [{ x: sp.wavenumbers, y: sp.truth, name: `truth · ${key}`, mode: "lines",
         line: { color: p.ink2, width: 2.2 }, hovertemplate: "truth: %{y:.3g}<extra></extra>" },
       ...shown.map((lead, i) => ({ x: sp.wavenumbers, y: sp.pred_by_lead[lead],
         name: `model · day ${lead}`, mode: "lines", line: { color: seq[i % seq.length], width: 2 },
         hovertemplate: `day ${lead}: %{y:.3g}<extra></extra>` }))],
      baseLayout("Wavenumber", "Power", {
        xaxis: Object.assign(baseLayout().xaxis, { type: "log", dtick: null, title: { text: `Wavenumber (${key})` } }),
        yaxis: Object.assign(baseLayout().yaxis, { type: "log", title: { text: "Power" } }),
      }), PLOT_CONF);
  } else if (diag && diag.spectral_rmse) {
    const key = diag.spectral_rmse[variable] ? variable : Object.keys(diag.spectral_rmse)[0];
    const sr = diag.spectral_rmse[key];
    Plotly.react("detSpec",
      [{ x: sr.wavenumbers, y: sr.rmse, name: `spectral RMSE · ${key}`, mode: "lines+markers",
         line: { color: runColor(record.run_id), width: 2 }, marker: { size: 4 },
         hovertemplate: `${key}: %{y:.6g}<extra></extra>` }],
      baseLayout("Wavenumber bin", `Spectral RMSE (${key}, H=${sr.horizon})`, {
        xaxis: Object.assign(baseLayout().xaxis, { dtick: null, title: { text: "Wavenumber bin" } }),
      }), PLOT_CONF);
  } else placeholder("detSpec", "Not recorded for this run.");

  const attn = $("detAttn");
  if (attn) {
    if (diag && diag.attention_table && diag.attention_table.length) {
      const cols = Object.keys(diag.attention_table[0]);
      attn.innerHTML = `<table><thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
        diag.attention_table.map((row) => `<tr>${cols.map((c) =>
          `<td class="num">${fmt(row[c])}</td>`).join("")}</tr>`).join("") + "</tbody></table>";
    } else attn.innerHTML = '<p class="empty">Not recorded for this run.</p>';
  }

  renderMapsSection(record);
}

/* ---------- qualitative maps section (per-variable / per-day modes) ---------- */
function renderMapsSection(record) {
  const qual = record.qualitative;
  const el = $("detMaps");
  const stamp = encodeURIComponent((qual && qual.generated_at) || record.last_scanned_at || "");
  const art = (rel) => `/api/artifacts/${encodeURIComponent(record.run_id)}/${rel}?v=${stamp}`;

  const hasByDay = qual && qual.by_day && Object.keys(qual.by_day).length;
  if (hasByDay) {
    const vars = Object.keys(qual.by_day).sort();
    const days = (qual.days && qual.days.length)
      ? qual.days : [...new Set(vars.flatMap((v) => Object.keys(qual.by_day[v]).map(Number)))].sort((a, b) => a - b);
    if (!state.mapMode) state.mapMode = "variable";
    if (!state.mapVar || !vars.includes(state.mapVar)) state.mapVar = vars[0];
    if (!state.mapDay || !days.includes(state.mapDay)) state.mapDay = days[0];

    let body;
    if (state.mapMode === "variable") {
      const files = qual.by_day[state.mapVar] || {};
      body = `<div class="maps-stack">${days.filter((d) => files[String(d)]).map((d) => `
        <figure><img loading="lazy" src="${art(files[String(d)])}"
          alt="${state.mapVar} day ${d}">
          <figcaption>${state.mapVar} · day ${d} · ground truth / prediction / bias</figcaption>
        </figure>`).join("")}</div>`;
    } else {
      body = `<div class="maps-stack">${vars.filter((v) => (qual.by_day[v] || {})[String(state.mapDay)]).map((v) => `
        <figure><img loading="lazy" src="${art(qual.by_day[v][String(state.mapDay)])}"
          alt="${v} day ${state.mapDay}">
          <figcaption>${v} · day ${state.mapDay} · ground truth / prediction / bias</figcaption>
        </figure>`).join("")}</div>`;
    }
    el.innerHTML = `
      <div class="controls">
        <span><label>View</label><select id="mapMode">
          <option value="variable" ${state.mapMode === "variable" ? "selected" : ""}>one variable, all days</option>
          <option value="day" ${state.mapMode === "day" ? "selected" : ""}>one day, all variables</option>
        </select></span>
        ${state.mapMode === "variable"
          ? `<span><label>Variable</label><select id="mapVar">${vars.map((v) =>
              `<option ${v === state.mapVar ? "selected" : ""}>${v}</option>`).join("")}</select></span>`
          : `<span><label>Day</label><select id="mapDay">${days.map((d) =>
              `<option value="${d}" ${d === state.mapDay ? "selected" : ""}>day ${d}</option>`).join("")}</select></span>`}
        <span class="hint">year-mean over the test split · generated offline from saved arrays</span>
      </div>${body}`;
    $("mapMode").onchange = () => { state.mapMode = $("mapMode").value; renderMapsSection(record); };
    const mv = $("mapVar"); if (mv) mv.onchange = () => { state.mapVar = mv.value; renderMapsSection(record); };
    const md = $("mapDay"); if (md) md.onchange = () => { state.mapDay = parseInt(md.value, 10); renderMapsSection(record); };
    return;
  }

  el.innerHTML = qual && Object.keys(qual.maps).length
    ? `<div class="maps">${Object.entries(qual.maps).map(([v, rel]) => `
        <figure><img loading="lazy" src="${art(rel)}" alt="${v} maps">
          <figcaption>${v} · leads ${(qual.metadata.lead_times || []).join(", ")} ·
            ${qual.metadata.aggregate_mode || ""}</figcaption></figure>`).join("")}</div>
       <p class="hint mt">Per-day maps not generated yet — run
         <code>python -m dashboard.generate_maps --run ${record.run_id} --gpu &lt;idx&gt;</code>.</p>`
    : `<p class="empty">No qualitative maps for this run yet. Generate them with
       <code>python -m dashboard.generate_maps --run ${record.run_id} --gpu &lt;idx&gt;</code>.</p>`;
}

/* ---------- architecture panel ---------- */
function architectureHtml(a) {
  const kv = (pairs) => `<dl class="kv">${pairs.filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([k, v]) => `<dt>${k}</dt><dd class="num">${v}</dd>`).join("")}</dl>`;
  const blocks = a.blocks || {};
  return `<div class="card"><h3>Architecture</h3>
    ${kv([
      ["resolution", a.resolution_mode],
      ["grid", (a.grid_shape || []).join(" × ")],
      ["levels", (a.level_shapes || []).map((s) => s.join("×")).join(" → ")],
      ["nodes / level", (a.node_counts || []).join(", ")],
      ["hidden dim", a.hidden_dim],
      ["heads", a.num_heads],
      ["k neighbors", (a.level_k_neighbors || []).join("/")],
      ["connectivity", a.graph_connectivity_strategy],
      ["blocks", Object.entries(blocks).filter(([, v]) => v != null).map(([k, v]) => `${k}:${v}`).join(" ")],
      ["params (M)", a.params_millions],
      ["LR", a.lr_schedule && a.lr_schedule.lr != null
        ? `${a.lr_schedule.type} ${a.lr_schedule.lr} → ${a.lr_schedule.min_lr}` : null],
      ["rollout", a.rollout && a.rollout.schedule
        ? `S${a.rollout.schedule[0]}…S${a.rollout.schedule[a.rollout.schedule.length - 1]}` +
          ` · ${a.rollout.max_epochs} epochs` : null],
      ["forcings", (a.forcings && a.forcings.known_future_variables || []).join(", ") || "none (predicted)"],
      ["init", a.init_from_checkpoint ? a.init_from_checkpoint.split("/").slice(-2)[0] : "from scratch"],
      ["tags", (a.tags || []).join(", ")],
    ])}</div>`;
}

/* ---------- problems ---------- */
function problemHtml(p) {
  return `<div class="problem ${p.severity}">
    <b>${p.reason}</b> · ${p.severity} · <code>${p.artifact}</code>
    ${p.eval_id ? ` · ${p.eval_id}` : ""}<br>
    expected: ${p.expected}<br>found: ${p.found}
    <div class="meta">${p.run_id} · ${p.path} · ${p.detected_at} · ${p.schema_version}</div>
  </div>`;
}
async function renderProblems() {
  const payload = await getJSON("/api/problems");
  $("problemsList").innerHTML = payload.problems.length
    ? payload.problems.map(problemHtml).join("")
    : '<p class="empty">No validation problems. Every run on disk matches the contract.</p>';
}

/* ---------- ideas view ---------- */
const escIdea = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;")
  .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
function ideaBlockHtml(b) {
  if (b.type === "p") return `<p class="idea-p">${escIdea(b.text)}</p>`;
  if (b.type === "list")
    return `<ul class="idea-list">${(b.items || []).map((it) => `<li>${escIdea(it)}</li>`).join("")}</ul>`;
  if (b.type === "table")
    return `<div class="tablewrap"><table>
      <thead><tr>${(b.columns || []).map((c) => `<th>${escIdea(c)}</th>`).join("")}</tr></thead>
      <tbody>${(b.rows || []).map((r) => `<tr>${r.map((c) => `<td>${escIdea(c)}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>`;
  return "";
}
async function renderIdeas() {
  let payload;
  try { payload = await getJSON("/api/ideas"); }
  catch (err) { $("ideasBody").innerHTML = `<div class="card"><p class="empty">${escIdea(err.message)}</p></div>`; return; }
  const ideas = payload.ideas || [];
  $("ideasBody").innerHTML = ideas.length ? ideas.map((idea) => `
    <div class="card mt idea-card">
      <h3>${escIdea(idea.title)}
        <span class="badge">${escIdea(idea.status || "idea")}</span>
        <span class="hint" style="text-transform:none;font-weight:400">· ${escIdea(idea.date || "")}</span></h3>
      ${idea.verdict ? `<p class="idea-verdict">${escIdea(idea.verdict)}</p>` : ""}
      ${(idea.sections || []).map((s) => `
        <h4 class="idea-h4">${escIdea(s.heading)}</h4>
        ${(s.blocks || []).map(ideaBlockHtml).join("")}`).join("")}
      ${(idea.references || []).length ? `
        <h4 class="idea-h4">References</h4>
        <ol class="idea-refs">${idea.references.map((r) =>
          `<li><a href="${escIdea(r.url)}" target="_blank" rel="noopener">${escIdea(r.label)}</a></li>`).join("")}</ol>` : ""}
      ${(idea.evidence_sources || []).length ? `
        <h4 class="idea-h4">Evidence sources (this repo)</h4>
        <ul class="idea-list hint">${idea.evidence_sources.map((s) => `<li><code>${escIdea(s)}</code></li>`).join("")}</ul>` : ""}
    </div>`).join("")
    : '<div class="card"><p class="empty">No ideas recorded yet — add entries to dashboard/ideas.json.</p></div>';
}

/* ---------- compare view ---------- */
function cmpLeadOptions() {
  const horizons = state.runs
    .filter((r) => state.cmp.selected.has(r.run_id) && r.headline)
    .map((r) => r.headline.lead_last);
  const maxH = horizons.length ? Math.max(...horizons) : 0;
  const opts = [];
  for (let k = 1; k <= maxH; k++) opts.push({ id: `day:${k}`, label: `day ${k}` });
  opts.push({ id: "mean", label: "mean over leads (derived)" });
  opts.push({ id: "crossing", label: "ACC < 0.6 crossing (derived)" });
  return opts;
}
function renderCmpControls() {
  const evaluated = state.runs.filter((r) => r.status === "evaluated");
  $("cmpChips").innerHTML = evaluated.map((r) => {
    const on = state.cmp.selected.has(r.run_id);
    return `<span class="chip ${on ? "on" : "off"}" data-id="${r.run_id}"
      style="--chipc:${runColor(r.run_id)}">
      <span class="dot" style="background:${runColor(r.run_id)}"></span>${r.name}</span>`;
  }).join("") || '<p class="empty">No evaluated runs yet.</p>';
  $("cmpChips").querySelectorAll(".chip").forEach((ch) => ch.addEventListener("click", () => {
    const id = ch.dataset.id;
    if (state.cmp.selected.has(id)) { if (state.cmp.selected.size > 1) state.cmp.selected.delete(id); }
    else state.cmp.selected.add(id);
    renderCompare();
  }));

  const evalIds = new Set(["primary"]);
  evaluated.filter((r) => state.cmp.selected.has(r.run_id))
    .forEach((r) => r.evaluations.filter((e) => e.is_valid).forEach((e) => evalIds.add(e.eval_id)));
  $("cmpEval").innerHTML = [...evalIds].map((id) =>
    `<option ${id === state.cmp.eval ? "selected" : ""}>${id}</option>`).join("");
  $("cmpEval").onchange = () => { state.cmp.eval = $("cmpEval").value; renderCompare(); };

  $("cmpVar").innerHTML = (state.meta.variables_union || []).map((v) =>
    `<option ${v === state.cmp.variable ? "selected" : ""}>${v}</option>`).join("");
  $("cmpVar").onchange = () => { state.cmp.variable = $("cmpVar").value; renderCompare(); };
  $("cmpMetric").value = state.cmp.metric;
  $("cmpMetric").onchange = () => { state.cmp.metric = $("cmpMetric").value; renderCompare(); };

  const opts = cmpLeadOptions();
  if (!state.cmp.lead || !opts.some((o) => o.id === state.cmp.lead))
    state.cmp.lead = opts.length > 2 ? opts[opts.length - 3].id : (opts[0] || {}).id;
  $("cmpLead").innerHTML = opts.map((o) =>
    `<option value="${o.id}" ${o.id === state.cmp.lead ? "selected" : ""}>${o.label}</option>`).join("");
  $("cmpLead").onchange = () => { state.cmp.lead = $("cmpLead").value; renderCompare(); };

  $("cmpPersistence").onchange = () => { state.cmp.persistence = $("cmpPersistence").checked; renderCompare(); };
  $("cmpExternals").innerHTML = (state.meta.external_baselines || []).map((x) => `
    <label><input type="checkbox" data-ext="${x.id}"
      ${state.cmp.externals.has(x.id) ? "checked" : ""}> ${x.label}</label>`).join(" ");
  $("cmpExternals").querySelectorAll("input").forEach((cb) => cb.onchange = () => {
    if (cb.checked) state.cmp.externals.add(cb.dataset.ext);
    else state.cmp.externals.delete(cb.dataset.ext);
    renderCompare();
  });
}
function leadCriterion() {
  if (state.cmp.lead === "mean") return { type: "mean_leads" };
  if (state.cmp.lead === "crossing") return { type: "acc_crossing", threshold: 0.6 };
  return { type: "day", k: parseInt(String(state.cmp.lead).split(":")[1], 10) };
}
async function renderCompare() {
  renderCmpControls();
  const runsCsv = [...state.cmp.selected].join(",");
  if (!runsCsv) return;
  const include = [];
  if (state.cmp.persistence) include.push("persistence");
  state.cmp.externals.forEach((id) => include.push(`external:${id}`));
  const q = (metric) => `/api/overlay?runs=${encodeURIComponent(runsCsv)}` +
    `&variable=${encodeURIComponent(state.cmp.variable)}&metric=${metric}` +
    `&eval=${encodeURIComponent(state.cmp.eval)}&include=${encodeURIComponent(include.join(","))}`;
  const [ovR, ovA, ranking] = await Promise.all([
    getJSON(q("rmse")), getJSON(q("acc")),
    postJSON("/api/ranking", {
      runs: [...state.cmp.selected], variable: state.cmp.variable,
      metric: state.cmp.metric, lead: leadCriterion(), eval: state.cmp.eval,
      include_external: state.cmp.externals.size > 0,
    }),
  ]);
  const winnerRow = ranking.rows.find((r) => r.rank === 1);
  const winnerId = winnerRow ? winnerRow.run_id : null;

  /* fixed-size charts: the colored chips above are the legend, so adding runs
     never shrinks the plot area */
  const bigLayout = (yTitle) => Object.assign(
    baseLayout("Lead time (days)", yTitle),
    { showlegend: false, margin: { l: 66, r: 16, t: 8, b: 52 } });
  Plotly.react("cmpRmse", overlayTraces(ovR, winnerId),
    bigLayout(`RMSE (${(state.meta.units.table || {})[state.cmp.variable] || "?"})`), PLOT_CONF);
  Plotly.react("cmpAcc", overlayTraces(ovA, winnerId),
    bigLayout("ACC"), PLOT_CONF);

  const allWarnings = [...ovR.warnings, ...ranking.warnings];
  const seen = new Set();
  $("cmpWarnings").innerHTML = allWarnings.filter((w) => {
    const key = w.type + (w.detail || "");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  }).map((w) => `<div class="banner"><b>${w.type}</b> — ${w.detail || ""}</div>`).join("");

  $("rankCriterion").textContent =
    `— ${state.cmp.variable} · ${ranking.criterion.metric} · ${JSON.stringify(ranking.criterion.lead)}`;
  $("rankTable").innerHTML = `<thead><tr><th>#</th><th>Run</th><th>Value</th>
    <th>CI</th><th>Δ vs best</th><th>Flags</th></tr></thead><tbody>` +
    ranking.rows.map((r) => `<tr class="${r.rank === 1 ? "winner" : ""}">
      <td class="num">${r.rank === 1 ? "🏆 1" : (r.rank ?? "—")}</td>
      <td><div class="runcell"><span class="dot" style="background:${runColor(r.run_id)}"></span>
        ${r.label || r.run_id}</div></td>
      <td class="num ${r.rank === 1 ? "best" : ""}">${fmt(r.value)}</td>
      <td class="num">${r.ci ? `[${fmt(r.ci[0])}, ${fmt(r.ci[1])}]` : "—"}</td>
      <td class="num">${r.delta_vs_best_pct === null ? "—" : fmt(r.delta_vs_best_pct, 3) + "%"}</td>
      <td>${r.flags.map((f) => `<span class="badge ${f === "external" ? "external" : "flag"}">${f}</span>`).join(" ")}</td>
    </tr>`).join("") + "</tbody>";

  const bestLead = leadCriterion().type === "day" ? leadCriterion().k
    : (ovR.lead_times[ovR.lead_times.length - 1] || 1);
  const extTokens = [...state.cmp.externals].map((id) => `external:${id}`).join(",");
  const best = await getJSON(`/api/best?runs=${encodeURIComponent(runsCsv)}` +
    `&lead=${bestLead}&metric=${state.cmp.metric}&eval=${encodeURIComponent(state.cmp.eval)}` +
    `&include=${encodeURIComponent(extTokens)}`);
  $("bestTable").innerHTML = `<thead><tr><th>Variable</th><th>Best run</th><th>Value @ day ${best.lead}</th>
    <th>CI</th><th>Within CI of runner-up</th></tr></thead><tbody>` +
    Object.entries(best.variables).map(([v, b]) => `<tr>
      <td>${v}</td>
      <td><div class="runcell"><span class="dot" style="background:${runColor(b.run_id)}"></span>${b.run_id}
        ${b.external ? ' <span class="badge external">external</span>' : ""}</div></td>
      <td class="num best">${fmt(b.value)}</td>
      <td class="num">${b.ci ? `[${fmt(b.ci[0])}, ${fmt(b.ci[1])}]` : "—"}</td>
      <td>${b.within_ci_of_runner_up === null
            ? (b.external ? '<span class="badge flag">no CI for external</span>' : "—")
            : b.within_ci_of_runner_up ? '<span class="badge flag">yes — statistically close</span>'
            : '<span class="badge evaluated">no — clear win</span>'}</td>
    </tr>`).join("") + "</tbody>";
}

/* ---------- summary tree (editable lineage) ---------- */
async function treeCall(method, url, body) {
  const resp = await fetch(url, {
    method, headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const payload = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(payload.detail || `${method} ${url} -> ${resp.status}`);
  return payload;
}

async function renderSummary() {
  const payload = await getJSON("/api/tree");
  state.tree.nodes = payload.nodes;
  if (!state.tree.selected || !state.tree.nodes.some((n) => n.id === state.tree.selected))
    state.tree.selected = (state.tree.nodes.find((n) => n.parent_id === null) ||
                           state.tree.nodes[0] || {}).id;
  drawTree();
  renderNodePanel();
}

function drawTree() {
  const svg = d3.select("#treeSvg");
  svg.selectAll("*").remove();
  if (!state.tree.nodes.length) return;

  /* FOREST support: hang every real root off a hidden virtual root, lay the
     whole forest out at once, then never render the virtual node/links. */
  const VIRTUAL = "__forest_root__";
  const data = [{ id: VIRTUAL, parent_id: null }].concat(
    state.tree.nodes.map((n) => ({ ...n, parent_id: n.parent_id ?? VIRTUAL })));
  const root = d3.stratify().id((d) => d.id).parentId((d) => d.parent_id)(data);
  const NW = 196, NH = 58;
  d3.tree().nodeSize([NH + 26, NW + 92])(root);

  const real = root.descendants().filter((d) => d.depth > 0);
  let x0 = Infinity, x1 = -Infinity, yMin = Infinity, y1 = -Infinity;
  real.forEach((d) => {
    x0 = Math.min(x0, d.x); x1 = Math.max(x1, d.x);
    yMin = Math.min(yMin, d.y); y1 = Math.max(y1, d.y);
  });
  const pad = 24;
  const W = (y1 - yMin) + NW + pad * 2, H = (x1 - x0) + NH + pad * 2;
  svg.attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
  const g = svg.append("g")
    .attr("transform", `translate(${pad - yMin},${pad - x0 + NH / 2})`);

  const winEdge = (d) => d.source.data.winning && d.target.data.winning;
  g.selectAll("path.link")
    .data(root.links().filter((l) => l.source.depth > 0)).join("path")
    .attr("class", (d) => "link" + (winEdge(d) ? " winning" : ""))
    .attr("d", d3.linkHorizontal()
      .source((d) => [d.source.y + NW, d.source.x])
      .target((d) => [d.target.y, d.target.x]));

  const node = g.selectAll("g.node").data(real).join("g")
    .attr("class", (d) => {
      let cls = "node " + String(d.data.status || "").replace("-", "");
      if (d.data.winning) cls += " winning";
      if (d.data.id === state.tree.selected) cls += " selected";
      return cls;
    })
    .attr("transform", (d) => `translate(${d.y},${d.x - NH / 2})`)
    .on("click", (_, d) => {
      state.tree.selected = d.data.id;
      state.tree.mode = "view";
      drawTree();
      renderNodePanel();
    });

  node.append("rect").attr("width", NW).attr("height", NH);
  node.append("text").attr("x", 12).attr("y", 22)
    .text((d) => (d.data.title || "").slice(0, 26));
  node.append("text").attr("class", "sub").attr("x", 12).attr("y", 40)
    .text((d) => {
      const t = d.data.change || "";
      return t.length > 30 ? t.slice(0, 29) + "…" : t;
    });

  const badge = node.append("g").attr("class", "nbadge")
    .attr("transform", `translate(${NW - 10},-6)`);
  badge.append("circle").attr("r", 4.5)
    .attr("class", (d) => d.data.status === "in-progress" ? "pulse" : null)
    .attr("fill", (d) => d.data.status === "completed" ? "var(--good)"
      : d.data.status === "in-progress" ? "var(--warn)" : "var(--muted)");
  badge.append("text").attr("x", -9).attr("y", 3.5).attr("text-anchor", "end")
    .attr("fill", "var(--muted)")
    .text((d) => d.data.status === "suggested" ? "proposed"
      : d.data.status === "in-progress" ? "in progress" : "done");
}

function nodeForm(n, isNew) {
  const runOpts = ['<option value="">— none —</option>']
    .concat(state.runs.map((r) =>
      `<option value="${r.run_id}" ${n.run_id === r.run_id ? "selected" : ""}>${r.run_id}</option>`));
  const opt = (list, cur) => list.map((v) =>
    `<option ${v === cur ? "selected" : ""}>${v}</option>`).join("");
  return `
    <label class="f">Title</label><input type="text" id="nfTitle" value="${(n.title || "").replace(/"/g, "&quot;")}">
    <label class="f">Status</label><select id="nfStatus">${opt(["completed", "in-progress", "suggested"], n.status || "suggested")}</select>
    <label class="f">Verdict</label><select id="nfVerdict">${opt(["keep", "kill", "promising", "pending"], n.verdict || "pending")}</select>
    <label class="f">Linked run</label><select id="nfRun">${runOpts.join("")}</select>
    <label class="f"><input type="checkbox" id="nfWinning" ${n.winning ? "checked" : ""}> on the winning path</label>
    <label class="f">Config change</label><textarea id="nfChange">${n.change || ""}</textarea>
    <label class="f">Why / rationale</label><textarea id="nfRationale">${n.rationale || ""}</textarea>
    <label class="f">Expected</label><textarea id="nfExpected">${n.expected || ""}</textarea>
    <label class="f">Actual</label><textarea id="nfActual">${n.actual || (isNew ? "not yet run" : "")}</textarea>
    <label class="f">Notes</label><textarea id="nfNotes">${n.notes || ""}</textarea>
    <div class="btnrow">
      <button class="btn primary" id="nfSave">${isNew ? "Add node" : "Save"}</button>
      <button class="btn" id="nfCancel">Cancel</button>
    </div>`;
}

function readNodeForm() {
  return {
    title: $("nfTitle").value.trim(),
    status: $("nfStatus").value,
    verdict: $("nfVerdict").value,
    run_id: $("nfRun").value || null,
    winning: $("nfWinning").checked,
    change: $("nfChange").value,
    rationale: $("nfRationale").value,
    expected: $("nfExpected").value,
    actual: $("nfActual").value,
    notes: $("nfNotes").value,
  };
}

function renderNodePanel() {
  const el = $("nodePanel");
  const n = state.tree.nodes.find((x) => x.id === state.tree.selected);
  if (!n) {
    if (state.tree.mode === "add-root") {
      el.innerHTML = "<h2>New independent tree</h2>" + nodeForm({}, true);
      $("nfSave").onclick = async () => {
        try {
          const created = await treeCall("POST", "/api/tree/nodes",
            { ...readNodeForm(), parent_id: null });
          state.tree.selected = created.node.id;
          state.tree.mode = "view";
          renderSummary();
        } catch (e) { alert(e.message); }
      };
      $("nfCancel").onclick = () => { state.tree.mode = "view"; renderNodePanel(); };
      return;
    }
    el.innerHTML = `<p class="hint">${state.tree.nodes.length
      ? "Select a node in the tree."
      : "No nodes yet — start your first tree."}</p>
      <div class="btnrow"><button class="btn primary" id="npNewTreeEmpty">🌱 New tree</button></div>`;
    $("npNewTreeEmpty").onclick = () => { state.tree.mode = "add-root"; renderNodePanel(); };
    return;
  }

  if (state.tree.mode === "edit" || state.tree.mode === "add" ||
      state.tree.mode === "add-root") {
    const isNew = state.tree.mode !== "edit";
    const isRoot = state.tree.mode === "add-root";
    el.innerHTML = `<h2>${isRoot ? "New independent tree"
        : isNew ? `New child of “${n.title}”` : `Edit “${n.title}”`}</h2>` +
      nodeForm(isNew ? {} : n, isNew);
    $("nfSave").onclick = async () => {
      try {
        const fields = readNodeForm();
        if (isNew) {
          const created = await treeCall("POST", "/api/tree/nodes",
            { ...fields, parent_id: isRoot ? null : n.id });
          state.tree.selected = created.node.id;
        } else {
          await treeCall("PATCH", `/api/tree/nodes/${encodeURIComponent(n.id)}`, fields);
        }
        state.tree.mode = "view";
        renderSummary();
      } catch (e) { alert(e.message); }
    };
    $("nfCancel").onclick = () => { state.tree.mode = "view"; renderNodePanel(); };
    return;
  }

  const run = n.run_id ? state.runs.find((r) => r.run_id === n.run_id) : null;
  const statusLabel = { completed: "completed", "in-progress": "in progress",
                        suggested: "proposed — not yet run" }[n.status] || n.status;
  const verdictCls = { keep: "evaluated", kill: "invalid", promising: "in_progress",
                       pending: "trained" }[n.verdict] || "trained";
  el.innerHTML = `
    <h2>${n.title}</h2>
    <span class="badge ${verdictCls}">${n.verdict}</span>
    <span class="badge">${statusLabel}</span>
    ${n.winning ? '<span class="badge external">winning path</span>' : ""}
    ${run ? `<div class="field"><b>Linked run</b>
       <p><span class="badge ${run.status}">${STATUS_LABEL[run.status] || run.status}</span>
       <span class="runlink" id="npOpenRun">open ${run.run_id} →</span></p></div>` : ""}
    <div class="field"><b>Config change</b><p>${n.change || "—"}</p></div>
    <div class="field"><b>Why we tried it</b><p>${n.rationale || "—"}</p></div>
    <div class="field"><b>Expected</b><p>${n.expected || "—"}</p></div>
    <div class="field"><b>Actual</b><p>${n.actual || "—"}</p></div>
    <div class="field"><b>Notes</b><p class="notes">${n.notes || "(none — click Edit to add)"}</p></div>
    <div class="btnrow">
      <button class="btn" id="npEdit">✎ Edit</button>
      <button class="btn" id="npAdd">＋ Add child</button>
      <button class="btn" id="npNewTree">🌱 New tree</button>
      <button class="btn danger" id="npDelete">🗑 Delete</button>
    </div>
    <p class="hint" style="margin-top:8px">updated ${n.updated_at || "—"}</p>`;

  $("npEdit").onclick = () => { state.tree.mode = "edit"; renderNodePanel(); };
  $("npAdd").onclick = () => { state.tree.mode = "add"; renderNodePanel(); };
  $("npNewTree").onclick = () => { state.tree.mode = "add-root"; renderNodePanel(); };
  const del = $("npDelete");
  if (del) del.onclick = async () => {
    if (!del.dataset.armed) {
      del.dataset.armed = "1";
      del.textContent = n.parent_id === null
        ? "🗑 Really delete? (children become their own trees)"
        : "🗑 Really delete? (children re-attach)";
      setTimeout(() => { del.dataset.armed = ""; del.textContent = "🗑 Delete"; }, 4000);
      return;
    }
    try {
      await treeCall("DELETE", `/api/tree/nodes/${encodeURIComponent(n.id)}`);
      state.tree.selected = n.parent_id;
      renderSummary();
    } catch (e) { alert(e.message); }
  };
  const open = $("npOpenRun");
  if (open) open.onclick = () => {
    state.detailRunId = n.run_id;
    state.detailEval = null; state.detailVar = null;
    setView("detail");
  };
}

/* ---------- orchestration ---------- */
function render() {
  if (state.view === "runs") renderRuns();
  else if (state.view === "detail") renderDetail();
  else if (state.view === "compare") renderCompare();
  else if (state.view === "summary") renderSummary();
  else if (state.view === "ideas") renderIdeas();
  else if (state.view === "problems") renderProblems();
}
window.addEventListener("resize", () => {
  document.querySelectorAll("section.view.active .chart").forEach((el) => {
    if (el.data) Plotly.Plots.resize(el);
  });
});

(async function boot() {
  await refreshData();
  render();
  connectSSE();
})();
