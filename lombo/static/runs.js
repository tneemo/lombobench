/* ================= analysis ("Runs" view) ================= */
'use strict';

async function loadRequests() {
  const params = new URLSearchParams({ limit: '200' });
  const model = $('#f-model').value.trim();
  const experiment = $('#f-experiment').value.trim();
  const condition = $('#f-condition').value.trim();
  const origin = $('#f-origin').value;
  const status = $('#f-status').value;
  if (model) params.set('model', model);
  if (experiment) params.set('experiment', experiment);
  if (condition) params.set('condition', condition);
  if (origin) params.set('origin', origin);
  const data = await (await fetch(`${API}/requests?${params}`)).json();
  let rows = data.requests || [];
  if (status === 'ok') rows = rows.filter(r => !r.error);
  else if (status === 'error') rows = rows.filter(r => r.error);

  const latencyById = await fetchLatencyLookup(model, experiment);
  renderRequestRows(rows, latencyById);
  renderAnalysisCharts(rows, latencyById);
}

async function fetchLatencyLookup(model, experiment) {
  try {
    const params = new URLSearchParams({ limit: '500' });
    if (model) params.set('model', model);
    if (experiment) params.set('experiment', experiment);
    const data = await (await fetch(`${API}/metrics/latency?${params}`)).json();
    return new Map((data.points || []).map(p => [p.id, p]));
  } catch { return new Map(); }
}

function durationBarCell(point) {
  if (!point) return '<div class="dur-bar empty" title="no breakdown (failed run or non-Ollama engine)"></div>';
  const total = (point.load_ms || 0) + (point.prompt_eval_ms || 0) + (point.eval_ms || 0);
  if (!total) return '<div class="dur-bar empty"></div>';
  const pct = ms => `${((ms || 0) / total * 100).toFixed(1)}%`;
  const title = `load ${(point.load_ms || 0).toFixed(0)}ms · prompt eval ${(point.prompt_eval_ms || 0).toFixed(0)}ms · eval ${(point.eval_ms || 0).toFixed(0)}ms`;
  return `<div class="dur-bar" title="${esc(title)}">
    <i class="seg-load" style="width:${pct(point.load_ms)}"></i>
    <i class="seg-prompt" style="width:${pct(point.prompt_eval_ms)}"></i>
    <i class="seg-eval" style="width:${pct(point.eval_ms)}"></i></div>`;
}

function renderRequestRows(rows, latencyById) {
  const tbody = $('#req-rows');
  tbody.innerHTML = '';
  for (const r of rows) {
    const tr = document.createElement('tr');
    const diffKeys = Object.keys(r.params_diff || {}).join(' ');
    tr.innerHTML = `
      <td>${(r.started_at || '').slice(5, 19).replace('T', ' ')}</td>
      <td>${esc(r.model || '?')}</td>
      <td><span class="pill ${r.origin === 'lab' ? 'lab' : ''}">${r.origin}</span></td>
      <td>${esc(conditionText(r.condition || '—'))}</td>
      <td class="diffkeys">${esc(diffKeys || '—')}</td>
      <td class="num">${r.eval_count ?? ''}</td>
      <td class="num">${r.tokens_per_second ? r.tokens_per_second.toFixed(1) : ''}</td>
      <td class="num">${r.time_to_first_token_ms ? r.time_to_first_token_ms.toFixed(0) : ''}</td>
      <td>${durationBarCell(latencyById.get(r.id))}</td>
      <td>${r.error ? `<span class="pill err">error</span>` : `<span class="pill">${r.status_code ?? ''}</span>`}</td>`;
    tr.onclick = () => openDrawer(r.id);
    tbody.appendChild(tr);
  }
}

function renderAnalysisCharts(rows, latencyById) {
  const histEl = $('#an-hist-tps'), scatterEl = $('#an-scatter'), volEl = $('#an-volume');
  const tpsValues = rows.map(r => r.tokens_per_second).filter(v => v != null);
  if (tpsValues.length >= 2) {
    const { edges, counts } = histogramOf(tpsValues, Math.min(16, Math.max(4, Math.round(Math.sqrt(tpsValues.length)))));
    LomboCharts.histogram(histEl, { edges, counts, name: 'runs' }, { toolbox: false });
  } else {
    LomboCharts.dispose(histEl); histEl.innerHTML = emptyPanel('Not enough tok/s data in the current filter.');
  }

  // prompt_eval_count only comes from details/latency (not the /requests listing);
  // reuse the lookup already fetched for the duration column.
  const scatterPoints = rows.map(r => latencyById.get(r.id))
    .filter(p => p && p.ttft_ms != null && p.prompt_eval_count != null)
    .map(p => [p.prompt_eval_count, p.ttft_ms]);
  if (scatterPoints.length >= 2) {
    LomboCharts.scatter(scatterEl, { series: [{ name: 'runs', data: scatterPoints }] },
      { xName: 'prompt tokens', yName: 'TTFT ms', toolbox: false });
  } else {
    LomboCharts.dispose(scatterEl); scatterEl.innerHTML = emptyPanel('No TTFT/token pairs in the current filter.');
  }

  const byHour = new Map();
  for (const r of rows) {
    const bucket = (r.started_at || '').slice(0, 13);
    if (!bucket) continue;
    byHour.set(bucket, (byHour.get(bucket) || 0) + 1);
  }
  const hours = [...byHour.keys()].sort();
  if (hours.length >= 2) {
    LomboCharts.bars(volEl, { categories: hours.map(h => h.replace('T', ' ') + 'h'),
      series: [{ name: 'runs', data: hours.map(h => byHour.get(h)) }] }, { toolbox: false });
  } else {
    LomboCharts.dispose(volEl); volEl.innerHTML = emptyPanel('Hourly volume appears with more runs in the filter.');
  }
}

function histogramOf(values, bins) {
  const sorted = [...values].sort((a, b) => a - b);
  let lo = sorted[0], hi = sorted[sorted.length - 1];
  if (lo === hi) hi = lo + 1;
  const width = (hi - lo) / bins;
  const counts = new Array(bins).fill(0);
  for (const v of sorted) counts[Math.min(bins - 1, Math.floor((v - lo) / width))]++;
  const edges = Array.from({ length: bins + 1 }, (_, i) => lo + i * width);
  return { edges, counts };
}

$('#btn-filter').onclick = loadRequests;
$('#btn-toggle-charts').onclick = () => {
  const strip = $('#analysis-charts');
  const btn = $('#btn-toggle-charts');
  strip.hidden = !strip.hidden;
  btn.setAttribute('aria-expanded', String(!strip.hidden));
  btn.textContent = strip.hidden ? 'Show charts' : 'Hide charts';
};
