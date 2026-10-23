/* ================= Hugging Face → "Local" tab: model_baselines catalog
   (family/parameter_size/quantization), quantization comparison, reused drawer
   (closed through the same generic huggingface.js logic). ================= */
'use strict';

let localModelsLoaded = false;

function baseModelName(model) {
  const idx = model.lastIndexOf(':');
  return idx > 0 ? model.slice(0, idx) : model;
}

async function loadLocalModels() {
  if (localModelsLoaded) return;
  const wrap = $('#hf-local-results');
  wrap.innerHTML = '<div class="hf-list-head">Loading…</div>';
  try {
    const data = await (await fetch(`${API}/models`)).json();
    const models = data.models || [];
    localModelsLoaded = true;
    $('#hf-local-count').textContent = `${models.length} local model(s)`;
    renderQuantizationComparison(models);
    wrap.innerHTML = models.length
      ? models.map(localCard).join('')
      : '<div class="hf-empty"><strong>No local models yet.</strong><span>Run a session in Prompt lab to record the first baseline.</span></div>';
    wrap.querySelectorAll('[data-model]').forEach(card => {
      card.onclick = () => openLocalDrawer(card.dataset.model);
    });
  } catch (err) {
    wrap.innerHTML = `<div class="hf-empty"><strong>Could not load the local catalog.</strong><span>${esc(err.message)}</span></div>`;
  }
}

function localCard(m) {
  return `<button class="hf-card" data-model="${esc(m.model)}" type="button">
    <div class="hf-card-top"><span class="hf-resource-icon">L</span>
      <div class="hf-card-name"><strong title="${esc(m.model)}">${esc(m.model)}</strong><span>${esc(m.family || 'unknown family')}</span></div>
      ${m.quantization_level ? `<span class="hf-access public">${esc(m.quantization_level)}</span>` : ''}</div>
    <div class="hf-card-meta">
      <span>${esc(m.parameter_size || 'size —')}</span>
      <span>${m.n_runs} run${m.n_runs === 1 ? '' : 's'}</span>
      ${m.avg_tps != null ? `<span>${m.avg_tps.toFixed(1)} tok/s</span>` : ''}
      ${m.error_rate ? `<span class="warn">${(m.error_rate * 100).toFixed(0)}% error</span>` : ''}
    </div>
    <div class="hf-card-foot"><span>${m.last_used ? 'Last used ' + esc(timeAgo(m.last_used)) : 'no runs'}</span><span>View details →</span></div>
  </button>`;
}

function renderQuantizationComparison(models) {
  const groups = new Map();
  for (const m of models) {
    if (!m.quantization_level || !m.n_runs) continue;
    const base = baseModelName(m.model);
    if (!groups.has(base)) groups.set(base, []);
    groups.get(base).push(m);
  }
  const multi = [...groups.entries()].filter(([, list]) => list.length >= 2);
  let panel = $('#local-quant-compare');
  if (!multi.length) { panel?.remove(); return; }
  if (!panel) {
    panel = document.createElement('section');
    panel.className = 'viz-panel';
    panel.id = 'local-quant-compare';
    panel.innerHTML = `<header class="panel-head"><div><span class="eyebrow">Comparison</span><h2>Quantizations of the same base model</h2><p>average tok/s by quantization level</p></div></header>
      <div class="chart-wrap medium-chart" id="local-quant-chart"></div>`;
    $('#hf-local-content').prepend(panel);
  }
  const [base, list] = multi.sort((a, b) => b[1].length - a[1].length)[0];
  list.sort((a, b) => (a.quantization_level || '').localeCompare(b.quantization_level || ''));
  LomboCharts.bars($('#local-quant-chart'), {
    categories: list.map(m => m.quantization_level),
    series: [{ name: `${base} · tok/s`, data: list.map(m => m.avg_tps || 0) }],
  }, { yName: 'tok/s', toolbox: false });
}

function openLocalDrawer(model) {
  const drawer = $('#hf-drawer'), backdrop = $('#hf-drawer-backdrop');
  drawer.classList.add('open');
  drawer.setAttribute('aria-hidden', 'false');
  backdrop.hidden = false;
  $('#hf-detail-type').textContent = 'Local model';
  $('#hf-detail-name').textContent = model;
  $('#hf-detail-body').innerHTML = '<div class="hf-detail-loading"><div class="hf-skeleton"><i></i><i></i><i></i></div></div>';
  fetch(`${API}/models/detail?model=${encodeURIComponent(model)}`)
    .then(async resp => {
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      renderLocalDetail(data);
    })
    .catch(err => {
      $('#hf-detail-body').innerHTML = `<div class="hf-empty"><strong>Details unavailable.</strong><span>${esc(err.message)}</span></div>`;
    });
}

function renderLocalDetail(item) {
  const details = item.details || {};
  const body = `<div class="hf-detail-summary">
      <div class="hf-detail-stats">
        <span><b>${item.n_runs}</b> runs</span>
        <span><b>${item.avg_tps != null ? item.avg_tps.toFixed(1) : '—'}</b> average tok/s</span>
        <span><b>${item.avg_ttft_ms != null ? item.avg_ttft_ms.toFixed(0) : '—'}</b> ttft ms</span>
        <span><b>${item.error_rate != null ? (item.error_rate * 100).toFixed(1) + '%' : '—'}</b> error rate</span>
      </div>
      <dl class="hf-detail-rows">
        <div><dt>Family</dt><dd>${esc(details.family || '—')}</dd></div>
        <div><dt>Size</dt><dd>${esc(details.parameter_size || '—')}</dd></div>
        <div><dt>Quantization</dt><dd>${esc(details.quantization_level || '—')}</dd></div>
        <div><dt>Digest</dt><dd>${esc(item.digest || '—')}</dd></div>
        <div><dt>Captured at</dt><dd>${esc(item.captured_at || '—')}</dd></div>
      </dl>
      <div class="hf-actions">
        <button class="btn primary" id="local-open-bench" type="button">Test in Bench</button>
        <button class="btn" id="local-open-lab" type="button">Open in Lab</button>
      </div>
    </div>
    <section class="hf-detail-section"><h2>Modelfile (parameters)</h2>
      ${item.parameters_raw ? `<pre>${esc(item.parameters_raw)}</pre>` : '<p class="hf-unavailable">No parameters recorded.</p>'}</section>
    <section class="hf-detail-section"><h2>Modelfile system prompt</h2>
      ${item.system_text ? `<pre>${esc(item.system_text)}</pre>` : '<p class="hf-unavailable">Not defined.</p>'}</section>
    <section class="hf-detail-section"><h2>Recent runs</h2>
      ${(item.recent_requests || []).length ? `<div class="hf-files">${item.recent_requests.map(r =>
        `<div><code>${esc((r.started_at || '').slice(0, 16).replace('T', ' '))} · ${esc(conditionText(r.condition || 'base'))}</code>
          <span>${r.error ? 'error' : (r.tokens_per_second != null ? r.tokens_per_second.toFixed(1) + ' tok/s' : r.status_code)}</span></div>`).join('')}</div>`
        : '<p class="hf-unavailable">No recent runs.</p>'}</section>`;
  $('#hf-detail-body').innerHTML = body;
  $('#local-open-bench').onclick = () => { closeLocalDrawer(); window.selectHuggingFaceModel(item.model, 'benchmarks'); };
  $('#local-open-lab').onclick = () => { closeLocalDrawer(); window.selectHuggingFaceModel(item.model, 'lab'); };
}

function closeLocalDrawer() {
  $('#hf-drawer').classList.remove('open');
  $('#hf-drawer').setAttribute('aria-hidden', 'true');
  $('#hf-drawer-backdrop').hidden = true;
}

/* ================= tab switching (without modifying huggingface.js) =================
   The Local tab shows/hides #hf-local-content and reuses the existing tab
   structure; the Models/Datasets tabs get an additional listener
   (coexisting with huggingface.js) to close the Local tab when the
   user returns to the Hub. */
$('#hf-tab-local').addEventListener('click', () => {
  $('#hf-tab-models').classList.remove('active');
  $('#hf-tab-models').setAttribute('aria-selected', 'false');
  $('#hf-tab-datasets').classList.remove('active');
  $('#hf-tab-datasets').setAttribute('aria-selected', 'false');
  $('#hf-tab-local').classList.add('active');
  $('#hf-tab-local').setAttribute('aria-selected', 'true');
  $('#hf-model-filters').hidden = true;
  $('#hf-dataset-filters').hidden = true;
  $('#hf-content').hidden = true;
  $('#hf-local-content').hidden = false;
  loadLocalModels();
});
for (const id of ['#hf-tab-models', '#hf-tab-datasets']) {
  $(id).addEventListener('click', () => {
    $('#hf-tab-local').classList.remove('active');
    $('#hf-tab-local').setAttribute('aria-selected', 'false');
    $('#hf-local-content').hidden = true;
    $('#hf-content').hidden = false;
  });
}
