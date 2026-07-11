/* ================= Ativações: captura real via engine Transformers
   (POST /requests/{id}/activations + .../activations/inspect) ================= */
'use strict';

const activationsState = { requestId: null, matrix: null, metric: 'l2', compareRef: null, selected: null, lastInspect: null };

const ACT_MODULE_LABELS = { attn: 'attn', mlp: 'mlp', resid: 'resid' };
const ACT_MODULE_COLORS = { attn: 'var(--pane-b)', mlp: 'var(--pane-a)', resid: 'var(--pane-c)' };

window.loadActivationRequests = async function loadActivationRequests() {
  const sel = $('#act-request');
  if (sel.dataset.populated) return;
  try {
    const data = await (await fetch(`${API}/requests?limit=200`)).json();
    const reqs = data.requests || [];
    sel.innerHTML = reqs.length
      ? reqs.map(r => `<option value="${esc(r.id)}">${esc((r.started_at || '').slice(5, 16).replace('T', ' '))} · ${esc(r.model || '?')} · ${esc(r.condition || 'base')}</option>`).join('')
      : '<option value="">nenhuma execução ainda</option>';
    sel.dataset.populated = '1';
  } catch { sel.innerHTML = '<option value="">indisponível</option>'; }
};

$('#act-capture').onclick = () => captureActivations();
$('#act-metric-toggle').querySelectorAll('button').forEach(btn => {
  btn.addEventListener('click', () => {
    activationsState.metric = btn.dataset.metric;
    if (activationsState.matrix) renderActivationMatrix();
  });
});
$('#act-add-compare').onclick = () => {
  if (!activationsState.lastInspect) return;
  activationsState.compareRef = activationsState.lastInspect;
  const btn = $('#act-add-compare');
  btn.textContent = 'Referência definida ✓';
  setTimeout(() => { btn.textContent = 'Adicionar ao comparador'; }, 1500);
};

function setMatrixEmpty(message) {
  $('#activation-matrix').innerHTML = '';
  $('#activation-matrix-y').innerHTML = '';
  $('#activation-axis').innerHTML = '';
  $('#act-scale-legend').hidden = true;
  const empty = $('#act-empty');
  empty.hidden = false;
  empty.textContent = message;
}

async function captureActivations() {
  const requestId = $('#act-request').value;
  if (!requestId) { alert('Selecione uma execução.'); return; }
  activationsState.requestId = requestId;
  const position = Number($('#act-position').value || -1);
  const btn = $('#act-capture');
  btn.disabled = true;
  btn.textContent = 'Capturando…';
  try {
    const resp = await fetch(`${API}/requests/${requestId}/activations`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ position }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || JSON.stringify(data));
    activationsState.matrix = data;
    activationsState.compareRef = null;
    activationsState.selected = null;
    populatePositionSelect(data.tokens, position);
    renderActivationMatrix();
    $('#act-matrix-caption').textContent =
      `${data.model?.model || ''} · posição ${data.position} · ${data.num_layers} camadas · ${data.modules.join(', ')}`;
  } catch (err) {
    activationsState.matrix = null;
    setMatrixEmpty(`Erro: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Capturar';
  }
}

function populatePositionSelect(tokens, current) {
  const sel = $('#act-position');
  if (!tokens || !tokens.length) return;
  sel.innerHTML = tokens.map((tok, i) => {
    const idx = i - tokens.length; // convenção do backend: índice negativo relativo ao fim
    return `<option value="${idx}">${idx} · "${esc(tok)}"</option>`;
  }).join('');
  sel.value = String(current);
  sel.onchange = () => captureActivations();
}

function renderActivationMatrix() {
  const data = activationsState.matrix;
  if (!data) return;
  const matrixEl = $('#activation-matrix'), yEl = $('#activation-matrix-y'), axisEl = $('#activation-axis');
  const metric = activationsState.metric;
  matrixEl.innerHTML = ''; yEl.innerHTML = ''; axisEl.innerHTML = '';
  $('#act-empty').hidden = true;
  $('#act-scale-legend').hidden = false;

  const modules = data.modules;
  matrixEl.style.gridTemplateColumns = `repeat(${data.num_layers}, 1fr)`;
  matrixEl.style.gridTemplateRows = `repeat(${modules.length}, 1fr)`;
  yEl.style.gridTemplateRows = `repeat(${modules.length}, 1fr)`;
  yEl.innerHTML = modules.map(m => `<span>${esc(ACT_MODULE_LABELS[m] || m)}</span>`).join('');

  // cada módulo tem sua própria escala de cor (normalização por linha, não global)
  const byModule = {};
  for (const m of modules) {
    const values = data.layers.map(l => l[m]?.[metric] ?? 0);
    byModule[m] = { min: Math.min(...values), max: Math.max(...values) };
  }
  for (const m of modules) {
    const { min, max } = byModule[m];
    const span = max - min || 1;
    for (const layer of data.layers) {
      const value = layer[m]?.[metric] ?? 0;
      const cell = document.createElement('i');
      cell.style.setProperty('--v', String(Math.round(((value - min) / span) * 100)));
      cell.style.setProperty('--heat-color', ACT_MODULE_COLORS[m] || 'var(--pane-a)');
      cell.dataset.layer = layer.index;
      cell.dataset.module = m;
      cell.title = `L${layer.index} · ${m} · ${metric}=${value.toFixed(4)}`;
      cell.onclick = () => inspectActivationCell(layer.index, m, cell);
      if (activationsState.selected?.layer === layer.index && activationsState.selected?.module === m) {
        cell.classList.add('selected');
      }
      matrixEl.appendChild(cell);
    }
  }
  axisEl.style.gridTemplateColumns = `repeat(${data.num_layers}, 1fr)`;
  for (const layer of data.layers) {
    const label = document.createElement('span');
    label.textContent = `L${layer.index}`;
    axisEl.appendChild(label);
  }
}

async function inspectActivationCell(layerIndex, module, cellEl) {
  if (!activationsState.requestId) return;
  $('#activation-matrix').querySelectorAll('.selected').forEach(el => el.classList.remove('selected'));
  cellEl.classList.add('selected');
  activationsState.selected = { layer: layerIndex, module };
  $('#act-selection-title').textContent = `L${layerIndex} · ${module}`;
  const position = Number($('#act-position').value || -1);
  try {
    const resp = await fetch(`${API}/requests/${activationsState.requestId}/activations/inspect`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ layer_index: layerIndex, module, position, top_k_dims: 16, bins: 32 }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || JSON.stringify(data));
    renderActivationInspector(data);
  } catch (err) {
    $('#act-inspector-metrics').innerHTML = `<p class="notice-inline" style="color:var(--err)">Erro: ${esc(err.message)}</p>`;
  }
}

function renderActivationInspector(data) {
  const stats = data.stats;
  const ref = activationsState.compareRef;
  const deltaFor = key => {
    if (!ref || ref.stats[key] == null || stats[key] == null) return '';
    const base = Math.abs(ref.stats[key]) || 1;
    const pct = ((stats[key] - ref.stats[key]) / base) * 100;
    return `<em class="${pct >= 0 ? 'positive' : 'negative'}">${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%</em>`;
  };
  $('#act-inspector-metrics').innerHTML = `
    <span><small>NORMA L2</small><b>${stats.l2.toFixed(3)}</b>${deltaFor('l2')}</span>
    <span><small>MEAN</small><b>${stats.mean.toFixed(4)}</b>${deltaFor('mean')}</span>
    <span><small>SPARSITY</small><b>${(stats.sparsity * 100).toFixed(1)}%</b>${deltaFor('sparsity')}</span>
    <span><small>MAX</small><b>${stats.max.toFixed(3)}</b>${deltaFor('max')}</span>`;

  const outlier = data.top_dims[0];
  const badge = $('#act-outlier-badge');
  if (outlier && Math.abs(outlier.zscore) >= 2) {
    badge.hidden = false;
    badge.textContent = `${outlier.zscore >= 0 ? '+' : ''}${outlier.zscore.toFixed(1)}σ`;
  } else badge.hidden = true;

  $('#act-dim-count').textContent = `${data.histogram.counts.reduce((a, b) => a + b, 0)} dims`;
  LomboCharts.histogram($('#act-histogram'), {
    edges: data.histogram.edges, counts: data.histogram.counts, name: 'dims',
  }, { toolbox: false });

  $('#act-top-dims').innerHTML = `<table class="dense-table"><thead><tr>
      <th>DIM</th><th>VALOR</th><th>Z-SCORE</th><th>PERCENTIL</th></tr></thead><tbody>
    ${data.top_dims.map(d => `<tr><td><code>#${d.dim}</code></td><td>${d.value.toFixed(4)}</td>
      <td class="${Math.abs(d.zscore) >= 3 ? 'warn' : ''}">${d.zscore >= 0 ? '+' : ''}${d.zscore.toFixed(2)}σ</td>
      <td class="num">${d.percentile.toFixed(2)}</td></tr>`).join('')}
  </tbody></table>`;

  activationsState.lastInspect = data;
  $('#act-add-compare').disabled = false;
}
