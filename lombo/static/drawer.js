/* Request detail drawer: layered parameters, NNsight trace,
   real attention map, and the NNsight replay bundle (reuses the same drawer). */
'use strict';

async function openDrawer(id) {
  const r = await (await fetch(`${API}/requests/${id}`)).json();
  state.currentDrawerRequest = r;
  $('#btn-inspect').textContent = 'Run NNsight trace';
  $('#btn-inspect').onclick = () => runNNsightTrace(r);
  $('#btn-attention').onclick = () => runAttention(r);
  $('#d-model').textContent = r.model || '?';
  $('#d-id').textContent = r.id;
  const c = $('#drawer-content');
  const inferred = r.inferred_params || {};
  const requested = r.requested_params || {};
  const confirmed = r.confirmed_params || {};
  const diff = r.params_diff || {};
  const paramKeys = [...new Set([...Object.keys(requested), ...Object.keys(inferred),
                                 ...Object.keys(confirmed)])].sort();
  const rows = paramKeys.map(k => {
    const inf = inferred[k];
    return `<tr class="${diff[k] ? 'changed' : ''}">
      <td>${esc(k)}</td>
      <td>${k in requested ? esc(JSON.stringify(requested[k])) : '<span class="empty">—</span>'}</td>
      <td>${inf ? `${esc(JSON.stringify(inf.value))} <span class="src ${inf.source}">(${inf.source})</span>` : '—'}</td>
      <td>${confirmed[k] ? esc(JSON.stringify(confirmed[k].value)) : '<span class="empty">not confirmed</span>'}</td>
      <td>${diff[k] ? esc(JSON.stringify(diff[k].base_value)) + ` <span class="src">(${diff[k].base_source})</span>` : ''}</td>
    </tr>`;
  }).join('');

  const evals = (r.evaluations || []).map(e =>
    `<span class="metric">${esc(e.metric)} <b>${esc(typeof e.value === 'object'
      ? (e.value.label ? humanLabelText(e.value.label) : JSON.stringify(e.value)) : String(e.value))}</b>
      <span class="src">${e.source}</span></span>`).join(' ');

  c.innerHTML = `
    <h3>Response</h3><pre>${esc(r.response_text || '(empty)')}</pre>
    ${r.thinking_text ? `<h3>Thinking (separate from the response)</h3><pre>${esc(r.thinking_text)}</pre>` : ''}
    ${r.error ? `<h3>Error</h3><pre>${esc(r.error)}</pre>` : ''}
    <h3>Parameters — layers are never merged</h3>
    <table class="params"><thead><tr>
      <th>parameter</th><th>requested</th><th>inferred</th><th>confirmed</th><th>base (if ≠)</th>
    </tr></thead><tbody>${rows}</tbody></table>
    <p class="notice-inline">“Inferred” = proxy estimate (request &gt; Modelfile &gt; default from
      Ollama). The server does not echo sampling parameters; only the “confirmed” column comes from the response.</p>
    <h3>Metrics</h3>
    <p>${['eval_count', 'tokens_per_second', 'time_to_first_token_ms', 'prompt_eval_count', 'done_reason']
      .filter(k => r[k] != null)
      .map(k => `<span class="metric">${k} <b>${typeof r[k] === 'number' ? +r[k].toFixed(2) : esc(String(r[k]))}</b></span>`)
      .join(' ')}</p>
    ${hasTimingBreakdown(r) ? `<h3>Request timeline</h3>
    <div class="request-timeline"><div class="chart-wrap" id="drawer-timeline"></div>
      <div class="request-timeline-legend">
        <span><i class="seg-load"></i> load</span><span><i class="seg-prompt"></i> prompt eval</span>
        <span><i class="seg-eval"></i> eval</span></div></div>` : ''}
    <h3>Evaluations</h3><p>${evals || '<span class="empty">none</span>'}</p>
    <div class="labels-row" id="drawer-labels"></div>
    <h3>Context</h3>
    <details><summary class="label">prompt / messages</summary>
      <pre>${esc(JSON.stringify(r.prompt_or_messages, null, 2) || '')}</pre></details>
    <details><summary class="label">raw request body (audit/replay)</summary>
      <pre>${esc(r.request_body_raw || '')}</pre></details>
    ${r.observed_runtime ? `<details><summary class="label">observed runtime (/api/ps — not confirmation)</summary>
      <pre>${esc(JSON.stringify(r.observed_runtime, null, 2))}</pre></details>` : ''}
    ${r.baseline ? `<details><summary class="label">model baseline (digest ${esc((r.baseline.digest || '?').slice(0, 12))})</summary>
      <pre>${esc(r.baseline.parameters_raw || '(no parameters in Modelfile)')}</pre></details>` : ''}
    ${r.environment ? `<h3>Environment</h3><p class="notice-inline">Ollama ${esc(r.environment.ollama_version || '?')}
      · ${esc(r.environment.os || '')} · Python ${esc(r.environment.python_version || '')}
      · ${esc(r.environment.cpu || '')} · ${r.environment.ram_gb || '?'} GB RAM</p>` : ''}`;

  if (hasTimingBreakdown(r)) renderDrawerTimeline(r);

  const labelsRow = $('#drawer-labels');
  for (const label of state.labels) {
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.textContent = humanLabelText(label);
    chip.onclick = async () => {
      await fetch(`${API}/requests/${id}/evaluations`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ labels: [label] }) });
      chip.classList.add('on');
    };
    labelsRow.appendChild(chip);
  }
  $('#drawer').classList.add('open');
}

function hasTimingBreakdown(r) {
  return r.load_duration_ns != null || r.prompt_eval_duration_ns != null || r.eval_duration_ns != null;
}

function renderDrawerTimeline(r) {
  const el = $('#drawer-timeline');
  if (!el) return;
  const ms = ns => ns != null ? ns / 1e6 : 0;
  const t = LomboCharts.tokens();
  LomboCharts.mount(el, {
    animation: false, grid: { left: 4, right: 4, top: 4, bottom: 4, containLabel: false },
    xAxis: { type: 'value', show: false },
    yAxis: { type: 'category', show: false, data: [''] },
    tooltip: { trigger: 'item', formatter: p => `${p.seriesName}: ${p.value.toFixed(0)} ms` },
    series: [
      { name: 'load', type: 'bar', stack: 't', data: [ms(r.load_duration_ns)],
       itemStyle: { color: t.paneC }, barWidth: '70%' },
      { name: 'prompt eval', type: 'bar', stack: 't', data: [ms(r.prompt_eval_duration_ns)],
       itemStyle: { color: t.paneB } },
      { name: 'eval', type: 'bar', stack: 't', data: [ms(r.eval_duration_ns)],
       itemStyle: { color: t.accent } },
    ],
  }, { toolbox: false, renderer: 'canvas' });
}

async function runNNsightTrace(r) {
  const input = prompt('Layer index to capture:', '0');
  if (input === null) return;
  const layerIndex = Number(input);
  if (!Number.isInteger(layerIndex)) { alert('Enter an integer index.'); return; }
  const btn = $('#btn-inspect');
  btn.disabled = true;
  btn.textContent = 'Running trace…';
  const content = $('#drawer-content');
  const status = document.createElement('p');
  status.className = 'notice-inline';
  status.textContent = `Running NNsight on layer ${layerIndex}…`;
  content.prepend(status);
  try {
    const resp = await fetch(`${API}/requests/${r.id}/trace`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ layer_index: layerIndex }),
    });
    const result = await resp.json();
    if (!resp.ok) throw new Error(result.detail || JSON.stringify(result));
    const logits = result.last_token_logits || [];
    const summary = {
      layer_index: result.layer_index,
      activation_shape: result.activation_shape,
      activation_dtype: result.activation_dtype,
      tokens: result.tokens,
      last_token_logits_count: logits.length,
      last_token_logits_preview: logits.slice(0, 20),
      model: result.model,
    };
    status.textContent = 'NNsight trace completed.';
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(summary, null, 2);
    status.after(pre);
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
    status.style.color = 'var(--err)';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run NNsight trace';
  }
}
/* ============ attention map (real internal weights, per request) ============ */
async function runAttention(r) {
  const btn = $('#btn-attention');
  const content = $('#drawer-content');
  let panel = $('#attn-panel');
  if (!panel) {
    panel = document.createElement('div');
    panel.id = 'attn-panel';
    panel.className = 'attn-panel';
    content.prepend(panel);
  }
  const st = { layer: 0, head: null };
  async function load() {
    const status = panel.querySelector('.attn-status');
    if (status) { status.textContent = 'Capturing…'; status.style.color = ''; }
    else panel.innerHTML = '<p class="notice-inline">Capturing attention…</p>';
    btn.disabled = true;
    try {
      const resp = await fetch(`${API}/requests/${r.id}/attention`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ layer_index: st.layer, head_index: st.head }),
      });
      const d = await resp.json();
      if (!resp.ok) throw new Error(d.detail || JSON.stringify(d));
      renderAttnPanel(panel, d, st, load);
    } catch (err) {
      panel.innerHTML = `<p class="notice-inline" style="color:var(--err)">Attention map — error: ${esc(err.message)}</p>`;
    } finally {
      btn.disabled = false;
    }
  }
  await load();
}

function renderAttnPanel(panel, d, st, reload) {
  const layerOpts = Array.from({ length: d.num_layers }, (_, i) =>
    `<option value="${i}"${i === d.layer_index ? ' selected' : ''}>L${i}</option>`).join('');
  const headOpts = ['<option value="">mean (all heads)</option>'].concat(
    Array.from({ length: d.num_heads }, (_, i) =>
      `<option value="${i}"${i === d.head_index ? ' selected' : ''}>head ${i}</option>`)).join('');
  panel.innerHTML = `
    <div class="attn-head">
      <h3 style="margin:0">Attention map</h3>
      <label class="attn-ctl">layer <select id="attn-layer">${layerOpts}</select></label>
      <label class="attn-ctl">head <select id="attn-head">${headOpts}</select></label>
      <span class="notice-inline attn-status"></span>
    </div>
    <div class="attn-scroll">${attnHeatmap(d)}</div>
    <p class="notice-inline"><b>Real</b> attention weights (read-only forward pass). Row = token that
      <i>queries</i> (query); column = <i>attended</i> token (key); each row sums to 1. Shows
      information routing — <b>suggestive</b>, not causal proof of use.</p>`;
  panel.querySelector('#attn-layer').onchange = e => { st.layer = Number(e.target.value); reload(); };
  panel.querySelector('#attn-head').onchange = e => {
    st.head = e.target.value === '' ? null : Number(e.target.value); reload();
  };
}

/* SVG heatmap: row i = query, column j = key; amber opacity ∝ weight. */
function attnHeatmap(d) {
  const m = d.matrix, n = m.length;
  if (!n) return '<p class="notice-inline">no tokens</p>';
  const vmax = Math.max(1e-9, ...m.flat());
  const cell = Math.max(7, Math.min(22, Math.floor(520 / n)));
  const labels = cell >= 11;
  const clip = s => { s = (s || '').replace(/\n/g, '⏎').replace(/\t/g, '⇥');
    return s.length > 10 ? s.slice(0, 10) + '…' : s; };
  const padL = labels ? 92 : 6, padT = labels ? 86 : 6;
  const W = padL + n * cell + 6, H = padT + n * cell + 6;
  let rects = '';
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const w = m[i][j];
      rects += `<rect x="${padL + j * cell}" y="${padT + i * cell}" width="${cell - 0.5}" height="${cell - 0.5}"
        fill="var(--accent)" fill-opacity="${(w / vmax).toFixed(3)}"><title>q«${esc(d.tokens[i])}» → k«${esc(d.tokens[j])}»: ${w.toFixed(4)}</title></rect>`;
    }
  }
  let axis = '';
  if (labels) {
    for (let i = 0; i < n; i++) {
      const t = esc(clip(d.tokens[i]));
      const c = padL + i * cell + cell * 0.72;
      axis += `<text x="${padL - 6}" y="${padT + i * cell + cell * 0.72}" class="attn-lbl" text-anchor="end">${t}</text>`;
      axis += `<text x="${c}" y="${padT - 6}" class="attn-lbl" text-anchor="start" transform="rotate(-90 ${c} ${padT - 6})">${t}</text>`;
    }
  }
  return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="attn-svg">${rects}${axis}</svg>`;
}

$('#btn-close-drawer').onclick = () => $('#drawer').classList.remove('open');
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') $('#drawer').classList.remove('open');
});

/* ================= NNsight replay bundle (out of scope: no activations run here) =================
   Reuses the same detail drawer (#drawer/#d-model/#d-id/#drawer-content). */
const NNSIGHT_NOTICE =
  'This bundle contains everything a deterministic replay needs: exact prompt, seed, ' +
  'parameters in the requested/inferred layers and the model digest. It does NOT run NNsight or ' +
  'extract activations — LomboProxy only observes the external inference controls of Ollama. ' +
  'Note: Ollama serves GGUF weights (llama.cpp); NNsight needs the HuggingFace/PyTorch version ' +
  'of the model, so inspected activations come from that other version, not literally from the ' +
  'same runtime that generated this response.';

function seedOf(r) {
  if (r.requested_params && 'seed' in r.requested_params) return r.requested_params.seed;
  const inf = r.inferred_params && r.inferred_params.seed;
  return inf ? inf.value : null;
}

function bundleExecution(r) {
  return {
    condition: r.condition || null,
    model: r.model,
    model_digest: r.model_digest || (r.baseline && r.baseline.digest) || null,
    system_prompt: r.system_prompt || null,
    messages: r.prompt_or_messages || null,
    seed: seedOf(r),
    params_solicitados: r.requested_params || {},
    params_inferidos: r.inferred_params || {},
    response_text: r.response_text || null,
    thinking_text: r.thinking_text || null,
  };
}

async function openNNsightBundle(pairId, singleRequest) {
  let execucoes, id;
  if (singleRequest) {
    execucoes = [bundleExecution(singleRequest)];
    id = singleRequest.id;
  } else {
    const data = await (await fetch(`${API}/compare?pair_id=${pairId}`)).json();
    execucoes = (data.requests || []).map(bundleExecution);
    id = pairId;
  }
  const bundle = { formato: 'lombobench-nnsight-replay-bundle-v1', aviso: NNSIGHT_NOTICE, execucoes };
  showBundleDrawer(bundle, id);
}

function showBundleDrawer(bundle, id) {
  $('#d-model').textContent = 'NNsight replay bundle';
  $('#d-id').textContent = id || '';
  $('#btn-inspect').onclick = null;
  const json = JSON.stringify(bundle, null, 2);
  $('#drawer-content').innerHTML = `
    <p class="notice-inline">${esc(bundle.aviso)}</p>
    <div class="row" style="margin:10px 0 4px"><button class="btn primary" id="btn-copy-bundle">Copy JSON</button></div>
    <pre>${esc(json)}</pre>`;
  $('#btn-copy-bundle').onclick = async () => {
    await navigator.clipboard.writeText(json);
    const btn = $('#btn-copy-bundle');
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy JSON'; }, 1500);
  };
  $('#drawer').classList.add('open');
}
