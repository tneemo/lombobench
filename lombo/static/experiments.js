/* ================= Experimentos: leaderboard, impacto de parâmetro,
   comparação de pares, árvore de branches, reprodutibilidade ================= */
'use strict';

let currentPairId = null;

async function loadExperiments() {
  const experiment = $('#exp-experiment').value;
  const qs = experiment ? `?experiment=${encodeURIComponent(experiment)}` : '';
  try {
    const [summaryData, evalData, reqData, sessionsData] = await Promise.all([
      fetch(`${API}/summary${qs}`).then(r => r.json()),
      fetch(`${API}/evaluations/summary${qs}`).then(r => r.json()),
      fetch(`${API}/requests?limit=200`).then(r => r.json()),
      fetch(`${API}/lab/sessions`).then(r => r.json()),
    ]);
    populateExperimentSelect(reqData.requests || []);
    renderLeaderboard(summaryData.groups || [], evalData.groups || []);
    populateTreeSessionSelect(sessionsData.sessions || []);
  } catch {
    $('#exp-leaderboard').innerHTML = emptyPanel('Proxy indisponível.');
  }
  loadParamImpact();
}

function populateExperimentSelect(requests) {
  const sel = $('#exp-experiment');
  if (sel.dataset.populated) return; // não reconstruir a cada refresh — perderia a seleção do usuário
  const values = [...new Set(requests.map(r => r.experiment).filter(Boolean))].sort();
  sel.innerHTML = '<option value="">todos os experimentos</option>' +
    values.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
  sel.dataset.populated = '1';
  sel.onchange = loadExperiments;
}

function renderLeaderboard(summaryGroups, evalGroups) {
  const wrap = $('#exp-leaderboard');
  if (!summaryGroups.length) {
    wrap.innerHTML = emptyPanel('Sem execuções registradas ainda. Use X-Lombo-Experiment nas suas chamadas ou rode uma sessão no Prompt lab.');
    return;
  }
  const evalByKey = new Map(evalGroups.map(g => [`${g.model}|${g.condition}`, g]));
  const maxTps = Math.max(1, ...summaryGroups.map(g => g.avg_tps || 0));
  const rows = [...summaryGroups].sort((a, b) => (b.avg_tps || 0) - (a.avg_tps || 0));
  wrap.innerHTML = `<table class="dense-table leaderboard-table"><thead><tr>
      <th>MODELO</th><th>CONDIÇÃO</th><th>N</th><th>TOK/S</th><th>TTFT</th><th>RÓTULOS HUMANOS</th></tr></thead><tbody>
    ${rows.map(g => {
      const ev = evalByKey.get(`${g.model}|${g.condition}`);
      const labels = ev ? Object.entries(ev.labels || {})
        .map(([l, n]) => `<span class="chip on" title="${n}×">${esc(l.replace(/_/g, ' '))} · ${n}</span>`).join('') : '';
      return `<tr>
        <td>${esc(g.model || '?')}</td><td>${esc(g.condition || 'base')}</td><td class="num">${g.n}</td>
        <td class="bar-cell"><i class="spark-bar" style="--v:${((g.avg_tps || 0) / maxTps * 100).toFixed(1)}%"></i><b>${(g.avg_tps || 0).toFixed(1)}</b></td>
        <td class="num">${g.avg_ttft_ms != null ? g.avg_ttft_ms.toFixed(0) : '—'}</td>
        <td class="chips-cell">${labels || '<span class="empty">—</span>'}</td></tr>`;
    }).join('')}</tbody></table>`;
}

async function loadParamImpact() {
  const param = $('#exp-param').value;
  try {
    const data = await (await fetch(`${API}/params/impact?param=${encodeURIComponent(param)}`)).json();
    renderParamImpact(data);
  } catch {
    $('#exp-param-chart').innerHTML = emptyPanel('Não foi possível carregar o impacto do parâmetro.');
  }
}
$('#exp-param').onchange = loadParamImpact;

function renderParamImpact(data) {
  const chartEl = $('#exp-param-chart'), tableEl = $('#exp-param-table');
  const groups = (data.groups || []).filter(g => g.n > 0)
    .sort((a, b) => JSON.stringify(a.value).localeCompare(JSON.stringify(b.value)));
  if (!groups.length) {
    LomboCharts.dispose(chartEl); chartEl.innerHTML = emptyPanel('Sem execuções com esse parâmetro inferido ainda.');
    tableEl.innerHTML = '';
    return;
  }
  const categories = groups.map(g => String(g.value));
  LomboCharts.bars(chartEl, {
    categories, series: [{ name: 'tok/s médio', data: groups.map(g => g.avg_tokens_per_second || 0) }],
  }, { yName: 'tok/s', toolbox: false });
  tableEl.innerHTML = `<table class="dense-table"><thead><tr>
      <th>VALOR</th><th>N</th><th>FONTE</th><th>TOK/S</th><th>TOKENS</th><th>TTFT</th></tr></thead><tbody>
    ${groups.map(g => `<tr><td><code>${esc(String(g.value))}</code></td><td class="num">${g.n}</td>
      <td>${g.sources.map(s => `<span class="src-badge ${s}">${esc(s)}</span>`).join(' ')}</td>
      <td class="num">${g.avg_tokens_per_second ?? '—'}</td>
      <td class="num">${g.avg_eval_count ?? '—'}</td>
      <td class="num">${g.avg_ttft_ms ?? '—'}</td></tr>`).join('')}
  </tbody></table>`;
}

/* ================= comparação de par + reprodutibilidade ================= */
window.loadExperimentPair = async function loadExperimentPair(pairId) {
  currentPairId = pairId;
  const wrap = $('#exp-pair-detail');
  wrap.innerHTML = emptyPanel('Carregando par…');
  try {
    const data = await (await fetch(`${API}/compare?pair_id=${encodeURIComponent(pairId)}`)).json();
    renderPairDetail(data);
  } catch (err) {
    wrap.innerHTML = emptyPanel('Não foi possível carregar o par: ' + err.message);
  }
};

function renderPairDetail(data) {
  const wrap = $('#exp-pair-detail');
  const reqs = data.requests || [];
  if (!reqs.length) { wrap.innerHTML = emptyPanel('Par sem execuções.'); return; }
  const cols = reqs.map(r => `<div class="pair-col">
      <header><b>${esc(r.condition || r.model || '?')}</b><span>${esc(r.model || '')}</span></header>
      <div class="pair-metrics">
        <span><small>TOK/S</small><b>${r.tokens_per_second != null ? r.tokens_per_second.toFixed(1) : '—'}</b></span>
        <span><small>TTFT</small><b>${r.time_to_first_token_ms != null ? r.time_to_first_token_ms.toFixed(0) : '—'}</b></span>
        <span><small>TOKENS</small><b>${r.eval_count ?? '—'}</b></span>
      </div>
      <div class="pair-labels">${(r.evaluations || []).filter(e => e.source === 'human')
        .map(e => `<span class="chip on">${esc((e.value?.label || '').replace(/_/g, ' '))}</span>`).join('') || '<span class="empty">sem rótulos humanos</span>'}</div>
      <details><summary class="label">resposta</summary><pre>${esc(r.response_text || '(vazia)')}</pre></details>
      <button class="btn small" data-req="${esc(r.id)}" type="button">Ver no drawer</button>
    </div>`).join('');
  wrap.innerHTML = `<div class="pair-columns">${cols}</div>
    ${reqs.length === 2 ? `<div class="diff-strip" id="exp-pair-diff"></div>` : ''}
    <div class="repro-card">
      <h3>Reprodutibilidade</h3>
      <dl class="repro-rows">
        ${reqs.map(r => `<div><dt>${esc(r.condition || r.model || '?')}</dt>
          <dd>seed ${esc(String(seedOf(r) ?? '—'))} · digest ${esc((r.model_digest || '—').slice(0, 12))}</dd></div>`).join('')}
      </dl>
      <button class="btn" id="exp-copy-bundle" type="button">Copiar bundle de replay</button>
    </div>`;
  wrap.querySelectorAll('[data-req]').forEach(btn => { btn.onclick = () => openDrawer(btn.dataset.req); });
  if (reqs.length === 2) {
    const diffEl = $('#exp-pair-diff');
    const parts = wordDiff(reqs[0].response_text || '', reqs[1].response_text || '').map(([op, w]) =>
      op === '=' ? esc(w) : op === '-' ? `<del>${esc(w)}</del>` : `<ins>${esc(w)}</ins>`);
    diffEl.innerHTML = `<div class="diff-head"><span class="label">diff das respostas</span></div>
      <div class="diff-body">${parts.join(' ')}</div>`;
  }
  $('#exp-copy-bundle').onclick = () => openNNsightBundle(data.pair_id);
}

$('#btn-exp-share').onclick = async () => {
  if (!currentPairId) { alert('Selecione um par primeiro (pelas execuções recentes ou pela árvore de branches).'); return; }
  const url = new URL(location.href);
  url.searchParams.set('view', 'experiments');
  url.searchParams.set('pair', currentPairId);
  await navigator.clipboard.writeText(url.toString());
  const btn = $('#btn-exp-share');
  const original = btn.textContent;
  btn.textContent = 'Link copiado!';
  setTimeout(() => { btn.textContent = original; }, 1500);
};

/* ================= árvore de branches ================= */
function populateTreeSessionSelect(sessions) {
  const sel = $('#exp-tree-session');
  const prev = sel.value;
  if (!sessions.length) { sel.innerHTML = '<option value="">nenhuma sessão</option>'; $('#exp-tree').innerHTML = emptyPanel('Sessões ramificadas aparecem aqui — crie uma no Prompt lab.'); return; }
  sel.innerHTML = sessions.map(s => `<option value="${esc(s.id)}">${esc(s.title)} · ${s.mode === 'controlled' ? 'controlado' : 'ramificado'}</option>`).join('');
  const target = sessions.some(s => s.id === prev) ? prev : sessions[0].id;
  sel.value = target;
  sel.onchange = () => loadBranchTree(sel.value);
  loadBranchTree(target);
}

async function loadBranchTree(sessionId) {
  const wrap = $('#exp-tree');
  if (!sessionId) { wrap.innerHTML = emptyPanel('Selecione uma sessão.'); return; }
  try {
    const data = await (await fetch(`${API}/lab/sessions/${sessionId}/tree`)).json();
    renderBranchTree(data.nodes || []);
  } catch {
    wrap.innerHTML = emptyPanel('Não foi possível carregar a árvore desta sessão.');
  }
}

function renderBranchTree(nodes) {
  const wrap = $('#exp-tree');
  if (!nodes.length) { wrap.innerHTML = emptyPanel('Sem execuções nesta sessão ainda.'); return; }
  const byParent = new Map();
  const byId = new Map(nodes.map(n => [n.id, n]));
  const roots = [];
  for (const n of nodes) {
    if (n.parent_request_id && byId.has(n.parent_request_id)) {
      if (!byParent.has(n.parent_request_id)) byParent.set(n.parent_request_id, []);
      byParent.get(n.parent_request_id).push(n);
    } else roots.push(n);
  }
  const renderNode = (n, depth) => {
    const children = byParent.get(n.id) || [];
    const dot = n.error ? 'run-err' : 'run-ok';
    return `<div class="tree-node" style="--depth:${depth}" data-req="${esc(n.id)}" role="button" tabindex="0">
        <i class="${dot}"></i><span><b>${esc(n.branch_id || n.condition || '?')}</b>
        <small>${esc(n.model || '')}${n.tokens_per_second != null ? ' · ' + n.tokens_per_second.toFixed(1) + ' tok/s' : ''}</small></span>
      </div>${children.map(c => renderNode(c, depth + 1)).join('')}`;
  };
  wrap.innerHTML = roots.map(n => renderNode(n, 0)).join('');
  wrap.querySelectorAll('[data-req]').forEach(el => { el.onclick = () => openDrawer(el.dataset.req); });
}
