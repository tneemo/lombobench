/* Lombo Lab — front vanilla. O Lab é só mais um cliente do proxy:
   envia POST /api/chat com headers X-Lombo-* e tudo cai nas mesmas tabelas. */
'use strict';

const API = '/_lombo/api';
const $ = (sel, el) => (el || document).querySelector(sel);
const rid = () => crypto.randomUUID().replace(/-/g, '').slice(0, 12);

const PANE_COLORS = ['var(--pane-a)', 'var(--pane-b)', 'var(--pane-c)', 'var(--pane-d)'];
const NUMERIC_PARAMS = new Set(['temperature', 'top_p', 'top_k', 'seed', 'repeat_penalty',
                                'num_ctx', 'num_predict', 'min_p']);
const VARIABLES = [
  ['', 'base (controle)'], ['model', 'model'], ['prompt', 'prompt (mensagem)'],
  ['temperature', 'temperature'], ['top_p', 'top_p'], ['top_k', 'top_k'], ['seed', 'seed'],
  ['repeat_penalty', 'repeat_penalty'], ['num_ctx', 'num_ctx'],
  ['num_predict', 'num_predict'], ['system', 'system prompt'],
];
// parâmetros que NÃO viram options.* do Ollama — resolvidos por outro caminho
const NON_OPTION_VARIABLES = new Set(['model', 'system', 'prompt']);

const state = {
  session: null,     // {id, mode, base_config, panes:[{name, variable, value}]}
  panes: [],         // runtime: {idx, name, variable, value, condition, history:[], lastRequestId}
  models: [],
  labels: [],
  sending: false,
};

/* ================= abas ================= */
function switchView(name) {
  for (const v of ['lab', 'analysis']) {
    $('#view-' + v).classList.toggle('active', v === name);
    $('#tab-' + v).setAttribute('aria-selected', String(v === name));
  }
  if (name === 'analysis') loadRequests();
}
$('#tab-lab').onclick = () => switchView('lab');
$('#tab-analysis').onclick = () => switchView('analysis');

/* ================= setup de sessão ================= */
function paneConfigRow(i, preset) {
  const row = document.createElement('div');
  row.className = 'pane-config';
  row.style.setProperty('--pane-color', PANE_COLORS[i % PANE_COLORS.length]);
  row.innerHTML = `
    <label class="field"><span class="label">Painel</span>
      <input class="p-name" value="${preset?.name || String.fromCharCode(65 + i)}"></label>
    <label class="field"><span class="label">Variável manipulada</span>
      <select class="p-var">${VARIABLES.map(([v, l]) =>
        `<option value="${v}"${preset?.variable === v ? ' selected' : ''}>${l}</option>`).join('')}
      </select></label>
    <label class="field"><span class="label">Valor</span>
      <input class="p-val" value="${preset?.value ?? ''}" placeholder="—"></label>
    <button class="btn p-del" type="button" title="remover painel">×</button>`;
  row.querySelector('.p-del').onclick = () => row.remove();
  return row;
}

function initSetup() {
  const wrap = $('#pane-configs');
  wrap.innerHTML = '';
  wrap.appendChild(paneConfigRow(0, { variable: '' }));
  wrap.appendChild(paneConfigRow(1, { variable: 'prompt', value: 'variante' }));
}
$('#btn-add-pane').onclick = () =>
  $('#pane-configs').appendChild(paneConfigRow($('#pane-configs').children.length));

async function loadModels() {
  try {
    const data = await (await fetch(`${API}/lab/models`)).json();
    state.models = (data.models || []).map(m => m.name);
  } catch { state.models = []; }
  $('#ollama-models').innerHTML = state.models.map(m => `<option value="${esc(m)}">`).join('');
  if (!$('#s-model').value && state.models.length) $('#s-model').value = state.models[0];
}

$('#s-engine').onchange = () => {
  const hf = $('#s-engine').value === 'transformers';
  $('#s-model').placeholder = hf ? 'ex.: Qwen/Qwen2.5-0.5B-Instruct' : 'modelo Ollama';
  if (hf) $('#s-model').value = '';
  else if (!$('#s-model').value && state.models.length) $('#s-model').value = state.models[0];
};

$('#btn-create-session').onclick = async () => {
  const base_config = { engine: $('#s-engine').value, model: $('#s-model').value.trim() };
  if (!base_config.model) { alert('Informe o modelo.'); return; }
  if ($('#s-temp').value !== '') base_config.temperature = parseFloat($('#s-temp').value);
  if ($('#s-seed').value !== '') base_config.seed = parseInt($('#s-seed').value, 10);
  if ($('#s-max-tokens').value !== '') {
    base_config.num_predict = parseInt($('#s-max-tokens').value, 10);
  }
  if ($('#s-system').value.trim()) base_config.system = $('#s-system').value.trim();
  const panes = [...$('#pane-configs').children].map(row => ({
    name: $('.p-name', row).value.trim() || '?',
    variable: $('.p-var', row).value,
    value: $('.p-val', row).value.trim(),
  }));
  if (panes.length < 2) { alert('Crie pelo menos 2 painéis para comparar.'); return; }
  const body = {
    title: $('#s-title').value.trim() || `Sessão ${new Date().toLocaleString()}`,
    mode: document.querySelector('input[name="s-mode"]:checked').value,
    base_config, panes,
  };
  const resp = await fetch(`${API}/lab/sessions`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body) });
  openSession(await resp.json(), []);
};

async function loadSessionList() {
  const data = await (await fetch(`${API}/lab/sessions`)).json();
  const list = $('#sessions-list');
  list.innerHTML = '';
  for (const s of data.sessions || []) {
    const b = document.createElement('button');
    b.innerHTML = `<strong>${esc(s.title)}</strong><br>
      <span class="meta">${s.mode} · ${s.base_config.model || '?'} · ${s.created_at.slice(0, 16)}</span>`;
    b.onclick = async () => {
      const full = await (await fetch(`${API}/lab/sessions/${s.id}`)).json();
      openSession(full, full.requests || []);
    };
    list.appendChild(b);
  }
  $('#sessions-prev').hidden = !(data.sessions || []).length;
}

/* ================= sessão ativa ================= */
function conditionOf(pane) {
  return pane.variable ? `${pane.variable}=${pane.value}` : 'base';
}

function openSession(session, pastRequests) {
  state.session = session;
  state.panes = session.panes.map((p, idx) => ({
    ...p, idx, condition: conditionOf(p), history: [], lastRequestId: null,
  }));
  $('#setup').hidden = true;
  $('#session-bar').hidden = false;
  $('#bench').hidden = false;
  $('#composer').hidden = false;
  $('#session-title').textContent = session.title;
  const mode = $('#session-mode');
  mode.textContent = session.mode === 'controlled' ? 'experimento controlado' : 'conversa ramificada';
  mode.classList.toggle('controlled', session.mode === 'controlled');
  $('#session-base').textContent = 'base: ' + Object.entries(session.base_config)
    .map(([k, v]) => `${k}=${String(v).slice(0, 40)}`).join(' · ');

  const heads = $('#pane-heads');
  heads.innerHTML = '';
  heads.style.gridTemplateColumns = `repeat(${state.panes.length}, 1fr)`;
  for (const pane of state.panes) {
    const h = document.createElement('div');
    h.className = 'pane-head';
    h.style.setProperty('--pane-color', PANE_COLORS[pane.idx % PANE_COLORS.length]);
    h.innerHTML = `<span class="pane-name">painel ${esc(pane.name)}</span>
      <span class="pane-cond ${pane.variable ? '' : 'base'}">${esc(pane.condition)}</span>`;
    heads.appendChild(h);
  }
  $('#turns').innerHTML = '';
  renderPastTurns(pastRequests);
  buildComposerBoxes();
  // se alguma variável manipulada for o próprio prompt, começa dessincronizado
  // — cada painel precisa mesmo de um texto diferente.
  $('#chk-sync').checked = !state.panes.some(p => p.variable === 'prompt');
}

// Casa um request com o painel que o gerou. branch_id = "pane-{idx}" é
// enviado em TODO envio (não só no modo ramificado) só pra isso — é a única
// forma confiável, já que duas panes podem ter a mesma condition (dois
// modelos comparados, só um com variável declarada, etc.). condition entra
// só como fallback pra registros antigos gravados antes desse identificador.
function findReqForPane(list, pane) {
  return list.find(r => r.branch_id === `pane-${pane.idx}`) ||
         list.find(r => r.condition === pane.condition);
}

function renderPastTurns(requests) {
  const byTurn = new Map();
  for (const r of requests) {
    const key = `${r.turn_id}|${r.replicate || 1}`;
    if (!byTurn.has(key)) byTurn.set(key, []);
    byTurn.get(key).push(r);
  }
  for (const group of byTurn.values()) {
    const texts = state.panes.map(pane => {
      const req = findReqForPane(group, pane);
      if (!req) return '';
      try {
        const msgs = JSON.parse(req.prompt_or_messages_json || '[]');
        const lastUser = [...msgs].reverse().find(m => m.role === 'user');
        return lastUser ? lastUser.content : '';
      } catch { return ''; }
    });
    const turn = makeTurnDom(texts, null, group[0].replicate || 1);
    for (const pane of state.panes) {
      const req = findReqForPane(group, pane);
      const cell = turn.cells[pane.idx];
      if (!req) { $('.body', cell).innerHTML = '<span class="empty">—</span>'; continue; }
      if (req.thinking_text) setThinking(cell, req.thinking_text);
      $('.body', cell).textContent = req.response_text || '';
      if (req.error) appendError(cell, req.error);
      finishCellFooter(cell, { id: req.id, eval_count: req.eval_count,
        tokens_per_second: req.tokens_per_second,
        time_to_first_token_ms: req.time_to_first_token_ms });
    }
    const respTexts = state.panes.map(pane => {
      const req = findReqForPane(group, pane);
      return req ? (req.response_text || '') : '';
    });
    if (respTexts.filter(Boolean).length >= 2)
      renderDiff(turn, respTexts[0], respTexts[1], group[0].pair_id);
  }
}

/* ================= envio: um composer por painel ================= */
function buildComposerBoxes() {
  const grid = $('#composer-grid');
  grid.innerHTML = '';
  grid.style.gridTemplateColumns = `repeat(${state.panes.length}, 1fr)`;
  for (const pane of state.panes) {
    const field = document.createElement('div');
    field.className = 'field';
    field.style.setProperty('--pane-color', PANE_COLORS[pane.idx % PANE_COLORS.length]);
    field.innerHTML = `<span class="label" style="color:${PANE_COLORS[pane.idx % PANE_COLORS.length]}">
        painel ${esc(pane.name)}</span>
      <textarea class="composer-box" placeholder="Mensagem para o painel ${esc(pane.name)}…"></textarea>`;
    grid.appendChild(field);
  }
  const boxes = [...grid.querySelectorAll('.composer-box')];
  for (const box of boxes) {
    box.addEventListener('input', () => {
      if ($('#chk-sync').checked) for (const other of boxes) if (other !== box) other.value = box.value;
    });
  }
}
$('#chk-sync').onchange = () => {
  if (!$('#chk-sync').checked) return;
  const boxes = [...document.querySelectorAll('.composer-box')];
  const master = boxes[0]?.value || '';
  for (const b of boxes) b.value = master;
};

$('#btn-send').onclick = () => {
  if (state.sending || !state.session) return;
  const boxes = [...document.querySelectorAll('.composer-box')];
  const texts = boxes.map(b => b.value.trim());
  if (!texts.some(Boolean)) return;
  if (texts.some(t => !t)) {
    alert('Preencha a mensagem de todos os painéis (ou ligue "mensagens sincronizadas").');
    return;
  }
  sendTurn(texts, rid(), 1);
  for (const b of boxes) b.value = '';
};

async function sendTurn(texts, turnId, replicate) {
  state.sending = true;
  $('#btn-send').disabled = true;
  $('#composer-status').textContent = 'gerando…';
  const pairId = rid();
  // seed explícita por rodada (igual em todos os painéis) quando ninguém definiu:
  // garante que a execução seja replayável (gancho da Fase 3 / NNsight).
  const injectedSeed = Math.floor(Math.random() * 2 ** 31);
  const turn = makeTurnDom(texts, { texts, turnId }, replicate);

  const statusPoll = state.session.base_config.engine === 'transformers'
    ? pollTransformerStatus(state.session.base_config.model) : null;
  await Promise.all(state.panes.map((pane, i) => {
    const prompt = texts[i];
    const condition = pane.condition +
      (pane.variable !== 'prompt' && texts[i] !== texts[0] ? '+prompt≠' : '');
    return streamToCell(pane, prompt, condition, turn.cells[pane.idx],
                        { turnId, pairId, replicate, seed: injectedSeed });
  }));
  if (statusPoll) statusPoll.stop();

  await attachPairData(turn, pairId);
  state.sending = false;
  $('#btn-send').disabled = false;
  $('#composer-status').textContent = '';
}

function pollTransformerStatus(model) {
  let active = true;
  const tick = async () => {
    try {
      const resp = await fetch(`${API}/lab/engine-status?model=${encodeURIComponent(model)}`);
      const status = await resp.json();
      if (active && status.state !== 'idle') {
        $('#composer-status').textContent = status.message +
          (status.state === 'loading' ? '…' : '');
      }
    } catch { /* a requisição principal exibirá o erro de conexão */ }
    if (active) setTimeout(tick, 750);
  };
  tick();
  return { stop() { active = false; } };
}

function buildRequest(pane, prompt, seed) {
  const base = state.session.base_config;
  const engine = base.engine || 'ollama';
  const model = pane.variable === 'model' ? pane.value : base.model;
  const options = {};
  if (base.temperature !== undefined) options.temperature = base.temperature;
  options.seed = base.seed !== undefined ? base.seed : seed;
  if (base.num_predict !== undefined) options.num_predict = base.num_predict;
  else if (engine === 'transformers') options.num_predict = 256;
  if (pane.variable && !NON_OPTION_VARIABLES.has(pane.variable)) {
    options[pane.variable] = NUMERIC_PARAMS.has(pane.variable)
      ? Number(pane.value) : pane.value;
  }
  let system = base.system;
  if (pane.variable === 'system') system = pane.value;

  const messages = [];
  if (system) messages.push({ role: 'system', content: system });
  if (state.session.mode === 'branched') messages.push(...pane.history);
  messages.push({ role: 'user', content: prompt });
  return { engine, model, options, messages };
}

async function streamToCell(pane, prompt, condition, cell, meta) {
  const { engine, model, options, messages } = buildRequest(pane, prompt, meta.seed);
  const headers = {
    'Content-Type': 'application/json',
    'X-Lombo-Origin': 'lab',
    'X-Lombo-Engine': engine,
    'X-Lombo-Session': state.session.id,
    'X-Lombo-Turn': meta.turnId,
    'X-Lombo-Pair': meta.pairId,
    // HTTP headers só aceitam ISO-8859-1; condition é texto livre (pode ter
    // '≠', emoji, aspas curvas...) então vai percent-encoded — o proxy decodifica.
    'X-Lombo-Condition': encodeURIComponent(condition),
    'X-Lombo-Replicate': String(meta.replicate),
  };
  // identifica o painel de origem independente do texto de condition — duas
  // panes podem ter a MESMA condition (ex.: dois modelos, só um deles com
  // variável declarada), então casar por condition sozinho gruda as duas no
  // mesmo registro. branch_id também encadeia o histórico no modo ramificado.
  headers['X-Lombo-Branch'] = `pane-${pane.idx}`;
  if (state.session.mode === 'branched' && pane.lastRequestId) {
    headers['X-Lombo-Parent'] = pane.lastRequestId;
  }
  const bodyEl = $('.body', cell);
  bodyEl.innerHTML = '<span class="cursor">▍</span>';
  let responseText = '', thinkingText = '';
  try {
    const resp = await fetch('/api/chat', { method: 'POST', headers,
      body: JSON.stringify({ model, options, messages, stream: true }) });
    if (!resp.ok) {
      appendError(cell, `HTTP ${resp.status}: ${(await resp.text()).slice(0, 300)}`);
      bodyEl.textContent = '';
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        let obj;
        try { obj = JSON.parse(line); } catch { continue; }
        if (obj.error) {
          appendError(cell, obj.error);
          bodyEl.textContent = '';
          continue;
        }
        const msg = obj.message || {};
        if (msg.thinking) { thinkingText += msg.thinking; setThinking(cell, thinkingText); }
        if (msg.content) responseText += msg.content;
        bodyEl.textContent = responseText;
        bodyEl.insertAdjacentHTML('beforeend', '<span class="cursor">▍</span>');
        if (obj.done) {
          cell.dataset.metrics = JSON.stringify({
            eval_count: obj.eval_count,
            tokens_per_second: obj.eval_count && obj.eval_duration
              ? obj.eval_count / (obj.eval_duration / 1e9) : null,
          });
        }
      }
    }
  } catch (err) {
    appendError(cell, String(err));
  }
  bodyEl.textContent = responseText;
  cell.dataset.responseText = responseText;
  if (state.session.mode === 'branched' && responseText) {
    pane.history.push({ role: 'user', content: prompt },
                      { role: 'assistant', content: responseText });
  }
}

/* após a rodada: busca os registros gravados (ids, métricas, ttft) pelo pair_id */
async function attachPairData(turn, pairId) {
  let requests = [];
  for (let attempt = 0; attempt < 12; attempt++) {
    await new Promise(r => setTimeout(r, 600));
    try {
      const data = await (await fetch(`${API}/compare?pair_id=${pairId}`)).json();
      requests = data.requests || [];
      if (requests.length >= state.panes.length) break;
    } catch { /* proxy ocupado; tenta de novo */ }
  }
  for (const pane of state.panes) {
    const cell = turn.cells[pane.idx];
    const req = findReqForPane(requests, pane);
    const local = cell.dataset.metrics ? JSON.parse(cell.dataset.metrics) : {};
    finishCellFooter(cell, req ? {
      id: req.id, eval_count: req.eval_count,
      tokens_per_second: req.tokens_per_second,
      time_to_first_token_ms: req.time_to_first_token_ms,
    } : { ...local, id: null });
    if (state.session.mode === 'branched' && req) pane.lastRequestId = req.id;
  }
  const texts = state.panes.map(p => turn.cells[p.idx].dataset.responseText || '');
  if (texts.filter(Boolean).length >= 2) renderDiff(turn, texts[0], texts[1], pairId);
}

/* ================= DOM da rodada ================= */
function makeTurnDom(texts, resendInfo, replicate) {
  const turnEl = document.createElement('div');
  turnEl.className = 'turn';
  const meta = document.createElement('div');
  meta.className = 'turn-meta';
  meta.innerHTML = `<span class="label">rodada${replicate > 1 ? ` · réplica ${replicate}` : ''}</span>`;
  if (resendInfo && state.session.mode === 'controlled') {
    const btn = document.createElement('button');
    btn.className = 'btn';
    btn.style.marginLeft = '10px';
    btn.textContent = 'reenviar (nova réplica)';
    let count = replicate;
    btn.onclick = () => { if (!state.sending) sendTurn(resendInfo.texts, resendInfo.turnId, ++count); };
    meta.appendChild(btn);
  }
  turnEl.appendChild(meta);

  const allSame = texts.every(t => t === texts[0]);
  if (allSame) {
    const user = document.createElement('div');
    user.className = 'user-msg';
    user.textContent = texts[0] || '(prompt indisponível)';
    turnEl.appendChild(user);
  } else {
    // mensagens divergem entre painéis: uma bolha por painel, não uma única
    // bolha compartilhada — é exatamente o caso "mesma base, prompts diferentes".
    const row = document.createElement('div');
    row.className = 'user-msgs';
    row.style.gridTemplateColumns = `repeat(${state.panes.length}, 1fr)`;
    state.panes.forEach((pane, i) => {
      const bubble = document.createElement('div');
      bubble.className = 'user-msg pane-msg';
      bubble.style.setProperty('--pane-color', PANE_COLORS[pane.idx % PANE_COLORS.length]);
      bubble.innerHTML = `<span class="pane-msg-name">${esc(pane.name)}</span>${esc(texts[i] || '(vazio)')}`;
      row.appendChild(bubble);
    });
    turnEl.appendChild(row);
  }

  const grid = document.createElement('div');
  grid.className = 'responses';
  grid.style.gridTemplateColumns = `repeat(${state.panes.length}, 1fr)`;
  const cells = state.panes.map(pane => {
    const cell = document.createElement('div');
    cell.className = 'cell';
    cell.style.setProperty('--pane-color', PANE_COLORS[pane.idx % PANE_COLORS.length]);
    cell.innerHTML = '<div class="body"></div><footer></footer>';
    grid.appendChild(cell);
    return cell;
  });
  turnEl.appendChild(grid);
  $('#turns').appendChild(turnEl);
  turnEl.scrollIntoView({ block: 'end' });
  return { el: turnEl, grid, cells };
}

function setThinking(cell, text) {
  let box = $('.thinking', cell);
  if (!box) {
    box = document.createElement('details');
    box.className = 'thinking';
    box.open = true;
    box.innerHTML = '<summary>thinking</summary><div class="t-text"></div>';
    cell.prepend(box);
  }
  $('.t-text', box).textContent = text;
}

function appendError(cell, text) {
  const e = document.createElement('div');
  e.className = 'error';
  e.textContent = text;
  cell.appendChild(e);
}

function finishCellFooter(cell, info) {
  const footer = $('footer', cell);
  footer.innerHTML = '';
  const m = (label, v, digits) => v != null
    ? `<span class="metric">${label} <b>${Number(v).toFixed(digits)}</b></span>` : '';
  footer.insertAdjacentHTML('beforeend',
    m('tok', info.eval_count, 0) + m('tok/s', info.tokens_per_second, 1) +
    m('ttft ms', info.time_to_first_token_ms, 0));
  if (info.id) {
    const labels = document.createElement('div');
    labels.className = 'labels-row';
    for (const label of state.labels) {
      const chip = document.createElement('button');
      chip.className = 'chip';
      chip.textContent = label.replace(/_/g, ' ');
      chip.onclick = async () => {
        chip.disabled = true;
        await fetch(`${API}/requests/${info.id}/evaluations`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ labels: [label] }) });
        chip.disabled = false;
        chip.classList.add('on');
      };
      labels.appendChild(chip);
    }
    footer.appendChild(labels);
    const detail = document.createElement('button');
    detail.className = 'btn';
    detail.textContent = 'detalhes';
    detail.onclick = () => openDrawer(info.id);
    footer.appendChild(detail);
  }
}

/* ================= diff palavra a palavra (LCS) ================= */
function wordDiff(a, b) {
  const A = a.split(/\s+/).slice(0, 800), B = b.split(/\s+/).slice(0, 800);
  const n = A.length, m = B.length;
  const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (A[i] === B[j]) { out.push(['=', A[i]]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) out.push(['-', A[i++]]);
    else out.push(['+', B[j++]]);
  }
  while (i < n) out.push(['-', A[i++]]);
  while (j < m) out.push(['+', B[j++]]);
  return out;
}

function renderDiff(turn, textA, textB, pairId) {
  let strip = $('.diff-strip', turn.grid);
  if (!strip) {
    strip = document.createElement('div');
    strip.className = 'diff-strip';
    turn.grid.appendChild(strip);
  }
  const parts = wordDiff(textA, textB).map(([op, w]) =>
    op === '=' ? esc(w) : op === '-' ? `<del>${esc(w)}</del>` : `<ins>${esc(w)}</ins>`);
  const nameA = state.panes[0].name, nameB = state.panes[1].name;
  strip.innerHTML = `<div class="diff-head">
      <span class="label">diff ${esc(nameA)} → ${esc(nameB)}
        (<del>só em ${esc(nameA)}</del> / <ins>só em ${esc(nameB)}</ins>)${
        state.panes.length > 2 ? ' — painéis extras não incluídos no diff' : ''}</span>
      <button class="btn nn-pair-btn" type="button">Abrir pacote NNsight (par)</button>
    </div><div class="diff-body">${parts.join(' ')}</div>`;
  const btn = $('.nn-pair-btn', strip);
  if (btn) btn.onclick = () => openNNsightBundle(pairId);
}

/* ================= pacote de replay para NNsight (fora do escopo: não roda ativações aqui) ================= */
const NNSIGHT_NOTICE =
  'Este pacote traz tudo que uma reexecução determinística precisa: prompt exato, seed, ' +
  'parâmetros nas camadas solicitada/inferida e o digest do modelo. Ele NÃO roda NNsight nem ' +
  'extrai ativações — o LomboProxy só observa os controles externos de inferência do Ollama. ' +
  'Atenção: Ollama serve pesos GGUF (llama.cpp); NNsight precisa da versão HuggingFace/PyTorch ' +
  'do modelo, então as ativações inspecionadas virão dessa outra versão, não literalmente do ' +
  'mesmo runtime que gerou esta resposta.';

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
  $('#d-model').textContent = 'Pacote de replay NNsight';
  $('#d-id').textContent = id || '';
  $('#btn-inspect').onclick = null;
  const json = JSON.stringify(bundle, null, 2);
  $('#drawer-content').innerHTML = `
    <p class="notice-inline">${esc(bundle.aviso)}</p>
    <div class="row" style="margin:10px 0 4px"><button class="btn primary" id="btn-copy-bundle">Copiar JSON</button></div>
    <pre>${esc(json)}</pre>`;
  $('#btn-copy-bundle').onclick = async () => {
    await navigator.clipboard.writeText(json);
    const btn = $('#btn-copy-bundle');
    btn.textContent = 'Copiado!';
    setTimeout(() => { btn.textContent = 'Copiar JSON'; }, 1500);
  };
  $('#drawer').classList.add('open');
}

/* ================= análise ================= */
async function loadRequests() {
  const params = new URLSearchParams({ limit: '200' });
  if ($('#f-model').value) params.set('model', $('#f-model').value);
  if ($('#f-experiment').value) params.set('experiment', $('#f-experiment').value);
  if ($('#f-origin').value) params.set('origin', $('#f-origin').value);
  const data = await (await fetch(`${API}/requests?${params}`)).json();
  const tbody = $('#req-rows');
  tbody.innerHTML = '';
  for (const r of data.requests || []) {
    const tr = document.createElement('tr');
    const diffKeys = Object.keys(r.params_diff || {}).join(' ');
    tr.innerHTML = `
      <td>${(r.started_at || '').slice(5, 19).replace('T', ' ')}</td>
      <td>${esc(r.model || '?')}</td>
      <td><span class="pill ${r.origin === 'lab' ? 'lab' : ''}">${r.origin}</span></td>
      <td>${esc(r.condition || '—')}</td>
      <td class="diffkeys">${esc(diffKeys || '—')}</td>
      <td class="num">${r.eval_count ?? ''}</td>
      <td class="num">${r.tokens_per_second ? r.tokens_per_second.toFixed(1) : ''}</td>
      <td class="num">${r.time_to_first_token_ms ? r.time_to_first_token_ms.toFixed(0) : ''}</td>
      <td>${r.error ? `<span class="pill err">erro</span>` : `<span class="pill">${r.status_code ?? ''}</span>`}</td>`;
    tr.onclick = () => openDrawer(r.id);
    tbody.appendChild(tr);
  }
}
$('#btn-filter').onclick = loadRequests;

/* ================= drawer de detalhe ================= */
async function openDrawer(id) {
  const r = await (await fetch(`${API}/requests/${id}`)).json();
  state.currentDrawerRequest = r;
  $('#btn-inspect').textContent = 'Executar trace NNsight';
  $('#btn-inspect').onclick = () => runNNsightTrace(r);
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
      <td>${confirmed[k] ? esc(JSON.stringify(confirmed[k].value)) : '<span class="empty">não confirmado</span>'}</td>
      <td>${diff[k] ? esc(JSON.stringify(diff[k].base_value)) + ` <span class="src">(${diff[k].base_source})</span>` : ''}</td>
    </tr>`;
  }).join('');

  const evals = (r.evaluations || []).map(e =>
    `<span class="metric">${esc(e.metric)} <b>${esc(typeof e.value === 'object'
      ? (e.value.label || JSON.stringify(e.value)) : String(e.value))}</b>
      <span class="src">${e.source}</span></span>`).join(' ');

  c.innerHTML = `
    <h3>Resposta</h3><pre>${esc(r.response_text || '(vazia)')}</pre>
    ${r.thinking_text ? `<h3>Thinking (separado da resposta)</h3><pre>${esc(r.thinking_text)}</pre>` : ''}
    ${r.error ? `<h3>Erro</h3><pre>${esc(r.error)}</pre>` : ''}
    <h3>Parâmetros — camadas nunca fundidas</h3>
    <table class="params"><thead><tr>
      <th>parâmetro</th><th>solicitado</th><th>inferido</th><th>confirmado</th><th>base (se ≠)</th>
    </tr></thead><tbody>${rows}</tbody></table>
    <p class="notice-inline">“Inferido” = estimativa do proxy (request &gt; Modelfile &gt; default do
      Ollama). O servidor não ecoa parâmetros de amostragem; só a coluna “confirmado” vem da resposta.</p>
    <h3>Métricas</h3>
    <p>${['eval_count', 'tokens_per_second', 'time_to_first_token_ms', 'prompt_eval_count', 'done_reason']
      .filter(k => r[k] != null)
      .map(k => `<span class="metric">${k} <b>${typeof r[k] === 'number' ? +r[k].toFixed(2) : esc(String(r[k]))}</b></span>`)
      .join(' ')}</p>
    <h3>Avaliações</h3><p>${evals || '<span class="empty">nenhuma</span>'}</p>
    <div class="labels-row" id="drawer-labels"></div>
    <h3>Contexto</h3>
    <details><summary class="label">prompt / messages</summary>
      <pre>${esc(JSON.stringify(r.prompt_or_messages, null, 2) || '')}</pre></details>
    <details><summary class="label">body bruto da requisição (auditoria/replay)</summary>
      <pre>${esc(r.request_body_raw || '')}</pre></details>
    ${r.observed_runtime ? `<details><summary class="label">runtime observado (/api/ps — não é confirmação)</summary>
      <pre>${esc(JSON.stringify(r.observed_runtime, null, 2))}</pre></details>` : ''}
    ${r.baseline ? `<details><summary class="label">baseline do modelo (digest ${esc((r.baseline.digest || '?').slice(0, 12))})</summary>
      <pre>${esc(r.baseline.parameters_raw || '(sem parameters no Modelfile)')}</pre></details>` : ''}
    ${r.environment ? `<h3>Ambiente</h3><p class="notice-inline">Ollama ${esc(r.environment.ollama_version || '?')}
      · ${esc(r.environment.os || '')} · Python ${esc(r.environment.python_version || '')}
      · ${esc(r.environment.cpu || '')} · ${r.environment.ram_gb || '?'} GB RAM</p>` : ''}`;

  const labelsRow = $('#drawer-labels');
  for (const label of state.labels) {
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.textContent = label.replace(/_/g, ' ');
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

async function runNNsightTrace(r) {
  const input = prompt('Índice da camada para capturar:', '0');
  if (input === null) return;
  const layerIndex = Number(input);
  if (!Number.isInteger(layerIndex)) { alert('Informe um índice inteiro.'); return; }
  const btn = $('#btn-inspect');
  btn.disabled = true;
  btn.textContent = 'Executando trace…';
  const content = $('#drawer-content');
  const status = document.createElement('p');
  status.className = 'notice-inline';
  status.textContent = `Executando NNsight na camada ${layerIndex}…`;
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
    status.textContent = 'Trace NNsight concluído.';
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(summary, null, 2);
    status.after(pre);
  } catch (err) {
    status.textContent = `Erro: ${err.message}`;
    status.style.color = 'var(--err)';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Executar trace NNsight';
  }
}
$('#btn-close-drawer').onclick = () => $('#drawer').classList.remove('open');
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') $('#drawer').classList.remove('open');
});

$('#btn-new-session').onclick = () => {
  $('#setup').hidden = false;
  $('#session-bar').hidden = true;
  $('#bench').hidden = true;
  $('#composer').hidden = true;
  loadSessionList();
};

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g,
    ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}

/* ================= boot ================= */
(async function boot() {
  const theme = new URLSearchParams(location.search).get('theme');
  if (theme) document.documentElement.dataset.theme = theme;
  const [, labels] = await Promise.all([loadModels(),
    fetch(`${API}/labels`).then(r => r.json())]);
  state.labels = labels.labels || [];
  initSetup();
  loadSessionList();
  const params = new URLSearchParams(location.search);
  const sid = params.get('session');
  if (sid) {
    const full = await (await fetch(`${API}/lab/sessions/${sid}`)).json();
    if (full.id) openSession(full, full.requests || []);
  }
  if (params.get('view') === 'analysis') switchView('analysis');
  const reqId = params.get('request');
  if (reqId) openDrawer(reqId);
})();
