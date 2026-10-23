/* Prompt lab — vanilla frontend. The Lab is another proxy client:
   sends POST /api/chat with X-Lombo-* headers and uses the same tables. */
'use strict';

const NUMERIC_PARAMS = new Set(['temperature', 'top_p', 'top_k', 'seed', 'repeat_penalty',
                                'num_ctx', 'num_predict', 'min_p']);
const VARIABLES = [
  ['', 'base (control)'], ['model', 'model'], ['prompt', 'prompt (message)'],
  ['temperature', 'temperature'], ['top_p', 'top_p'], ['top_k', 'top_k'], ['seed', 'seed'],
  ['repeat_penalty', 'repeat_penalty'], ['num_ctx', 'num_ctx'],
  ['num_predict', 'num_predict'], ['system', 'system prompt'],
];
// parameters that do NOT become Ollama options.* — resolved through another path
const NON_OPTION_VARIABLES = new Set(['model', 'system', 'prompt']);

const state = {
  session: null,     // {id, mode, base_config, panes:[{name, variable, value}]}
  panes: [],         // runtime: {idx, name, variable, value, condition, history:[], lastRequestId}
  models: [],
  labels: [],
  sending: false,
};

/* ================= session setup ================= */
function paneConfigRow(i, preset) {
  const row = document.createElement('div');
  row.className = 'pane-config';
  row.style.setProperty('--pane-color', PANE_COLORS[i % PANE_COLORS.length]);
  row.innerHTML = `
    <label class="field"><span class="label">Pane</span>
      <input class="p-name" value="${preset?.name || String.fromCharCode(65 + i)}"></label>
    <label class="field"><span class="label">Manipulated variable</span>
      <select class="p-var">${VARIABLES.map(([v, l]) =>
        `<option value="${v}"${preset?.variable === v ? ' selected' : ''}>${l}</option>`).join('')}
      </select></label>
    <label class="field"><span class="label">Value</span>
      <input class="p-val" value="${preset?.value ?? ''}" placeholder="—"></label>
    <button class="btn p-del" type="button" title="remove pane">×</button>`;
  // Display the legacy default in English and submit its original stored value.
  const valueInput = row.querySelector('.p-val');
  if (preset?.variable === 'prompt' && Object.hasOwn(PANE_VALUE_TEXT, preset?.value)) {
    valueInput.value = PANE_VALUE_TEXT[preset.value];
    valueInput.dataset.storedValue = preset.value;
    valueInput.dataset.displayValue = valueInput.value;
  }
  row.querySelector('.p-del').onclick = () => row.remove();
  return row;
}

function paneConfigValue(row) {
  const input = $('.p-val', row);
  return input.dataset.displayValue === input.value.trim()
    ? input.dataset.storedValue : input.value.trim();
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
  let ollamaOk = false;
  try {
    const resp = await fetch(`${API}/lab/models`);
    const data = await resp.json();
    ollamaOk = resp.ok && !data.error;
    state.models = (data.models || []).map(m => m.name);
  } catch { state.models = []; }
  $('#ollama-models').innerHTML = state.models.map(m => `<option value="${esc(m)}">`).join('');
  if (!$('#s-model').value && state.models.length) $('#s-model').value = state.models[0];
  const dot = $('#runtime-dot'), detail = $('#runtime-detail');
  if (dot && detail) {
    dot.classList.toggle('off', !ollamaOk);
    detail.textContent = ollamaOk
      ? `connected · ${state.models.length} model(s)`
      : 'offline · Transformers engine remains available';
  }
}

$('#s-engine').onchange = () => {
  const hf = $('#s-engine').value === 'transformers';
  $('#s-model').placeholder = hf ? 'e.g. Qwen/Qwen2.5-0.5B-Instruct' : 'Ollama model';
  if (hf) $('#s-model').value = '';
  else if (!$('#s-model').value && state.models.length) $('#s-model').value = state.models[0];
};

$('#btn-create-session').onclick = async () => {
  const base_config = { engine: $('#s-engine').value, model: $('#s-model').value.trim() };
  if (!base_config.model) { alert('Enter a model.'); return; }
  if ($('#s-temp').value !== '') base_config.temperature = parseFloat($('#s-temp').value);
  if ($('#s-seed').value !== '') base_config.seed = parseInt($('#s-seed').value, 10);
  if ($('#s-max-tokens').value !== '') {
    base_config.num_predict = parseInt($('#s-max-tokens').value, 10);
  }
  if ($('#s-system').value.trim()) base_config.system = $('#s-system').value.trim();
  const panes = [...$('#pane-configs').children].map(row => ({
    name: $('.p-name', row).value.trim() || '?',
    variable: $('.p-var', row).value,
    value: paneConfigValue(row),
  }));
  if (panes.length < 2) { alert('Create at least 2 panes to compare.'); return; }
  const body = {
    // Keep generated titles stable in storage; translate them when displayed.
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
    b.innerHTML = `<strong>${esc(sessionTitleText(s.title))}</strong><br>
      <span class="meta">${s.mode} · ${s.base_config.model || '?'} · ${s.created_at.slice(0, 16)}</span>`;
    b.onclick = async () => {
      const full = await (await fetch(`${API}/lab/sessions/${s.id}`)).json();
      openSession(full, full.requests || []);
    };
    list.appendChild(b);
  }
  $('#sessions-prev').hidden = !(data.sessions || []).length;
}

/* ================= active session ================= */
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
  $('#session-title').textContent = sessionTitleText(session.title);
  const mode = $('#session-mode');
  mode.textContent = session.mode === 'controlled' ? 'controlled experiment' : 'branched conversation';
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
    h.innerHTML = `<span class="pane-name">pane ${esc(pane.name)}</span>
      <span class="pane-cond ${pane.variable ? '' : 'base'}">${esc(conditionText(pane.condition))}</span>`;
    heads.appendChild(h);
  }
  $('#turns').innerHTML = '';
  renderPastTurns(pastRequests);
  buildComposerBoxes();
  // if any manipulated variable is the prompt itself, start unsynchronized
  // — each pane needs different text.
  $('#chk-sync').checked = !state.panes.some(p => p.variable === 'prompt');
}

// Match a request to its originating pane. branch_id = "pane-{idx}" is
// sent on EVERY submission (including controlled mode) for this — it is the only
// reliable way, since two panes can share a condition (two
// models compared, only one with a declared variable, etc.). condition is
// only a fallback for older records saved before this identifier.
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

/* ================= sending: one composer per pane ================= */
function buildComposerBoxes() {
  const grid = $('#composer-grid');
  grid.innerHTML = '';
  grid.style.gridTemplateColumns = `repeat(${state.panes.length}, 1fr)`;
  for (const pane of state.panes) {
    const field = document.createElement('div');
    field.className = 'field';
    field.style.setProperty('--pane-color', PANE_COLORS[pane.idx % PANE_COLORS.length]);
    field.innerHTML = `<span class="label" style="color:${PANE_COLORS[pane.idx % PANE_COLORS.length]}">
        pane ${esc(pane.name)}</span>
      <textarea class="composer-box" placeholder="Message for pane ${esc(pane.name)}…"></textarea>`;
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
    alert('Enter a message for every pane (or enable "synchronized messages").');
    return;
  }
  sendTurn(texts, rid(), 1);
  for (const b of boxes) b.value = '';
};

async function sendTurn(texts, turnId, replicate) {
  state.sending = true;
  $('#btn-send').disabled = true;
  $('#composer-status').textContent = 'generating…';
  const pairId = rid();
  // explicit seed per round (the same in all panes) when none is set:
  // ensures the run can be replayed (hook for Phase 3 / NNsight).
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
    } catch { /* the main request will display the connection error */ }
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
    // HTTP headers only accept ISO-8859-1; condition is free text (may contain
    // '≠', emoji, curly quotes...) so it is percent-encoded — the proxy decodes it.
    'X-Lombo-Condition': encodeURIComponent(condition),
    'X-Lombo-Replicate': String(meta.replicate),
  };
  // identify the originating pane independently of condition text — two
  // panes can have the SAME condition (e.g. two models, only one with a
  // declared variable), so matching only by condition links both to the
  // same record. branch_id also links history in branched mode.
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

/* after the round: fetch saved records (ids, metrics, ttft) by pair_id */
async function attachPairData(turn, pairId) {
  let requests = [];
  for (let attempt = 0; attempt < 12; attempt++) {
    await new Promise(r => setTimeout(r, 600));
    try {
      const data = await (await fetch(`${API}/compare?pair_id=${pairId}`)).json();
      requests = data.requests || [];
      if (requests.length >= state.panes.length) break;
    } catch { /* proxy busy; try again */ }
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

/* ================= Round DOM ================= */
function makeTurnDom(texts, resendInfo, replicate) {
  const turnEl = document.createElement('div');
  turnEl.className = 'turn';
  const meta = document.createElement('div');
  meta.className = 'turn-meta';
  meta.innerHTML = `<span class="label">round${replicate > 1 ? ` · replicate ${replicate}` : ''}</span>`;
  if (resendInfo && state.session.mode === 'controlled') {
    const btn = document.createElement('button');
    btn.className = 'btn';
    btn.style.marginLeft = '10px';
    btn.textContent = 'resend (new replicate)';
    let count = replicate;
    btn.onclick = () => { if (!state.sending) sendTurn(resendInfo.texts, resendInfo.turnId, ++count); };
    meta.appendChild(btn);
  }
  turnEl.appendChild(meta);

  const allSame = texts.every(t => t === texts[0]);
  if (allSame) {
    const user = document.createElement('div');
    user.className = 'user-msg';
    user.textContent = texts[0] || '(prompt unavailable)';
    turnEl.appendChild(user);
  } else {
    // messages differ across panes: one bubble per pane, rather than one
    // shared bubble — exactly the "same baseline, different prompts" case.
    const row = document.createElement('div');
    row.className = 'user-msgs';
    row.style.gridTemplateColumns = `repeat(${state.panes.length}, 1fr)`;
    state.panes.forEach((pane, i) => {
      const bubble = document.createElement('div');
      bubble.className = 'user-msg pane-msg';
      bubble.style.setProperty('--pane-color', PANE_COLORS[pane.idx % PANE_COLORS.length]);
      bubble.innerHTML = `<span class="pane-msg-name">${esc(pane.name)}</span>${esc(texts[i] || '(empty)')}`;
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
      chip.textContent = humanLabelText(label);
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
    detail.textContent = 'details';
    detail.onclick = () => openDrawer(info.id);
    footer.appendChild(detail);
  }
}

/* ================= word-by-word diff (LCS) ================= */
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
        (<del>only in ${esc(nameA)}</del> / <ins>only in ${esc(nameB)}</ins>)${
        state.panes.length > 2 ? ' — extra panes not included in the diff' : ''}</span>
      <button class="btn nn-exp-btn" type="button">View in Experiments →</button>
      <button class="btn nn-pair-btn" type="button">Open NNsight bundle (pair)</button>
    </div><div class="diff-body">${parts.join(' ')}</div>`;
  const expBtn = $('.nn-exp-btn', strip);
  if (expBtn) expBtn.onclick = () => { switchView('experiments'); window.loadExperimentPair?.(pairId); };
  const btn = $('.nn-pair-btn', strip);
  if (btn) btn.onclick = () => openNNsightBundle(pairId);
}

$('#btn-new-session').onclick = () => {
  $('#setup').hidden = false;
  $('#session-bar').hidden = true;
  $('#bench').hidden = true;
  $('#composer').hidden = true;
  loadSessionList();
};
