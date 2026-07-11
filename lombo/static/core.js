/* Núcleo do shell: helpers compartilhados, roteamento de views, boot.
   Outros arquivos (drawer.js, lab-session.js, overview.js, runs.js,
   diagnostics.js, huggingface.js) declaram funções no escopo global — como
   antes da divisão de lab.js — e podem chamar as daqui livremente, desde
   que só o façam depois do boot (DOMContentLoaded), quando todo script já
   carregou. */
'use strict';

const API = '/_lombo/api';
const $ = (sel, el) => (el || document).querySelector(sel);
const rid = () => crypto.randomUUID().replace(/-/g, '').slice(0, 12);

const PANE_COLORS = ['var(--pane-a)', 'var(--pane-b)', 'var(--pane-c)', 'var(--pane-d)'];

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g,
    ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}

function timeAgo(iso) {
  if (!iso) return '—';
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (!Number.isFinite(s)) return '—';
  if (s < 60) return 'agora há pouco';
  if (s < 3600) return `há ${Math.floor(s / 60)} min`;
  if (s < 86400) return `há ${Math.floor(s / 3600)} h`;
  return `há ${Math.floor(s / 86400)} d`;
}

const emptyPanel = text => `<div class="panel-empty">${esc(text)}</div>`;

/* ================= abas ================= */
const VIEW_NAMES = {
  overview: 'Visão geral', lab: 'Prompt lab', analysis: 'Execuções',
  experiments: 'Experimentos', performance: 'Desempenho',
  diagnostics: 'Camadas & logits', activations: 'Ativações',
  benchmarks: 'Benchmarks', huggingface: 'Modelos & datasets',
};

function switchView(name) {
  if (!VIEW_NAMES[name]) return;
  for (const v of Object.keys(VIEW_NAMES)) {
    $('#view-' + v).classList.toggle('active', v === name);
    $('#tab-' + v).setAttribute('aria-selected', String(v === name));
  }
  const viewName = $('#global-view-name');
  if (viewName) viewName.textContent = VIEW_NAMES[name];
  document.dispatchEvent(new CustomEvent('lombo:viewchange', { detail: { view: name } }));
  if (name === 'overview') loadOverview();
  if (name === 'analysis') loadRequests();
  if (name === 'experiments') loadExperiments();
  if (name === 'performance') loadPerformance();
  if (name === 'diagnostics') window.loadVisualDiagPairs?.();
  if (name === 'activations') window.loadActivationRequests?.();
  if (name === 'benchmarks') window.loadBenchmarksHistory?.();
  if (name === 'huggingface') window.loadHuggingFace?.();
}
window.selectHuggingFaceModel = (model, destination = 'lab') => {
  if (destination === 'benchmarks') {
    $('#bench-engine').value = 'transformers';
    $('#bench-model').value = model;
    switchView('benchmarks');
  } else {
    $('#s-engine').value = 'transformers';
    $('#s-model').value = model;
    $('#s-model').placeholder = 'ex.: Qwen/Qwen2.5-0.5B-Instruct';
    switchView('lab');
  }
};
for (const name of Object.keys(VIEW_NAMES)) {
  $('#tab-' + name).onclick = () => switchView(name);
}

/* ================= research workspace visuals ================= */
function initResearchWorkspace() {
  document.querySelectorAll('[data-open-view]').forEach(button => {
    button.addEventListener('click', () => switchView(button.dataset.openView));
  });
  document.querySelectorAll('.segmented').forEach(group => group.querySelectorAll('button').forEach(button => {
    button.addEventListener('click', () => {
      group.querySelector('.active')?.classList.remove('active'); button.classList.add('active');
    });
  }));
  document.addEventListener('keydown', event => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.target.matches('input, textarea, select') || event.target.isContentEditable) return;
    const shortcuts = { g: 'overview', p: 'lab', l: 'diagnostics', a: 'activations', b: 'benchmarks',
                        e: 'experiments', d: 'performance' };
    if (shortcuts[event.key.toLowerCase()]) switchView(shortcuts[event.key.toLowerCase()]);
  });
}

/* ================= boot =================
   Todos os arquivos de view carregam como <script> simples (mesmo escopo
   global desta divisão) na ordem definida em index.html; DOMContentLoaded só
   dispara depois que o último script do body terminou de executar, então
   por essa altura loadModels/initSetup/openSession/openDrawer etc. — vindos
   de lab-session.js, drawer.js, overview.js, runs.js — já existem. */
async function boot() {
  initResearchWorkspace();
  loadOverview();
  const [, labels] = await Promise.all([loadModels(),
    fetch(`${API}/labels`).then(r => r.json())]);
  state.labels = labels.labels || [];
  initSetup();
  loadSessionList();
  const params = new URLSearchParams(location.search);
  const sid = params.get('session');
  let sessionOpened = false;
  if (sid) {
    const full = await (await fetch(`${API}/lab/sessions/${sid}`)).json();
    if (full.id) { openSession(full, full.requests || []); sessionOpened = true; }
  }
  const pairId = params.get('pair');
  if (pairId && !params.get('view')) {
    switchView('experiments');
    window.loadExperimentPair?.(pairId);
  } else if (VIEW_NAMES[params.get('view')]) {
    switchView(params.get('view'));
    if (pairId) window.loadExperimentPair?.(pairId);
  } else if (sessionOpened) switchView('lab');
  const reqId = params.get('request');
  if (reqId) openDrawer(reqId);
}
document.addEventListener('DOMContentLoaded', boot);
