/* ================= visão geral (dados reais do lombo.db) ================= */
'use strict';

async function loadOverview() {
  let requests = [], sessions = [], timeseries = null, latency = null;
  try {
    const [reqData, sesData, tsData, latData] = await Promise.all([
      fetch(`${API}/requests?limit=200`).then(r => r.json()),
      fetch(`${API}/lab/sessions`).then(r => r.json()),
      fetch(`${API}/metrics/timeseries?bucket=hour&limit=24`).then(r => r.json()).catch(() => null),
      fetch(`${API}/metrics/latency?limit=500`).then(r => r.json()).catch(() => null),
    ]);
    requests = reqData.requests || [];
    sessions = sesData.sessions || [];
    timeseries = tsData;
    latency = latData;
  } catch { /* proxy fora do ar: mantém os estados vazios */ }
  renderOverviewFacts(requests, sessions, timeseries);
  renderOverviewSession(sessions[0], requests);
  renderOverviewTpsChart(requests);
  renderOverviewModels(requests);
  renderOverviewRuns(requests);
  renderOverviewSessions(sessions);
  renderOverviewLatency(latency);
  const count = $('#nav-count-runs');
  if (count) { count.hidden = !requests.length; count.textContent = requests.length >= 200 ? '200+' : String(requests.length); }
}

function renderOverviewFacts(requests, sessions, timeseries) {
  const latest = sessions[0];
  const model = latest?.base_config?.model || requests[0]?.model || null;
  $('#ov-model-name').textContent = model || 'nenhum modelo ainda';
  $('#ov-model-avatar').textContent = (model ? model.replace(/^.*\//, '') : '·').charAt(0).toUpperCase() || '·';
  const engineTag = $('#ov-model-engine');
  const engine = latest?.base_config?.engine;
  engineTag.hidden = !engine;
  if (engine) engineTag.textContent = engine;
  const avg = values => {
    const ok = values.filter(v => v != null && Number.isFinite(v));
    return ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : null;
  };
  const tps = avg(requests.map(r => r.tokens_per_second));
  const ttft = avg(requests.map(r => r.time_to_first_token_ms));
  $('#ov-fact-runs').textContent = requests.length ? (requests.length >= 200 ? '200+' : String(requests.length)) : '0';
  $('#ov-fact-sessions').textContent = String(sessions.length);
  $('#ov-fact-tps').textContent = tps != null ? tps.toFixed(1) : '—';
  $('#ov-fact-ttft').textContent = ttft != null ? `${ttft.toFixed(0)} ms` : '—';

  const points = timeseries?.points || [];
  const spark = (id, key, color) => {
    const el = $(id);
    if (!el) return;
    const values = points.map(p => p[key] ?? 0);
    if (values.filter(v => v > 0).length < 2) { LomboCharts.dispose(el); el.innerHTML = ''; return; }
    LomboCharts.sparkline(el, values, color);
  };
  const t = LomboCharts.tokens();
  spark('#ov-spark-runs', 'n', t.paneB);
  spark('#ov-spark-tps', 'avg_tps', t.accent);
  spark('#ov-spark-ttft', 'avg_ttft_ms', t.paneC);
}

function renderOverviewLatency(latency) {
  const wrap = $('#ov-latency-chart');
  const groups = (latency?.groups || []).filter(g => g.model && g.n > 0)
    .sort((a, b) => b.n - a.n).slice(0, 6);
  if (!groups.length) {
    LomboCharts.dispose(wrap);
    wrap.innerHTML = emptyPanel('Sem execuções bem-sucedidas registradas ainda.');
    return;
  }
  const categories = groups.map(g => g.model);
  LomboCharts.stackedBars(wrap, {
    categories,
    series: [
      { name: 'load', data: groups.map(g => g.avg_load_ms ?? 0), color: LomboCharts.tokens().paneC },
      { name: 'prompt eval', data: groups.map(g => g.avg_prompt_eval_ms ?? 0), color: LomboCharts.tokens().paneB },
      { name: 'eval', data: groups.map(g => g.avg_eval_ms ?? 0), color: LomboCharts.tokens().accent },
    ],
  }, { horizontal: true, yName: 'ms', left: 120, toolbox: false });
}

function renderOverviewSession(session, requests) {
  const strip = $('#ov-session-strip');
  if (!session) {
    strip.dataset.empty = 'true';
    $('#ov-open-session').textContent = 'Criar sessão →';
    return;
  }
  strip.dataset.empty = '';
  $('#ov-session-eyebrow').textContent =
    `SESSÃO MAIS RECENTE · ${session.mode === 'controlled' ? 'EXPERIMENTO CONTROLADO' : 'CONVERSA RAMIFICADA'}`;
  $('#ov-session-title').textContent = session.title;
  $('#ov-session-base').textContent = Object.entries(session.base_config || {})
    .map(([k, v]) => `${k}=${String(v).slice(0, 40)}`).join(' · ') || '—';
  const sessionRuns = requests.filter(r => r.lab_session_id === session.id);
  $('#ov-session-panes').textContent = String((session.panes || []).length || '—');
  $('#ov-session-runs').textContent = String(sessionRuns.length);
  $('#ov-session-last').textContent = timeAgo(sessionRuns[0]?.started_at || session.created_at);
  const btn = $('#ov-open-session');
  btn.textContent = 'Abrir estudo →';
  btn.removeAttribute('data-open-view');
  btn.onclick = () => openSessionById(session.id);
}

async function openSessionById(id) {
  const full = await (await fetch(`${API}/lab/sessions/${id}`)).json();
  if (full.id) { openSession(full, full.requests || []); switchView('lab'); }
}

/* Série real de tokens/s por execução (ordem cronológica), uma cor por modelo.
   Timeline ECharts: dataZoom + crosshair + exportação de PNG substituem o
   SVG desenhado à mão; o eixo de tempo real também deixa claro o espaçamento
   irregular entre execuções (algo que o índice sequencial escondia). */
function renderOverviewTpsChart(requests) {
  const wrap = $('#ov-tps-chart');
  const legend = $('#ov-tps-legend');
  const finding = $('#ov-finding');
  const points = requests.filter(r => r.tokens_per_second != null && r.started_at).reverse();
  if (points.length < 2) {
    LomboCharts.dispose(wrap);
    wrap.innerHTML = emptyPanel('Sem medições ainda. Rode prompts no Prompt lab ou no Benchmark para registrar tokens/s reais.');
    if (legend) legend.innerHTML = '';
    finding.hidden = true;
    return;
  }
  if (legend) legend.innerHTML = '';
  const byCount = new Map();
  for (const r of points) byCount.set(r.model, (byCount.get(r.model) || 0) + 1);
  const models = [...byCount.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4).map(([m]) => m);
  // sem `color` explícito: as fábricas de charts.js ciclam a paleta do tema
  // ([accent, paneB, paneC, paneD, ...]) na mesma ordem usada em todo o app.
  const series = models.map(model => ({
    name: model,
    data: points.filter(r => r.model === model)
      .map(r => [new Date(r.started_at).getTime(), r.tokens_per_second]),
  }));
  LomboCharts.timeline(wrap, { series }, { yName: 'tok/s' });
  const best = points.reduce((a, b) => b.tokens_per_second > a.tokens_per_second ? b : a);
  finding.hidden = false;
  $('#ov-finding-title').textContent =
    `Melhor throughput: ${best.tokens_per_second.toFixed(1)} tok/s (${best.model})`;
  $('#ov-finding-note').textContent =
    `${points.length} execuções com métricas · ${byCount.size} modelo(s) distintos no período carregado.`;
}

function renderOverviewModels(requests) {
  const list = $('#ov-models');
  const groups = new Map();
  for (const r of requests) {
    if (!r.model) continue;
    const g = groups.get(r.model) || { n: 0, tps: [] };
    g.n++;
    if (r.tokens_per_second != null) g.tps.push(r.tokens_per_second);
    groups.set(r.model, g);
  }
  const rows = [...groups.entries()].sort((a, b) => b[1].n - a[1].n).slice(0, 6);
  if (!rows.length) { list.innerHTML = emptyPanel('Nenhuma execução registrada.'); return; }
  const max = rows[0][1].n;
  list.innerHTML = rows.map(([model, g], i) => {
    const avg = g.tps.length ? (g.tps.reduce((a, b) => a + b, 0) / g.tps.length).toFixed(1) + ' tok/s' : '—';
    return `<li><span>${i + 1}</span><code title="${esc(model)}">${esc(model)}</code>` +
      `<i style="--v:${(g.n / max * 100).toFixed(1)}%"></i><b>${g.n}× · ${avg}</b></li>`;
  }).join('');
}

function renderOverviewRuns(requests) {
  const wrap = $('#ov-runs');
  const rows = requests.slice(0, 8);
  if (!rows.length) { wrap.innerHTML = emptyPanel('Nenhuma execução registrada ainda — tudo que passa pelo proxy aparece aqui.'); return; }
  wrap.innerHTML = rows.map(r => {
    const dot = r.error ? 'run-err' : 'run-ok';
    const metric = r.tokens_per_second != null ? `${r.tokens_per_second.toFixed(1)} tok/s`
      : r.error ? 'erro' : (r.status_code ?? '—');
    return `<div class="run-row" data-request="${esc(r.id)}" role="button" tabindex="0">
      <i class="${dot}"></i>
      <span><b>${esc(r.model || '?')}${r.condition ? ` · ${esc(r.condition)}` : ''}</b>
        <small>${esc(r.origin || '')} · ${esc(timeAgo(r.started_at))}</small></span>
      <code>${esc(String(metric))}</code></div>`;
  }).join('');
  wrap.querySelectorAll('[data-request]').forEach(row => {
    row.onclick = () => openDrawer(row.dataset.request);
  });
}

function renderOverviewSessions(sessions) {
  const wrap = $('#ov-sessions');
  const switcherTitle = $('#session-switcher-title');
  const switcherSub = $('#session-switcher-sub');
  const switcherGlyph = $('#session-switcher-glyph');
  const latest = sessions[0];
  if (latest) {
    switcherTitle.textContent = latest.title || 'sessão sem título';
    switcherSub.textContent = latest.mode === 'controlled' ? 'experimento controlado' : 'conversa ramificada';
    switcherGlyph.textContent = (latest.title || '?').charAt(0).toUpperCase();
    $('#session-switcher').onclick = () => openSessionById(latest.id);
  } else {
    $('#session-switcher').onclick = () => switchView('lab');
  }
  if (!sessions.length) { wrap.innerHTML = emptyPanel('Nenhuma sessão ainda.'); return; }
  wrap.innerHTML = sessions.slice(0, 5).map(s => `
    <div class="run-row" data-session="${esc(s.id)}" role="button" tabindex="0">
      <i class="run-ok"></i>
      <span><b>${esc(s.title)}</b><small>${s.mode === 'controlled' ? 'controlado' : 'ramificado'} · ${esc(s.base_config?.model || '?')}</small></span>
      <code>${esc(timeAgo(s.created_at))}</code></div>`).join('');
  wrap.querySelectorAll('[data-session]').forEach(row => {
    row.onclick = () => openSessionById(row.dataset.session);
  });
}
