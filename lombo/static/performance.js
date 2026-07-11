/* ================= Desempenho: latência, distribuições, atividade, VRAM, ambientes ================= */
'use strict';

async function loadPerformance(metricOverride) {
  const model = $('#perf-model').value;
  // metricOverride vem do clique direto no segmented control — não confiar na
  // classe .active aqui, pois o toggler genérico de core.js (registrado em
  // DOMContentLoaded, depois deste listener) só a atualiza DEPOIS deste código rodar.
  const metric = metricOverride || $('#perf-metric-toggle .active')?.dataset.metric || 'tokens_per_second';
  const modelQs = model ? `model=${encodeURIComponent(model)}` : '';
  try {
    const [latency, dist, timeseries, vram, envs, requests] = await Promise.all([
      fetch(`${API}/metrics/latency${modelQs ? '?' + modelQs : ''}`).then(r => r.json()),
      fetch(`${API}/metrics/distributions?metric=${metric}${modelQs ? '&' + modelQs : ''}`).then(r => r.json()),
      fetch(`${API}/metrics/timeseries?bucket=hour&limit=48${modelQs ? '&' + modelQs : ''}`).then(r => r.json()),
      fetch(`${API}/runtime/vram?limit=200`).then(r => r.json()),
      fetch(`${API}/environments`).then(r => r.json()),
      fetch(`${API}/requests?limit=200`).then(r => r.json()),
    ]);
    populatePerfModelSelect(requests.requests || []);
    renderPerfLatency(latency);
    renderPerfDistribution(dist, metric);
    renderPerfActivity(timeseries);
    renderPerfVram(vram);
    renderPerfEnvironments(envs);
  } catch {
    $('#perf-latency-chart').innerHTML = emptyPanel('Proxy indisponível.');
  }
}

function populatePerfModelSelect(requests) {
  const sel = $('#perf-model');
  if (sel.dataset.populated) return;
  const values = [...new Set(requests.map(r => r.model).filter(Boolean))].sort();
  sel.innerHTML = '<option value="">todos os modelos</option>' +
    values.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
  sel.dataset.populated = '1';
  sel.onchange = loadPerformance;
}

function renderPerfLatency(latency) {
  const el = $('#perf-latency-chart');
  const groups = (latency?.groups || []).filter(g => g.model && g.n > 0).sort((a, b) => b.n - a.n);
  if (!groups.length) {
    LomboCharts.dispose(el); el.innerHTML = emptyPanel('Sem execuções bem-sucedidas registradas ainda.');
    return;
  }
  const t = LomboCharts.tokens();
  LomboCharts.stackedBars(el, {
    categories: groups.map(g => g.model),
    series: [
      { name: 'load', data: groups.map(g => g.avg_load_ms ?? 0), color: t.paneC },
      { name: 'prompt eval', data: groups.map(g => g.avg_prompt_eval_ms ?? 0), color: t.paneB },
      { name: 'eval', data: groups.map(g => g.avg_eval_ms ?? 0), color: t.accent },
    ],
  }, { horizontal: true, yName: 'ms', left: 130 });
}

function renderPerfDistribution(dist, metric) {
  const el = $('#perf-dist-chart');
  const groups = (dist?.groups || []).filter(g => g.n >= 2);
  if (!groups.length) {
    LomboCharts.dispose(el);
    el.innerHTML = emptyPanel('Sem amostras suficientes para essa métrica ainda (mínimo 2 execuções por modelo).');
    return;
  }
  const unit = metric === 'time_to_first_token_ms' ? 'ms' : metric === 'eval_count' ? 'tokens' : 'tok/s';
  LomboCharts.violin(el, {
    categories: groups.map(g => g.model || '?'),
    distributions: groups.map(g => g.histogram),
  }, { yName: unit });
}

function renderPerfActivity(timeseries) {
  const el = $('#perf-activity-chart');
  const points = timeseries?.points || [];
  if (points.length < 2) {
    LomboCharts.dispose(el); el.innerHTML = emptyPanel('Sem atividade suficiente no período carregado.');
    return;
  }
  const t = LomboCharts.tokens();
  LomboCharts.mount(el, {
    animation: !window.matchMedia?.('(prefers-reduced-motion: reduce)').matches,
    animationDuration: 220,
    grid: { left: 46, right: 46, top: 28, bottom: 40, containLabel: true },
    legend: { top: 0, left: 0 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: points.map(p => p.t.replace('T', ' ') + 'h'),
            axisLabel: { interval: Math.max(0, Math.ceil(points.length / 8) - 1), fontFamily: t.mono, fontSize: 9 } },
    yAxis: [
      { type: 'value', name: 'execuções', axisLabel: { fontFamily: t.mono, fontSize: 9 } },
      { type: 'value', name: 'tok/s', position: 'right', axisLabel: { fontFamily: t.mono, fontSize: 9 } },
    ],
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 14, bottom: 4 }],
    series: [
      { name: 'execuções', type: 'bar', data: points.map(p => p.n), itemStyle: { color: t.paneB, opacity: .55 } },
      { name: 'erros', type: 'bar', stack: 'vol', data: points.map(p => p.errors), itemStyle: { color: t.err, opacity: .85 } },
      { name: 'tok/s médio', type: 'line', yAxisIndex: 1, data: points.map(p => p.avg_tps),
       itemStyle: { color: t.accent }, lineStyle: { color: t.accent, width: 2 }, smooth: true, showSymbol: false },
    ],
  }, { toolbox: false });
}

function renderPerfVram(vram) {
  const el = $('#perf-vram-chart');
  const points = vram?.points || [];
  if (points.length < 2) {
    LomboCharts.dispose(el);
    el.innerHTML = emptyPanel('Sem observações de VRAM ainda — aparecem quando o Ollama reporta /api/ps durante uma execução.');
    return;
  }
  const modelNames = [...new Set(points.flatMap(p => p.models.map(m => m.name)))].slice(0, 6);
  if (!modelNames.length) {
    LomboCharts.dispose(el); el.innerHTML = emptyPanel('Runtime observado sem modelos carregados nas amostras.');
    return;
  }
  const series = modelNames.map(name => ({
    name,
    data: points.map(p => {
      const found = p.models.find(m => m.name === name);
      return [new Date(p.observed_at).getTime(), found ? (found.size_vram || 0) / 1e9 : 0];
    }),
  }));
  LomboCharts.timeline(el, { series }, { yName: 'GB VRAM', stack: true });
}

function renderPerfEnvironments(envs) {
  const wrap = $('#perf-environments');
  const list = envs?.environments || [];
  if (!list.length) { wrap.innerHTML = emptyPanel('Nenhum ambiente registrado ainda.'); return; }
  wrap.innerHTML = list.map(e => `<div class="env-card">
      <header><b>${esc(e.os || 'SO desconhecido')}</b><span>${esc(e.hostname || '')}</span></header>
      <dl class="env-rows">
        <div><dt>Ollama</dt><dd>${esc(e.ollama_version || '—')}</dd></div>
        <div><dt>CPU</dt><dd>${esc(e.cpu || '—')}</dd></div>
        <div><dt>RAM</dt><dd>${e.ram_gb ? e.ram_gb.toFixed(1) + ' GB' : '—'}</dd></div>
        <div><dt>GPU</dt><dd>${(e.gpu || []).map(g => esc(g.name || JSON.stringify(g))).join(', ') || '—'}</dd></div>
        <div><dt>Execuções</dt><dd>${e.n_requests ?? 0}</dd></div>
        <div><dt>Período</dt><dd>${esc(timeAgo(e.first_seen))} → ${esc(timeAgo(e.last_seen))}</dd></div>
      </dl></div>`).join('');
}

$('#perf-metric-toggle').querySelectorAll('button').forEach(btn => {
  btn.addEventListener('click', () => loadPerformance(btn.dataset.metric));
});
