/* ================= Benchmarks: quick-bench + comparative profile + history ================= */
'use strict';

$('#bench-run').onclick = async () => {
  const button = $('#bench-run');
  const engine = $('#bench-engine').value;
  const model = $('#bench-model').value.trim();
  const prompt = $('#bench-prompt').value.trim();
  if (!model || !prompt) { $('#bench-status').textContent = 'Enter a model and prompt.'; return; }
  button.disabled = true;
  $('#bench-status').textContent = engine === 'transformers'
    ? 'Downloading/loading the model (this may take a while on the first run)…'
    : 'Running…';
  $('#bench-result').innerHTML = '';
  const started = performance.now();
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Lombo-Engine': engine,
                'X-Lombo-Origin': 'benchmark', 'X-Lombo-Experiment': 'quick-benchmark'},
      body: JSON.stringify({model, messages: [{role: 'user', content: prompt}],
                            options: {temperature: 0, num_predict: 128}, stream: false}),
    });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
    const elapsed = (performance.now() - started) / 1000;
    const text = data.message?.content || '';
    $('#bench-status').textContent = `Completed in ${elapsed.toFixed(2)}s · ${data.eval_count ?? '—'} generated tokens`;
    $('#bench-result').innerHTML = `<div class="hf-detail-section"><h2>Actual response</h2><pre>${esc(text)}</pre></div>`;
    updateBenchmarkSummary({ elapsed, model, engine, evalCount: data.eval_count,
      tps: data.eval_count && data.eval_duration ? data.eval_count / (data.eval_duration / 1e9) : null });
    loadBenchmarksHistory();
  } catch (error) {
    $('#bench-status').textContent = `Failed: ${error.message}`;
  } finally { button.disabled = false; }
};

function updateBenchmarkSummary({ elapsed, model, engine, evalCount, tps }) {
  $('#bs-elapsed').textContent = elapsed.toFixed(2);
  $('#bs-tps').textContent = tps != null ? tps.toFixed(1) : '—';
  $('#bs-tokens').textContent = evalCount ?? '—';
  $('#bs-model').textContent = model;
  $('#bs-engine').textContent = engine;
}

window.loadBenchmarksHistory = async function loadBenchmarksHistory() {
  try {
    const [summary, evalSummary, history] = await Promise.all([
      fetch(`${API}/summary`).then(r => r.json()),
      fetch(`${API}/evaluations/summary`).then(r => r.json()),
      fetch(`${API}/requests?origin=benchmark&limit=200`).then(r => r.json()),
    ]);
    renderBenchmarkRadar(summary.groups || [], evalSummary.groups || []);
    renderBenchmarkBars(summary.groups || []);
    renderBenchmarkHistoryChart(history.requests || []);
  } catch {
    $('#bench-radar').innerHTML = emptyPanel('Proxy unavailable.');
  }
};

function renderBenchmarkRadar(summaryGroups, evalGroups) {
  const el = $('#bench-radar');
  const byModel = new Map();
  for (const g of summaryGroups) {
    if (!g.model) continue;
    const cur = byModel.get(g.model) || { n: 0, tps: [], count: 0 };
    cur.n += g.n; cur.count++;
    if (g.avg_tps != null) cur.tps.push(g.avg_tps);
    byModel.set(g.model, cur);
  }
  const evalByModel = new Map();
  for (const g of evalGroups) {
    if (!g.model) continue;
    const cur = evalByModel.get(g.model) || { repetition: [], truncated: [], words: [] };
    if (g.metrics?.repetition_rate) cur.repetition.push(g.metrics.repetition_rate.avg);
    if (g.metrics?.response_words) cur.words.push(g.metrics.response_words.avg);
    if (g.rates?.truncated != null) cur.truncated.push(g.rates.truncated);
    evalByModel.set(g.model, cur);
  }
  const top = [...byModel.entries()].sort((a, b) => b[1].n - a[1].n).slice(0, 3);
  if (top.length < 1) {
    LomboCharts.dispose(el); el.innerHTML = emptyPanel('Not enough runs to build a profile yet.');
    return;
  }
  const maxTps = Math.max(1, ...top.map(([, g]) => g.tps.length ? Math.max(...g.tps) : 0));
  const maxWords = Math.max(1, ...top.map(([m]) => (evalByModel.get(m)?.words || [0]).reduce((a, b) => a + b, 0) / (evalByModel.get(m)?.words.length || 1)));
  const indicators = [
    { name: 'tok/s', max: maxTps },
    { name: 'no repetition', max: 100 },
    { name: 'no truncation', max: 100 },
    { name: 'consistency', max: 100 },
  ];
  const avg = arr => (arr && arr.length) ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
  const series = top.map(([model, g]) => {
    const ev = evalByModel.get(model) || {};
    const tps = avg(g.tps) || 0;
    const repetition = avg(ev.repetition);
    const truncated = avg(ev.truncated);
    return {
      name: model,
      values: [
        tps,
        repetition != null ? Math.max(0, (1 - repetition) * 100) : 50,
        truncated != null ? Math.max(0, (1 - truncated) * 100) : 50,
        Math.min(100, (g.count / (g.count + 1)) * 100),
      ],
    };
  });
  LomboCharts.radar(el, { indicators, series }, { toolbox: false });
}

function renderBenchmarkBars(summaryGroups) {
  const el = $('#bench-table');
  const byModel = new Map();
  for (const g of summaryGroups) {
    if (!g.model || g.avg_tps == null) continue;
    const cur = byModel.get(g.model) || { tps: [], n: 0 };
    cur.tps.push(g.avg_tps); cur.n += g.n;
    byModel.set(g.model, cur);
  }
  const rows = [...byModel.entries()]
    .map(([model, g]) => ({ model, n: g.n, tps: g.tps.reduce((a, b) => a + b, 0) / g.tps.length }))
    .sort((a, b) => b.tps - a.tps).slice(0, 10);
  if (!rows.length) {
    LomboCharts.dispose(el); el.innerHTML = emptyPanel('No tok/s recorded yet.');
    return;
  }
  LomboCharts.bars(el, {
    categories: rows.map(r => r.model), series: [{ name: 'average tok/s', data: rows.map(r => r.tps) }],
  }, { horizontal: true, yName: 'tok/s', left: 140, labelWidth: 120 });
}

function renderBenchmarkHistoryChart(requests) {
  const el = $('#bench-history-chart');
  const points = requests.filter(r => r.tokens_per_second != null && r.started_at)
    .slice().reverse();
  if (points.length < 2) {
    LomboCharts.dispose(el);
    el.innerHTML = emptyPanel('Run a few tests above to see tok/s history per run.');
    return;
  }
  const byModel = new Map();
  for (const r of points) {
    if (!byModel.has(r.model)) byModel.set(r.model, []);
    byModel.get(r.model).push([new Date(r.started_at).getTime(), r.tokens_per_second]);
  }
  const series = [...byModel.entries()].slice(0, 4).map(([model, data]) => ({ name: model, data }));
  LomboCharts.timeline(el, { series }, { yName: 'tok/s' });
}
