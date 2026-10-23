/* Lombo Lab — visual layer diagnostics. Native SVG, no chart dependency. */
'use strict';

(() => {
  const API_ROOT = '/_lombo/api';
  const q = (selector, root = document) => root.querySelector(selector);
  const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;',
  })[char]);
  const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value));
  const percent = value => value == null ? '—' : `${(value * 100).toFixed(value >= .1 ? 1 : 2)}%`;
  const number = (value, digits = 2) => value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits);
  const cssColor = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const COLORS = ['--pane-a', '--pane-b', '--pane-c', '--pane-d'];

  const diagnosticState = {
    entries: [], target: '', selectedLayer: 0, hoverLayer: null,
    playing: false, timer: null, speed: 900,
    visible: new Set(['target', 'confidence', 'entropy', 'delta']),
    metrics: null,
  };

  const layerCount = () => Math.max(0, ...diagnosticState.entries.map(entry => entry.diag.layers?.length || 0));
  const activeLayer = () => diagnosticState.hoverLayer ?? diagnosticState.selectedLayer;
  const layerAt = (entry, index = activeLayer()) => entry.diag.layers?.[Math.min(index, entry.diag.layers.length - 1)] || null;
  const leader = layer => layer?.top?.[0] || null;
  const targetValue = layer => layer?.target_prob ?? null;

  function candidateValue(layer, tokenId) {
    return layer?.top?.find(token => token.token_id === tokenId)?.prob ?? 0;
  }

  function getTrackedToken(entry) {
    if (entry.diag.target_token_id != null) {
      const found = entry.diag.layers.flatMap(layer => layer.top || [])
        .find(token => token.token_id === entry.diag.target_token_id);
      return { id: entry.diag.target_token_id, label: diagnosticState.target || found?.token || 'target token' };
    }
    const final = leader(entry.diag.layers.at(-1));
    return { id: final?.token_id ?? null, label: final?.token || 'final token' };
  }

  function trajectory(entry, key) {
    const tracked = getTrackedToken(entry);
    return entry.diag.layers.map(layer => {
      if (key === 'target') return targetValue(layer) ?? candidateValue(layer, tracked.id);
      if (key === 'confidence') return leader(layer)?.prob ?? null;
      if (key === 'entropy') return layer.entropy ?? null;
      if (key === 'delta') return layer.delta_norm ?? null;
      if (key === 'norm') return layer.hidden_norm ?? null;
      return null;
    });
  }

  function firstStableLayer(entry) {
    const layers = entry.diag.layers || [];
    const finalId = leader(layers.at(-1))?.token_id;
    if (finalId == null) return null;
    for (let index = 0; index < layers.length; index++) {
      if (layers.slice(index).every(layer => leader(layer)?.token_id === finalId)) return index;
    }
    return null;
  }

  function deriveMetrics() {
    const primary = diagnosticState.entries[0];
    if (!primary?.diag.layers?.length) return null;
    const values = trajectory(primary, 'target');
    const finalValue = values.at(-1) || 0;
    const threshold = Math.max(.001, finalValue * .1);
    const emergence = values.findIndex(value => value != null && value >= threshold);
    let jumpLayer = 0;
    for (let i = 1; i < values.length; i++) {
      if ((values[i] || 0) - (values[i - 1] || 0) > (values[jumpLayer] || 0) - (values[Math.max(0, jumpLayer - 1)] || 0)) jumpLayer = i;
    }
    const divergences = divergenceValues();
    const separation = divergences.length ? divergences.indexOf(Math.max(...divergences)) : null;
    return {
      emergence: emergence >= 0 ? emergence : null,
      jump: jumpLayer,
      stable: firstStableLayer(primary),
      separation,
      finalConfidence: leader(primary.diag.layers.at(-1))?.prob ?? null,
      finalEntropy: primary.diag.layers.at(-1)?.entropy ?? null,
    };
  }

  function divergenceValues() {
    if (diagnosticState.entries.length < 2) return [];
    const [a, b] = diagnosticState.entries;
    const length = Math.min(a.diag.layers.length, b.diag.layers.length);
    return Array.from({ length }, (_, index) => {
      const la = a.diag.layers[index], lb = b.diag.layers[index];
      if (la.target_prob != null && lb.target_prob != null) return Math.abs(la.target_prob - lb.target_prob);
      const ids = new Set((la.top || []).map(token => token.token_id));
      const overlap = (lb.top || []).filter(token => ids.has(token.token_id)).length;
      return 1 - overlap / Math.max(1, Math.max(la.top?.length || 0, lb.top?.length || 0));
    });
  }

  function tooltip(label, help) {
    return `<span class="help-tip" tabindex="0" aria-label="${escapeHtml(help)}">${escapeHtml(label)}<span role="tooltip">${escapeHtml(help)}</span></span>`;
  }

  function metricCard(label, value, help, tone = '') {
    return `<article class="diag-metric ${tone}">${tooltip(label, help)}<strong>${escapeHtml(value)}</strong></article>`;
  }

  function renderSummary() {
    const first = diagnosticState.entries[0];
    const model = first?.diag.model || {};
    const metrics = diagnosticState.metrics;
    const tracked = first ? getTrackedToken(first) : { label: '—' };
    return `<section class="diag-summary" aria-label="Run summary">
      <div class="run-context"><span class="eyebrow">Run completed</span><strong>${escapeHtml(model.name || model.model || model.architecture || 'Transformers model')}</strong><span>${diagnosticState.entries.map(entry => `<i style="--condition:${entry.color}"></i>${escapeHtml(conditionText(entry.name))}`).join(' · ')}</span></div>
      ${metricCard('Analyzed token', tracked.label, 'Supplied target token or, if absent, the leading token in the final layer.')}
      ${metricCard('Layers', String(first.diag.num_layers ?? first.diag.layers.length), 'Number of decoder blocks observed in the forward pass.')}
      ${metricCard('First observable emergence', metrics.emergence == null ? 'not observed' : `L${metrics.emergence}`, 'First point where the projected score reaches 10% of its final value, with a floor of 0.1%. This is a correlational heuristic, not causation.')}
      ${metricCard('Largest increase', `L${metrics.jump}`, 'Largest increase in projected score between two consecutive layers.')}
      ${metricCard('Largest separation', metrics.separation == null ? 'one condition' : `L${metrics.separation}`, 'Largest absolute target-token difference; without a target, uses 1 − top-k overlap.')}
      ${metricCard('Stable after', metrics.stable == null ? 'not stable' : `L${metrics.stable}`, 'First layer after which the final leading token continuously remains at rank 1.')}
      ${metricCard('Final confidence', percent(metrics.finalConfidence), 'Softmax probability of the rank-1 token in the final layer.')}
      ${metricCard('Final entropy', metrics.finalEntropy == null ? 'unavailable' : `${number(metrics.finalEntropy, 2)} nat`, 'Entropy over the full vocabulary. Older runs may not provide this metric.')}
    </section>`;
  }

  function chartCard(title, subtitle, body, extraClass = '') {
    return `<section class="chart-card ${extraClass}"><header><div><h2>${escapeHtml(title)}</h2><p>${escapeHtml(subtitle)}</p></div></header>${body}</section>`;
  }

  function linePath(values, x, y) {
    let path = '', open = false;
    values.forEach((value, index) => {
      if (value == null || !Number.isFinite(value)) { open = false; return; }
      path += `${open ? 'L' : 'M'}${x(index).toFixed(1)},${y(value).toFixed(1)} `;
      open = true;
    });
    return path;
  }

  function mainTrajectory() {
    const width = 1100, height = 330, left = 48, right = 20, top = 22, bottom = 42;
    const count = layerCount();
    const x = index => left + index * (width - left - right) / Math.max(1, count - 1);
    const available = {
      target: { label: 'Analyzed token', color: cssColor('--accent'), probability: true },
      confidence: { label: 'Confidence', color: cssColor('--ok'), probability: true },
      entropy: { label: 'Entropy', color: cssColor('--pane-c') },
      delta: { label: 'Delta residual', color: cssColor('--pane-a') },
    };
    const all = [];
    for (const [key, config] of Object.entries(available)) {
      diagnosticState.entries.forEach((entry, conditionIndex) => {
        const raw = trajectory(entry, key);
        if (!raw.some(value => value != null)) return;
        const max = config.probability ? 1 : Math.max(...raw.filter(Number.isFinite), 1e-9);
        all.push({ key, config, raw, values: raw.map(value => value == null ? null : value / max), entry, conditionIndex });
      });
    }
    const y = value => height - bottom - clamp(value) * (height - top - bottom);
    const grid = [0, .25, .5, .75, 1].map(value => `<line x1="${left}" y1="${y(value)}" x2="${width - right}" y2="${y(value)}"/><text x="${left - 9}" y="${y(value) + 4}" text-anchor="end">${Math.round(value * 100)}</text>`).join('');
    const paths = all.filter(series => diagnosticState.visible.has(series.key)).map(series => {
      const dash = series.conditionIndex ? '7 5' : (series.key === 'target' ? '' : '3 4');
      const color = series.key === 'target' ? series.entry.color : series.config.color;
      return `<path d="${linePath(series.values, x, y)}" stroke="${color}" stroke-dasharray="${dash}" class="series-path" data-series="${series.key}"/>`;
    }).join('');
    const markers = [
      ['emergence', diagnosticState.metrics.emergence, 'Emergence'],
      ['jump', diagnosticState.metrics.jump, 'Largest jump'],
      ['stable', diagnosticState.metrics.stable, 'Stable'],
      ['separation', diagnosticState.metrics.separation, 'Separation'],
    ].filter(([, index]) => index != null).map(([kind, index, label], markerIndex) => `<line x1="${x(index)}" y1="${top}" x2="${x(index)}" y2="${height - bottom}" class="event-line ${kind}"/><text x="${x(index) + 5}" y="${top + 13 + markerIndex * 13}" class="event-label">${label} · L${index}</text>`).join('');
    const selected = x(activeLayer());
    const overlays = Array.from({ length: count }, (_, index) => `<rect x="${x(index) - Math.max(5, (width - left - right) / Math.max(1, count - 1) / 2)}" y="${top}" width="${Math.max(10, (width - left - right) / Math.max(1, count - 1))}" height="${height - top - bottom}" class="layer-hit" data-layer="${index}" tabindex="0" aria-label="Layer ${index}"/>`).join('');
    const legend = Object.entries(available).map(([key, config]) => {
      const hasData = all.some(series => series.key === key);
      return `<label class="series-toggle ${hasData ? '' : 'unavailable'}"><input type="checkbox" data-toggle-series="${key}" ${diagnosticState.visible.has(key) && hasData ? 'checked' : ''} ${hasData ? '' : 'disabled'}><i style="--series:${config.color}"></i>${config.label}</label>`;
    }).join('');
    const svg = `<div class="chart-legend">${legend}</div><div class="chart-stage"><svg class="trajectory-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Normalized metric trajectories by layer"><g class="chart-grid">${grid}</g>${markers}${paths}<line x1="${selected}" y1="${top}" x2="${selected}" y2="${height - bottom}" class="selected-line" data-sync-left="${left}" data-sync-right="${right}" data-sync-width="${width}"/>${overlays}<text x="${left}" y="${height - 12}">L0</text><text x="${width - right}" y="${height - 12}" text-anchor="end">L${count - 1}</text></svg><div class="chart-hover" id="diag-hover" hidden></div></div>`;
    return chartCard('Trajectory by layer', 'Normalized series for joint viewing; hover shows actual values.', svg, 'trajectory-card');
  }

  function replayControls() {
    const max = Math.max(0, layerCount() - 1);
    const metrics = diagnosticState.metrics;
    return `<section class="replay-bar" aria-label="Replay by layer">
      <div><span class="eyebrow">Replay</span><strong>L<span id="replay-layer">${diagnosticState.selectedLayer}</span></strong></div>
      <button class="transport" data-replay="first" aria-label="First layer">|‹</button>
      <button class="transport" data-replay="previous" aria-label="Previous layer">‹</button>
      <button class="transport play" data-replay="play" aria-label="Play">${diagnosticState.playing ? 'Ⅱ' : '▶'}</button>
      <button class="transport" data-replay="next" aria-label="Next layer">›</button>
      <button class="transport" data-replay="last" aria-label="Last layer">›|</button>
      <input id="layer-slider" type="range" min="0" max="${max}" value="${diagnosticState.selectedLayer}" aria-label="Selected layer">
      <label class="speed-label">Speed <select id="replay-speed" aria-label="Replay speed"><option value="1400">0.5×</option><option value="900" ${diagnosticState.speed === 900 ? 'selected' : ''}>1×</option><option value="450">2×</option></select></label>
      ${metrics.separation == null ? '' : `<button class="btn" data-jump-layer="${metrics.separation}">Largest separation</button>`}
      ${metrics.stable == null ? '' : `<button class="btn" data-jump-layer="${metrics.stable}">Stabilization</button>`}
    </section>`;
  }

  function candidateColumns() {
    const primary = diagnosticState.entries[0];
    const scores = new Map();
    for (const layer of primary.diag.layers) for (const token of layer.top || []) {
      const current = scores.get(token.token_id) || { ...token, max: 0, count: 0 };
      current.max = Math.max(current.max, token.prob); current.count += 1; scores.set(token.token_id, current);
    }
    return [...scores.values()].sort((a, b) => (b.count - a.count) || (b.max - a.max)).slice(0, 8);
  }

  function heatmap() {
    const entry = diagnosticState.entries[0];
    const candidates = candidateColumns();
    if (!candidates.length) return chartCard('Candidate map', 'Layer × token', emptyState('Empty top-k', 'The run returned no candidates.'), 'heatmap-card');
    const cellW = 112, cellH = 30, left = 54, top = 46;
    const width = left + candidates.length * cellW + 8, height = top + entry.diag.layers.length * cellH + 8;
    const max = Math.max(...entry.diag.layers.flatMap(layer => layer.top || []).map(token => token.prob), 1e-9);
    const labels = candidates.map((token, index) => `<text x="${left + index * cellW + cellW / 2}" y="24" text-anchor="middle">${escapeHtml(token.token)}</text>`).join('');
    const rows = entry.diag.layers.map((layer, row) => {
      const cells = candidates.map((candidate, column) => {
        const token = layer.top?.find(item => item.token_id === candidate.token_id);
        const previous = row ? candidateValue(entry.diag.layers[row - 1], candidate.token_id) : null;
        const value = token?.prob || 0, rank = token ? layer.top.indexOf(token) + 1 : null;
        return `<rect x="${left + column * cellW + 1}" y="${top + row * cellH + 1}" width="${cellW - 2}" height="${cellH - 2}" rx="3" fill-opacity="${(.08 + clamp(value / max) * .88).toFixed(3)}" class="heat-cell" data-layer="${row}" data-token="${escapeHtml(candidate.token)}" data-value="${value}" data-rank="${rank ?? ''}" data-delta="${previous == null ? '' : value - previous}" tabindex="0"/>`;
      }).join('');
      return `<text x="${left - 10}" y="${top + row * cellH + 20}" text-anchor="end" class="layer-label ${activeLayer() === row ? 'active' : ''}">L${row}</text>${cells}`;
    }).join('');
    const svg = `<div class="heat-scroll"><svg class="heatmap-svg" viewBox="0 0 ${width} ${height}" style="min-width:${Math.min(width, 720)}px" role="img" aria-label="Candidate probability heatmap by layer">${labels}${rows}</svg></div>`;
    return chartCard('Candidate map', 'Intensity = top-k probability · click to select a layer', svg, 'heatmap-card');
  }

  function bumpChart() {
    const entry = diagnosticState.entries[0], candidates = candidateColumns().slice(0, 6);
    if (!candidates.length) return '';
    const width = 1100, height = 250, left = 50, right = 110, top = 25, bottom = 36;
    const count = entry.diag.layers.length;
    const x = index => left + index * (width - left - right) / Math.max(1, count - 1);
    const y = rank => top + (rank - 1) * (height - top - bottom) / Math.max(1, candidates.length - 1);
    const palette = [cssColor('--accent'), cssColor('--pane-a'), cssColor('--pane-b'), cssColor('--pane-c'), cssColor('--ok'), cssColor('--warn')];
    const lines = candidates.map((candidate, ci) => {
      const points = entry.diag.layers.map((layer, index) => {
        const rank = layer.top?.findIndex(token => token.token_id === candidate.token_id) ?? -1;
        return rank < 0 ? null : { index, rank: rank + 1 };
      });
      let path = '', open = false;
      for (const point of points) {
        if (!point) { open = false; continue; }
        path += `${open ? 'L' : 'M'}${x(point.index).toFixed(1)},${y(point.rank).toFixed(1)} `; open = true;
      }
      const last = [...points].reverse().find(Boolean);
      return `<path d="${path}" stroke="${palette[ci]}" class="bump-line"/><text x="${last ? x(last.index) + 8 : width - right}" y="${last ? y(last.rank) + 4 : top}" fill="${palette[ci]}">${escapeHtml(candidate.token)}</text>`;
    }).join('');
    const ranks = candidates.map((_, index) => `<text x="${left - 12}" y="${y(index + 1) + 4}" text-anchor="end">#${index + 1}</text><line x1="${left}" y1="${y(index + 1)}" x2="${width - right}" y2="${y(index + 1)}"/>`).join('');
    return chartCard('Candidate evolution', 'Top-k ranking; gaps indicate that the token left the returned candidates.', `<svg class="bump-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Ranking evolution of the leading candidates">${ranks}${lines}<line x1="${x(activeLayer())}" y1="${top}" x2="${x(activeLayer())}" y2="${height - bottom}" class="selected-line" data-sync-left="${left}" data-sync-right="${right}" data-sync-width="${width}"/></svg>`);
  }

  function layerDetails() {
    const index = activeLayer();
    const blocks = diagnosticState.entries.map(entry => {
      const layer = layerAt(entry, index);
      if (!layer) return '';
      const max = leader(layer)?.prob || 1;
      const bars = (layer.top || []).slice(0, 7).map((token, rank) => `<li><span class="rank">${rank + 1}</span><span class="token">${escapeHtml(token.token || '∅')}</span><span class="prob">${percent(token.prob)}</span><i style="width:${clamp(token.prob / max) * 100}%"></i></li>`).join('');
      return `<div class="condition-detail"><h3><i style="--condition:${entry.color}"></i>${escapeHtml(conditionText(entry.name))}</h3><ul class="token-bars">${bars || '<li class="muted">Empty top-k</li>'}</ul><dl class="layer-stats"><div><dt>Confidence</dt><dd>${percent(leader(layer)?.prob)}</dd></div><div><dt>Entropy</dt><dd>${layer.entropy == null ? '—' : `${number(layer.entropy)} nat`}</dd></div><div><dt>Norm</dt><dd>${number(layer.hidden_norm)}</dd></div><div><dt>Δ residual</dt><dd>${number(layer.delta_norm)}</dd></div></dl></div>`;
    }).join('');
    const divergence = divergenceValues()[index];
    return chartCard(`Layer details L${index}`, divergence == null ? 'Readout by condition' : `Difference between conditions: ${number(divergence, 4)}`, blocks, 'detail-card');
  }

  function comparison() {
    if (diagnosticState.entries.length < 2) return chartCard('Divergence between conditions', 'One condition available', emptyState('Comparison unavailable', 'Select a run with two conditions to see synchronized trajectories.'), 'comparison-card');
    const [a, b] = diagnosticState.entries, count = Math.min(a.diag.layers.length, b.diag.layers.length);
    const width = 1100, height = 330, left = 52, right = 20, top = 25, bottom = 38;
    const x = i => left + i * (width - left - right) / Math.max(1, count - 1);
    const tracks = [
      { key: 'target', label: 'Token', a: trajectory(a, 'target'), b: trajectory(b, 'target') },
      { key: 'confidence', label: 'Confidence', a: trajectory(a, 'confidence'), b: trajectory(b, 'confidence') },
      { key: 'entropy', label: 'Entropy', a: trajectory(a, 'entropy'), b: trajectory(b, 'entropy') },
      { key: 'norm', label: 'Norm', a: trajectory(a, 'norm'), b: trajectory(b, 'norm') },
      { key: 'delta', label: 'Δ residual', a: trajectory(a, 'delta'), b: trajectory(b, 'delta') },
    ].filter(track => [...track.a, ...track.b].some(value => value != null));
    const trackH = (height - top - bottom) / tracks.length;
    const content = tracks.map((track, trackIndex) => {
      const baseline = top + trackH * trackIndex, max = Math.max(...[...track.a, ...track.b].filter(Number.isFinite), 1e-9);
      const y = value => baseline + trackH - 12 - (value / max) * (trackH - 22);
      return `<rect x="${left}" y="${baseline}" width="${width - left - right}" height="${trackH}" class="track-bg"/><text x="${left - 8}" y="${baseline + 16}" text-anchor="end">${track.label}</text><path d="${linePath(track.a, x, y)}" stroke="${a.color}" class="compare-line"/><path d="${linePath(track.b, x, y)}" stroke="${b.color}" stroke-dasharray="6 4" class="compare-line"/>`;
    }).join('');
    const divergences = divergenceValues(), maxDiv = Math.max(...divergences, 1e-9);
    const bars = divergences.map((value, index) => `<rect x="${x(index) - 2}" y="${height - bottom - 8 * value / maxDiv}" width="4" height="${8 * value / maxDiv}" class="divergence-tick"/>`).join('');
    const legend = `<div class="condition-legend"><span><i style="--condition:${a.color}"></i>${escapeHtml(conditionText(a.name))}</span><span><i style="--condition:${b.color}"></i>${escapeHtml(conditionText(b.name))} · dashed</span><span class="muted">bottom marks = divergence</span></div>`;
    return chartCard('Divergence between conditions', 'Same axes per metric; values scaled within each track.', `${legend}<svg class="comparison-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Synchronized comparison between the two conditions">${content}${bars}<line x1="${x(activeLayer())}" y1="${top}" x2="${x(activeLayer())}" y2="${height - bottom}" class="selected-line" data-sync-left="${left}" data-sync-right="${right}" data-sync-width="${width}"/></svg>`, 'comparison-card');
  }

  function technicalData() {
    return `<details class="technical-data"><summary>Technical data <span>payload, shapes and dtype</span></summary><div class="technical-head"><p>Raw endpoint data. No activation is sent to the browser.</p><button class="btn" id="copy-diag-json">Copy JSON</button></div><pre>${escapeHtml(JSON.stringify(diagnosticState.entries.map(entry => ({ condition: entry.name, ...entry.diag })), null, 2))}</pre></details>`;
  }

  function emptyState(title, message, loading = false) {
    return `<div class="diag-empty ${loading ? 'loading' : ''}" role="status"><i aria-hidden="true"></i><strong>${escapeHtml(title)}</strong><span>${escapeHtml(message)}</span></div>`;
  }

  function renderDashboard({ preserveScroll = false } = {}) {
    const container = q('#diag-results');
    if (!container || !diagnosticState.entries.length) return;
    diagnosticState.metrics = deriveMetrics();
    diagnosticState.selectedLayer = clamp(diagnosticState.selectedLayer, 0, layerCount() - 1);
    const scroll = preserveScroll ? container.scrollTop : 0;
    container.innerHTML = `${renderSummary()}${mainTrajectory()}${replayControls()}<div class="diag-two-col">${heatmap()}${layerDetails()}</div>${bumpChart()}${comparison()}${technicalData()}`;
    bindDashboardEvents();
    if (preserveScroll) container.scrollTop = scroll;
  }

  function updateSelection(layer, commit = true) {
    const next = clamp(Number(layer), 0, layerCount() - 1);
    if (commit) { diagnosticState.selectedLayer = next; diagnosticState.hoverLayer = null; }
    else diagnosticState.hoverLayer = next;
    renderDashboard({ preserveScroll: true });
  }

  function showChartTooltip(event, layer) {
    const tip = q('#diag-hover');
    if (!tip) return;
    const rows = diagnosticState.entries.map(entry => {
      const current = layerAt(entry, layer);
      return `<b><i style="--condition:${entry.color}"></i>${escapeHtml(conditionText(entry.name))}</b><span>${escapeHtml(leader(current)?.token || '—')} · ${percent(leader(current)?.prob)}</span><span>target ${percent(targetValue(current))}</span><span>H ${current?.entropy == null ? '—' : number(current.entropy)} · Δ ${number(current?.delta_norm)}</span>`;
    }).join('');
    tip.innerHTML = `<strong>Layer ${layer}</strong>${rows}`;
    tip.hidden = false;
    const stage = tip.parentElement.getBoundingClientRect();
    tip.style.left = `${clamp(event.clientX - stage.left + 12, 8, stage.width - 245)}px`;
    tip.style.top = `${clamp(event.clientY - stage.top + 12, 8, stage.height - 145)}px`;
  }

  function syncLayerPreview(layer) {
    const index = clamp(Number(layer), 0, layerCount() - 1);
    q('#replay-layer')?.replaceChildren(document.createTextNode(String(index)));
    if (q('#layer-slider')) q('#layer-slider').value = index;
    document.querySelectorAll('.layer-label').forEach(label => label.classList.toggle('active', label.textContent === `L${index}`));
    document.querySelectorAll('.selected-line[data-sync-width]').forEach(line => {
      const width = Number(line.dataset.syncWidth), left = Number(line.dataset.syncLeft), right = Number(line.dataset.syncRight);
      const position = left + index * (width - left - right) / Math.max(1, layerCount() - 1);
      line.setAttribute('x1', position); line.setAttribute('x2', position);
    });
    const currentDetail = q('.detail-card');
    if (currentDetail) currentDetail.outerHTML = layerDetails();
  }

  function stopReplay() {
    diagnosticState.playing = false;
    if (diagnosticState.timer) clearInterval(diagnosticState.timer);
    diagnosticState.timer = null;
  }

  function startReplay() {
    stopReplay(); diagnosticState.playing = true;
    diagnosticState.timer = setInterval(() => {
      if (diagnosticState.selectedLayer >= layerCount() - 1) diagnosticState.selectedLayer = 0;
      else diagnosticState.selectedLayer += 1;
      renderDashboard({ preserveScroll: true });
    }, diagnosticState.speed);
    renderDashboard({ preserveScroll: true });
  }

  function bindDashboardEvents() {
    q('#layer-slider')?.addEventListener('input', event => updateSelection(event.target.value));
    q('#replay-speed')?.addEventListener('change', event => {
      diagnosticState.speed = Number(event.target.value);
      if (diagnosticState.playing) startReplay();
    });
    document.querySelectorAll('[data-replay]').forEach(button => button.addEventListener('click', () => {
      const action = button.dataset.replay;
      if (action === 'play') { diagnosticState.playing ? (stopReplay(), renderDashboard({ preserveScroll: true })) : startReplay(); return; }
      stopReplay();
      const positions = { first: 0, previous: diagnosticState.selectedLayer - 1, next: diagnosticState.selectedLayer + 1, last: layerCount() - 1 };
      updateSelection(positions[action]);
    }));
    document.querySelectorAll('[data-jump-layer]').forEach(button => button.addEventListener('click', () => updateSelection(button.dataset.jumpLayer)));
    document.querySelectorAll('[data-toggle-series]').forEach(input => input.addEventListener('change', () => {
      input.checked ? diagnosticState.visible.add(input.dataset.toggleSeries) : diagnosticState.visible.delete(input.dataset.toggleSeries);
      renderDashboard({ preserveScroll: true });
    }));
    document.querySelectorAll('.layer-hit').forEach(hit => {
      hit.addEventListener('mousemove', event => showChartTooltip(event, Number(hit.dataset.layer)));
      hit.addEventListener('mouseenter', () => { diagnosticState.hoverLayer = Number(hit.dataset.layer); syncLayerPreview(hit.dataset.layer); });
      hit.addEventListener('mouseleave', () => { diagnosticState.hoverLayer = null; syncLayerPreview(diagnosticState.selectedLayer); q('#diag-hover').hidden = true; });
      hit.addEventListener('click', () => updateSelection(hit.dataset.layer));
      hit.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') updateSelection(hit.dataset.layer); });
    });
    document.querySelectorAll('.heat-cell').forEach(cell => {
      const describe = () => cell.setAttribute('aria-label', `Layer ${cell.dataset.layer}, token ${cell.dataset.token}, score ${percent(Number(cell.dataset.value))}, rank ${cell.dataset.rank || 'outside top-k'}, change ${cell.dataset.delta === '' ? 'unavailable' : percent(Number(cell.dataset.delta))}`);
      describe();
      cell.addEventListener('click', () => updateSelection(cell.dataset.layer));
      cell.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') updateSelection(cell.dataset.layer); });
    });
    q('#copy-diag-json')?.addEventListener('click', async event => {
      await navigator.clipboard.writeText(JSON.stringify(diagnosticState.entries.map(entry => ({ condition: entry.name, ...entry.diag })), null, 2));
      event.target.textContent = 'Copied'; setTimeout(() => { event.target.textContent = 'Copy JSON'; }, 1200);
    });
  }

  async function loadPairs() {
    const select = q('#diag-pair'), status = q('#diag-status');
    if (!select) return;
    const previous = select.value;
    status.textContent = 'Refreshing runs…';
    try {
      const response = await fetch(`${API_ROOT}/requests?limit=200`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json(), pairs = new Map();
      for (const request of data.requests || []) {
        if (!request.pair_id) continue;
        const group = pairs.get(request.pair_id) || { id: request.pair_id, model: request.model, when: request.started_at, conditions: [] };
        if (request.condition && !group.conditions.includes(request.condition)) group.conditions.push(request.condition);
        pairs.set(request.pair_id, group);
      }
      select.innerHTML = '<option value="">— select a run —</option>' + [...pairs.values()].map(group => `<option value="${escapeHtml(group.id)}">${escapeHtml(`${(group.when || '').slice(5, 16).replace('T', ' ')} · ${group.model || '?'} · ${group.conditions.map(conditionText).join(' × ') || group.id.slice(0, 8)}`)}</option>`).join('');
      if (previous) select.value = previous;
      status.textContent = pairs.size ? `${pairs.size} available runs` : 'No comparable runs available';
      if (!pairs.size) q('#diag-results').innerHTML = emptyState('Diagnostics unavailable', 'Create a run with the Transformers engine to enable per-layer inspection.');
    } catch (error) {
      status.textContent = 'Failed to load runs';
      q('#diag-results').innerHTML = emptyState('API error', error.message);
    }
  }

  async function run() {
    const pairId = q('#diag-pair').value, target = q('#diag-target').value.trim();
    const status = q('#diag-status'), container = q('#diag-results'), button = q('#btn-diagnose');
    stopReplay();
    if (!pairId) { status.textContent = 'Select a run.'; q('#diag-pair').focus(); return; }
    button.disabled = true; status.textContent = 'Loading diagnostics…';
    container.innerHTML = emptyState('Reading representations', 'The model is running one forward pass per condition.', true);
    try {
      const comparisonResponse = await fetch(`${API_ROOT}/compare?pair_id=${encodeURIComponent(pairId)}`);
      if (!comparisonResponse.ok) throw new Error(`Comparison unavailable (HTTP ${comparisonResponse.status})`);
      const comparisonData = await comparisonResponse.json();
      const requests = comparisonData.requests || [];
      if (!requests.length) throw new Error('The run contains no conditions that support diagnostics.');
      const settled = await Promise.allSettled(requests.map(async (request, index) => {
        const response = await fetch(`${API_ROOT}/requests/${request.id}/diagnostics`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ target_token: target || null, top_k: 10 }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
        return { diag: payload, color: cssColor(COLORS[index % COLORS.length]), name: request.condition || `condition ${index + 1}` };
      }));
      const entries = settled.filter(result => result.status === 'fulfilled').map(result => result.value);
      const failures = settled.filter(result => result.status === 'rejected');
      if (!entries.length) throw failures[0].reason;
      if (!entries.some(entry => entry.diag.layers?.length)) throw new Error('The endpoint returned a run without layers.');
      diagnosticState.entries = entries; diagnosticState.target = target;
      diagnosticState.selectedLayer = 0; diagnosticState.hoverLayer = null;
      status.textContent = failures.length ? `${entries.length} condition available · ${failures.length} unavailable` : `${entries.length} condition(s) · ${layerCount()} layers`;
      renderDashboard();
    } catch (error) {
      status.textContent = 'Diagnostics unavailable';
      container.innerHTML = emptyState('Could not run', error.message || String(error));
    } finally { button.disabled = false; }
  }

  q('#btn-diag-help')?.addEventListener('click', event => {
    const method = q('#diag-method'); method.hidden = !method.hidden;
    event.currentTarget.setAttribute('aria-expanded', String(!method.hidden));
  });
  q('#btn-diagnose').onclick = run;
  q('#btn-diag-reload').onclick = loadPairs;
  q('#diag-target')?.addEventListener('keydown', event => { if (event.key === 'Enter') run(); });
  // Navigating to another tab pauses replay; the router fires this
  // event on each view change (including tabs created after this one).
  document.addEventListener('lombo:viewchange', event => {
    if (event.detail.view !== 'diagnostics') stopReplay();
  });

  // Entry point consumed by the existing tab router.
  window.loadVisualDiagPairs = loadPairs;
})();
