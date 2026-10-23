/* Hugging Face Hub browser and model execution selector. */
'use strict';

(() => {
  const service = window.HuggingFaceService;
  const hfState = { type: 'models', loaded: false, loading: false, request: 0, nextCursor: null };
  const $hf = selector => document.querySelector(selector);

  window.loadHuggingFace = () => {
    if (!hfState.loaded) loadResults();
  };

  function setType(type) {
    hfState.type = type;
    hfState.nextCursor = null;
    for (const value of ['models', 'datasets']) {
      const tab = $hf(`#hf-tab-${value}`);
      tab.classList.toggle('active', value === type);
      tab.setAttribute('aria-selected', String(value === type));
      $hf(`#hf-${value === 'models' ? 'model' : 'dataset'}-filters`).hidden = value !== type;
    }
    loadResults();
  }

  async function loadResults({ append = false } = {}) {
    if (hfState.loading && append) return;
    const requestId = ++hfState.request;
    hfState.loading = true;
    const grid = $hf('#hf-results');
    const form = activeForm();
    const filters = Object.fromEntries(new FormData(form).entries());
    if (append && hfState.nextCursor) filters.cursor = hfState.nextCursor;
    if (append) {
      const button = $hf('#hf-load-more');
      if (button) { button.disabled = true; button.textContent = 'Loading…'; }
    } else {
      hfState.nextCursor = null;
      grid.innerHTML = skeletons(8);
      $hf('#hf-result-count').textContent = 'Querying Hugging Face Hub…';
    }
    try {
      const data = hfState.type === 'models'
        ? await service.listModels(filters)
        : await service.listDatasets(filters);
      if (requestId !== hfState.request) return;
      const items = data.items || [];
      hfState.nextCursor = data.next_cursor || null;
      hfState.loaded = true;
      if (append) {
        $hf('.hf-load-more')?.remove();
        grid.insertAdjacentHTML('beforeend', items.map(itemCard).join(''));
      } else {
        grid.innerHTML = items.length ? items.map(itemCard).join('') : emptyState();
      }
      bindCards();
      if (hfState.nextCursor) {
        grid.insertAdjacentHTML('beforeend', '<div class="hf-load-more"><button class="btn" id="hf-load-more" type="button">Load more</button></div>');
        $hf('#hf-load-more').onclick = () => loadResults({ append: true });
      }
      const loaded = grid.querySelectorAll('.hf-card').length;
      $hf('#hf-result-count').textContent = `${fmtNumber(loaded)} ${hfState.type === 'models' ? 'models' : 'datasets'} loaded`;
    } catch (error) {
      if (requestId !== hfState.request) return;
      if (append) {
        const button = $hf('#hf-load-more');
        if (button) { button.disabled = false; button.textContent = 'Try loading again'; }
      } else {
        $hf('#hf-result-count').textContent = 'Hub unavailable';
        grid.innerHTML = `<div class="hf-empty"><strong>Could not load resources.</strong><span>${html(error.message)}</span><button class="btn" id="hf-retry">Try again</button></div>`;
        $hf('#hf-retry').onclick = () => loadResults();
      }
    } finally {
      if (requestId === hfState.request) hfState.loading = false;
    }
  }

  function bindCards() {
    $hf('#hf-results').querySelectorAll('.hf-card:not([data-bound])').forEach(card => {
      card.dataset.bound = 'true';
      card.onclick = () => openDetail(card.dataset.id);
    });
  }

  function activeForm() {
    return $hf(hfState.type === 'models' ? '#hf-model-filters' : '#hf-dataset-filters');
  }

  function itemCard(item) {
    const isModel = hfState.type === 'models';
    const tags = (item.tags || []).filter(tag => !tag.includes(':')).slice(0, 4);
    const access = item.gated ? 'Gated' : 'Public';
    const accessClass = item.gated ? 'gated' : 'public';
    const secondary = isModel
      ? `<span>${icon('branch')} ${html(item.pipeline || 'pipeline not provided')}</span><span>${html(item.architecture || 'architecture not provided')}</span>`
      : `<span>${html(joinOr(item.modalities, 'modality not provided'))}</span><span>${html(joinOr(item.tasks, 'task not provided'))}</span>`;
    return `<button class="hf-card" data-id="${attr(item.id)}" type="button">
      <div class="hf-card-top"><span class="hf-resource-icon">${isModel ? 'M' : 'D'}</span><div class="hf-card-name"><strong>${html(item.id)}</strong><span>${html(item.author || 'author not provided')}</span></div><span class="hf-access ${accessClass}">${access}</span></div>
      <div class="hf-card-meta"><span title="Downloads">${icon('download')} ${fmtNumber(item.downloads)}</span><span title="Likes">${icon('heart')} ${fmtNumber(item.likes)}</span><span>${html(item.license || 'license —')}</span>${!isModel && item.size ? `<span>${fmtBytes(item.size)}</span>` : ''}</div>
      <div class="hf-card-secondary">${secondary}</div>
      <div class="hf-tags">${tags.map(tag => `<span>${html(tag)}</span>`).join('') || '<span>no tags</span>'}</div>
      <div class="hf-card-foot"><span>Updated ${fmtDate(item.last_modified)}</span><span>View details →</span></div>
    </button>`;
  }

  async function openDetail(id) {
    const drawer = $hf('#hf-drawer');
    const backdrop = $hf('#hf-drawer-backdrop');
    drawer.classList.add('open');
    drawer.setAttribute('aria-hidden', 'false');
    backdrop.hidden = false;
    $hf('#hf-detail-type').textContent = hfState.type === 'models' ? 'Model' : 'Dataset';
    $hf('#hf-detail-name').textContent = id;
    $hf('#hf-detail-body').innerHTML = `<div class="hf-detail-loading">${skeletons(3)}</div>`;
    try {
      const data = hfState.type === 'models' ? await service.getModel(id) : await service.getDataset(id);
      $hf('#hf-detail-body').innerHTML = hfState.type === 'models' ? modelDetail(data) : datasetDetail(data);
      if (hfState.type === 'models') bindModelActions(data.id);
    } catch (error) {
      $hf('#hf-detail-body').innerHTML = `<div class="hf-empty"><strong>Details unavailable.</strong><span>${html(error.message)}</span></div>`;
    }
  }

  function modelDetail(item) {
    return `<div class="hf-detail-summary">
      <div class="hf-detail-stats"><span><b>${fmtNumber(item.downloads)}</b> downloads</span><span><b>${fmtNumber(item.likes)}</b> likes</span><span><b>${html(item.pipeline || '—')}</b> pipeline</span><span><b>${html(item.license || '—')}</b> license</span></div>
      ${detailRows([['Author', item.author], ['Access', item.gated ? 'Gated' : 'Public'], ['Architecture', item.architecture], ['Context length', item.context_length ? `${fmtNumber(item.context_length)} tokens` : null], ['Last updated', fmtFullDate(item.last_modified)]])}
      <div class="hf-actions"><button class="btn primary" id="hf-open-bench" type="button">Test in Bench</button><button class="btn" id="hf-open-lab" type="button">Open in Lab</button></div>
      <a class="hf-link" href="${attr(item.official_url)}" target="_blank" rel="noopener noreferrer">Open official page on Hugging Face ↗</a>
    </div>
    ${tagsSection(item.tags)}
    ${section('Model Card', item.card ? `<pre class="hf-card-markdown">${html(item.card)}</pre>` : unavailable('Model Card unavailable.'))}
    ${section(`Files (${(item.files || []).length})`, filesTable(item.files))}
    ${section('Configuration', item.config && Object.keys(item.config).length ? `<pre>${html(JSON.stringify(item.config, null, 2))}</pre>` : unavailable('config.json unavailable.'))}`;
  }

  function bindModelActions(id) {
    $hf('#hf-open-bench').onclick = () => { closeDetail(); window.selectHuggingFaceModel(id, 'benchmarks'); };
    $hf('#hf-open-lab').onclick = () => { closeDetail(); window.selectHuggingFaceModel(id, 'lab'); };
  }

  function datasetDetail(item) {
    const splits = item.splits || [];
    return `<div class="hf-detail-summary">
      <div class="hf-detail-stats"><span><b>${fmtNumber(item.downloads)}</b> downloads</span><span><b>${fmtNumber(item.likes)}</b> likes</span><span><b>${item.size ? fmtBytes(item.size) : '—'}</b> size</span></div>
      ${detailRows([['Author', item.author], ['Modality', joinOr(item.modalities, null)], ['Task', joinOr(item.tasks, null)], ['License', item.license], ['Last updated', fmtFullDate(item.last_modified)]])}
      <div class="hf-actions"><button class="btn primary" disabled>Run Benchmark</button></div>
      <a class="hf-link" href="${attr(item.official_url)}" target="_blank" rel="noopener noreferrer">Open official page on Hugging Face ↗</a>
    </div>
    ${item.description ? section('Description', `<p class="hf-description">${html(item.description)}</p>`) : ''}
    ${tagsSection(item.tags)}
    ${section('Dataset Card', item.card ? `<pre class="hf-card-markdown">${html(item.card)}</pre>` : unavailable('Dataset Card unavailable.'))}
    ${section(`Splits (${splits.length})`, splits.length ? `<div class="hf-split-list">${splits.map(split => `<div><strong>${html(split.config)} / ${html(split.name)}</strong><span>${fmtNumber(split.num_examples)} examples · ${fmtBytes(split.num_bytes)}</span></div>`).join('')}</div>` : unavailable('Splits not provided by Dataset Viewer.'))}
    ${section('Features', item.features && Object.keys(item.features).length ? `<pre>${html(JSON.stringify(item.features, null, 2))}</pre>` : unavailable('Features not provided.'))}`;
  }

  function detailRows(rows) {
    return `<dl class="hf-detail-rows">${rows.filter(([, value]) => value).map(([key, value]) => `<div><dt>${html(key)}</dt><dd>${html(value)}</dd></div>`).join('')}</dl>`;
  }

  function tagsSection(tags) {
    return section('Tags', `<div class="hf-detail-tags">${(tags || []).map(tag => `<span>${html(tag)}</span>`).join('') || '<span>Not provided</span>'}</div>`);
  }

  function section(title, content) {
    return `<section class="hf-detail-section"><h2>${html(title)}</h2>${content}</section>`;
  }

  function filesTable(files) {
    if (!(files || []).length) return unavailable('File list unavailable.');
    return `<div class="hf-files">${files.map(file => `<div><code>${html(file.name)}</code><span>${file.size ? fmtBytes(file.size) : '—'}</span></div>`).join('')}</div>`;
  }

  function unavailable(text) { return `<p class="hf-unavailable">${html(text)}</p>`; }
  function joinOr(values, fallback) { return (values || []).filter(Boolean).join(', ') || fallback; }
  function fmtNumber(value) { return value === null || value === undefined ? '—' : new Intl.NumberFormat('en-US', { notation: value >= 10000 ? 'compact' : 'standard', maximumFractionDigits: 1 }).format(value); }
  function fmtBytes(value) {
    if (value === null || value === undefined) return '—';
    const units = ['B', 'KB', 'MB', 'GB', 'TB']; let number = Number(value), unit = 0;
    while (number >= 1024 && unit < units.length - 1) { number /= 1024; unit++; }
    return `${new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 }).format(number)} ${units[unit]}`;
  }
  function fmtDate(value) { return value ? new Intl.DateTimeFormat('en-US', { dateStyle: 'medium' }).format(new Date(value)) : '—'; }
  function fmtFullDate(value) { return value ? new Intl.DateTimeFormat('en-US', { dateStyle: 'long', timeStyle: 'short' }).format(new Date(value)) : null; }
  function html(value) { return String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char])); }
  function attr(value) { return html(value); }
  function icon(name) {
    if (name === 'heart') return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1.1-1.1a5.5 5.5 0 0 0-7.8 7.8l1.1 1.1L12 21l7.8-7.5 1.1-1.1a5.5 5.5 0 0 0-.1-7.8Z"/></svg>';
    if (name === 'branch') return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="6" cy="5" r="2"/><circle cx="18" cy="6" r="2"/><circle cx="6" cy="19" r="2"/><path d="M6 7v10M8 11h4a6 6 0 0 0 6-3"/></svg>';
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 3v12m0 0 5-5m-5 5-5-5M5 20h14"/></svg>';
  }
  function skeletons(count) { return Array.from({ length: count }, () => '<div class="hf-skeleton"><i></i><i></i><i></i></div>').join(''); }
  function emptyState() { return '<div class="hf-empty"><strong>No resources found.</strong><span>Adjust the filters and try again.</span></div>'; }

  function closeDetail() {
    $hf('#hf-drawer').classList.remove('open');
    $hf('#hf-drawer').setAttribute('aria-hidden', 'true');
    $hf('#hf-drawer-backdrop').hidden = true;
  }

  $hf('#hf-tab-models').onclick = () => setType('models');
  $hf('#hf-tab-datasets').onclick = () => setType('datasets');
  for (const form of [$hf('#hf-model-filters'), $hf('#hf-dataset-filters')]) {
    form.onsubmit = event => { event.preventDefault(); loadResults(); };
    let timer;
    form.querySelector('[name="search"]').oninput = () => {
      clearTimeout(timer); timer = setTimeout(loadResults, 450);
    };
  }
  $hf('#hf-close-drawer').onclick = closeDetail;
  $hf('#hf-drawer-backdrop').onclick = closeDetail;
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && $hf('#hf-drawer').classList.contains('open')) closeDetail(); });
})();
