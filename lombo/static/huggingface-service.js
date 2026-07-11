/* Fronteira HTTP da integração Hugging Face. A UI usa somente esta interface. */
'use strict';

window.HuggingFaceService = (() => {
  const BASE = '/_lombo/api/huggingface';

  async function request(path, params) {
    const url = new URL(BASE + path, window.location.origin);
    for (const [key, value] of Object.entries(params || {})) {
      if (value !== '' && value !== null && value !== undefined) url.searchParams.set(key, value);
    }
    const response = await fetch(url, { headers: { Accept: 'application/json' } });
    let data;
    try { data = await response.json(); }
    catch { data = null; }
    if (!response.ok) throw new Error(data?.detail || `Falha ao consultar o Hub (${response.status})`);
    return data;
  }

  return Object.freeze({
    listModels: filters => request('/models', filters),
    getModel: id => request(`/models/${encodeURIComponent(id)}`),
    listDatasets: filters => request('/datasets', filters),
    getDataset: id => request(`/datasets/${encodeURIComponent(id)}`),
  });
})();
