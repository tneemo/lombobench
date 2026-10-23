/* LomboLab chart layer — thin wrapper over Apache ECharts.
   The "lombo" theme reads CSS tokens (dark/amber); standardized factories by
   chart type. Views call LomboCharts.<type>(el, data, opts) and
   receive a mounted ECharts instance — they never call echarts directly. */
'use strict';

window.LomboCharts = (() => {
  const instances = new WeakMap();
  let resizeObserver = null;
  let themed = false;

  function reducedMotion() {
    return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
  }

  function tokens() {
    const style = getComputedStyle(document.documentElement);
    const v = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
    return {
      bg: v('--bg', '#0b0d0c'), surface: v('--surface', '#111412'),
      surface2: v('--surface-2', '#181c19'), line: v('--line', '#292e29'),
      lineStrong: v('--line-strong', '#3a4039'), ink: v('--ink', '#eceee9'),
      muted: v('--muted', '#858b83'), accent: v('--accent', '#e6a94c'),
      paneA: v('--pane-a', '#e6a94c'), paneB: v('--pane-b', '#5eb7a6'),
      paneC: v('--pane-c', '#9e8bd1'), paneD: v('--pane-d', '#a3bd65'),
      ok: v('--ok', '#8fb86c'), warn: v('--warn', '#e6a94c'), err: v('--err', '#dc746b'),
      mono: v('--mono', 'monospace'),
    };
  }

  function registerTheme() {
    const t = tokens();
    const axis = {
      axisLine: { lineStyle: { color: t.line } },
      axisTick: { lineStyle: { color: t.line } },
      axisLabel: { color: t.muted, fontFamily: t.mono, fontSize: 9 },
      splitLine: { lineStyle: { color: t.line, type: 'dashed' } },
    };
    window.echarts.registerTheme('lombo', {
      color: [t.accent, t.paneB, t.paneC, t.paneD, t.ok, t.err],
      backgroundColor: 'transparent',
      textStyle: { fontFamily: t.mono, color: t.muted, fontSize: 10 },
      title: { textStyle: { color: t.ink, fontFamily: t.mono, fontWeight: 600, fontSize: 12 } },
      categoryAxis: axis, valueAxis: axis, timeAxis: axis, logAxis: axis,
      line: { lineStyle: { width: 1.8 }, symbol: 'circle', symbolSize: 5 },
      legend: { textStyle: { color: t.muted, fontFamily: t.mono, fontSize: 9 }, icon: 'roundRect' },
      tooltip: {
        backgroundColor: t.surface2, borderColor: t.lineStrong, borderWidth: 1,
        textStyle: { color: t.ink, fontFamily: t.mono, fontSize: 10 },
        axisPointer: { lineStyle: { color: t.accent, type: 'dashed' }, crossStyle: { color: t.accent } },
      },
      radar: {
        axisLine: { lineStyle: { color: t.line } }, splitLine: { lineStyle: { color: t.line } },
        splitArea: { areaStyle: { color: ['transparent'] } },
        axisName: { color: t.muted, fontFamily: t.mono, fontSize: 9 },
      },
      visualMap: { textStyle: { color: t.muted, fontFamily: t.mono, fontSize: 9 } },
      dataZoom: { textStyle: { color: t.muted, fontFamily: t.mono, fontSize: 9 },
                 borderColor: t.line, fillerColor: 'rgba(230,169,76,.08)',
                 handleStyle: { color: t.accent, borderColor: t.accent } },
    });
    themed = true;
    return t;
  }

  function ensureTheme() { if (!themed) registerTheme(); return tokens(); }

  function ensureResizeObserver() {
    if (resizeObserver) return resizeObserver;
    resizeObserver = new ResizeObserver(entries => {
      for (const entry of entries) {
        const instance = instances.get(entry.target);
        if (instance && !instance.isDisposed()) instance.resize();
      }
    });
    return resizeObserver;
  }

  // Views re-render on every data refresh; disposing the previous instance
  // on remount (rather than reusing it) keeps series/axis state from a
  // different chart shape from leaking into the next setOption call.
  function init(el, opts = {}) {
    ensureTheme();
    const existing = instances.get(el);
    if (existing && !existing.isDisposed()) existing.dispose();
    const instance = window.echarts.init(el, 'lombo', { renderer: opts.renderer || 'canvas' });
    instances.set(el, instance);
    ensureResizeObserver().observe(el);
    return instance;
  }

  function chrome(opts = {}) {
    const t = tokens();
    const base = {
      animation: !reducedMotion(), animationDuration: 220, animationEasing: 'cubicOut',
    };
    if (opts.toolbox !== false) {
      base.toolbox = {
        right: 6, top: 0, itemSize: 12, iconStyle: { borderColor: t.muted },
        emphasis: { iconStyle: { borderColor: t.accent } },
        feature: { saveAsImage: { title: 'export PNG', backgroundColor: t.bg, pixelRatio: 2 } },
      };
    }
    return base;
  }

  function grid(opts = {}) {
    return { left: opts.left ?? 46, right: opts.right ?? 16,
            top: opts.top ?? (opts.toolbox === false ? 10 : 28),
            bottom: opts.bottom ?? 30, containLabel: true };
  }

  function mount(el, option, opts) {
    const instance = init(el, opts);
    instance.setOption(option, true);
    return instance;
  }

  function seriesColor(t, index, explicit) {
    if (explicit) return explicit;
    return [t.accent, t.paneB, t.paneC, t.paneD, t.ok, t.err][index % 6];
  }

  function line(el, { categories, series, timeAxis = false }, opts = {}) {
    const t = ensureTheme();
    const option = {
      ...chrome(opts), grid: grid(opts),
      xAxis: timeAxis
        ? { type: 'time' }
        : { type: 'category', data: categories, boundaryGap: !!opts.area },
      yAxis: { type: 'value', name: opts.yName || '', nameTextStyle: { fontSize: 9 } },
      tooltip: { trigger: 'axis' },
      legend: series.length > 1 ? { top: 0, left: 0 } : undefined,
      series: series.map((s, i) => ({
        name: s.name, type: 'line', data: s.data, smooth: opts.smooth ?? false,
        showSymbol: (s.data?.length || 0) <= 40,
        stack: opts.stack ? (opts.stackName || 'total') : undefined,
        itemStyle: { color: seriesColor(t, i, s.color) },
        lineStyle: { color: seriesColor(t, i, s.color), width: 1.8,
                    type: s.dashed ? 'dashed' : 'solid' },
        areaStyle: (opts.area || opts.stack)
          ? { color: seriesColor(t, i, s.color), opacity: opts.stack ? .55 : .16 } : undefined,
      })),
    };
    return mount(el, option, opts);
  }

  function timeline(el, { series }, opts = {}) {
    const base = line(el, { categories: null, series, timeAxis: true }, { ...opts, area: opts.area ?? true });
    base.setOption({
      dataZoom: [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 4 }],
    });
    return base;
  }

  function bars(el, { categories, series }, opts = {}) {
    const t = ensureTheme();
    const horizontal = !!opts.horizontal;
    const catAxis = { type: 'category', data: categories,
                      axisLabel: { fontFamily: t.mono, fontSize: 9,
                                  width: opts.labelWidth, overflow: opts.labelWidth ? 'truncate' : undefined } };
    const valAxis = { type: 'value', name: opts.yName || '', splitNumber: 3,
                      axisLabel: { formatter: _fmtAxisNum, hideOverlap: true } };
    const option = {
      ...chrome(opts), grid: grid({ ...opts, left: horizontal ? (opts.left ?? 96) : opts.left }),
      xAxis: horizontal ? valAxis : catAxis, yAxis: horizontal ? catAxis : valAxis,
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: series.length > 1 ? { top: 0, left: 0 } : undefined,
      series: series.map((s, i) => ({
        name: s.name, type: 'bar', data: s.data, barMaxWidth: 22,
        itemStyle: { color: seriesColor(t, i, s.color), borderRadius: horizontal ? [0, 2, 2, 0] : [2, 2, 0, 0] },
      })),
    };
    return mount(el, option, opts);
  }

  function stackedBars(el, { categories, series }, opts = {}) {
    const t = ensureTheme();
    const horizontal = !!opts.horizontal;
    const catAxis = { type: 'category', data: categories };
    const valAxis = { type: 'value', name: opts.yName || '', splitNumber: 3,
                      axisLabel: { formatter: _fmtAxisNum, hideOverlap: true } };
    const option = {
      ...chrome(opts), grid: grid({ ...opts, left: horizontal ? (opts.left ?? 96) : opts.left }),
      xAxis: horizontal ? valAxis : catAxis, yAxis: horizontal ? catAxis : valAxis,
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { top: 0, left: 0 },
      series: series.map((s, i) => ({
        name: s.name, type: 'bar', stack: opts.stackName || 'total', data: s.data,
        barMaxWidth: 22, itemStyle: { color: seriesColor(t, i, s.color) },
      })),
    };
    return mount(el, option, opts);
  }

  function scatter(el, { series }, opts = {}) {
    const t = ensureTheme();
    const option = {
      ...chrome(opts), grid: grid(opts),
      xAxis: { type: 'value', name: opts.xName || '', nameTextStyle: { fontSize: 9 } },
      yAxis: { type: 'value', name: opts.yName || '', nameTextStyle: { fontSize: 9 } },
      tooltip: { trigger: 'item', formatter: opts.tooltipFormatter },
      legend: series.length > 1 ? { top: 0, left: 0 } : undefined,
      series: series.map((s, i) => ({
        name: s.name, type: 'scatter', data: s.data, symbolSize: opts.symbolSize ?? 7,
        itemStyle: { color: seriesColor(t, i, s.color), opacity: .82 },
      })),
    };
    return mount(el, option, opts);
  }

  function histogram(el, { edges, counts, name, color }, opts = {}) {
    const t = ensureTheme();
    const labels = counts.map((_, i) => ((edges[i] + edges[i + 1]) / 2));
    const option = {
      ...chrome(opts), grid: grid(opts),
      xAxis: { type: 'category', data: labels.map(v => _fmtAxisNum(v)),
              axisLabel: { interval: Math.ceil(labels.length / 8) } },
      yAxis: { type: 'value', name: 'n' },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      series: [{ name: name || 'n', type: 'bar', data: counts, barMinWidth: 2, barCategoryGap: '2%',
                itemStyle: { color: color || t.accent } }],
    };
    return mount(el, option, opts);
  }

  function heatmap(el, { xCategories, yCategories, data, min, max }, opts = {}) {
    const t = ensureTheme();
    const values = data.map(d => d[2]);
    const option = {
      ...chrome({ ...opts, toolbox: opts.toolbox ?? false }),
      grid: grid({ ...opts, left: opts.left ?? 60 }),
      xAxis: { type: 'category', data: xCategories, splitArea: { show: true } },
      yAxis: { type: 'category', data: yCategories, splitArea: { show: true } },
      tooltip: { trigger: 'item', formatter: opts.tooltipFormatter },
      visualMap: {
        min: min ?? Math.min(0, ...values), max: max ?? Math.max(1, ...values),
        calculable: false, orient: 'horizontal', right: 6, top: 0, itemWidth: 8, itemHeight: 60,
        inRange: { color: [t.bg, t.accent] },
      },
      series: [{ type: 'heatmap', data, itemStyle: { borderColor: t.bg, borderWidth: 1 } }],
    };
    return mount(el, option, opts);
  }

  function radar(el, { indicators, series }, opts = {}) {
    const t = ensureTheme();
    const option = {
      ...chrome(opts),
      legend: { top: 0, left: 0 },
      radar: { indicator: indicators, radius: '62%', center: ['50%', '54%'] },
      series: [{
        type: 'radar',
        data: series.map((s, i) => ({
          name: s.name, value: s.values,
          areaStyle: { color: seriesColor(t, i, s.color), opacity: .14 },
          lineStyle: { color: seriesColor(t, i, s.color), width: 1.6 },
          itemStyle: { color: seriesColor(t, i, s.color) },
        })),
      }],
    };
    return mount(el, option, opts);
  }

  function boxplot(el, { categories, data, outliers }, opts = {}) {
    const t = ensureTheme();
    const series = [{
      name: opts.name || 'distribution', type: 'boxplot', data,
      itemStyle: { color: t.surface2, borderColor: t.accent, borderWidth: 1.4 },
    }];
    if (outliers?.length) {
      series.push({ name: 'outliers', type: 'scatter', data: outliers, symbolSize: 5,
                    itemStyle: { color: t.err, opacity: .7 } });
    }
    const option = {
      ...chrome(opts), grid: grid(opts),
      xAxis: { type: 'category', data: categories, boundaryGap: true },
      yAxis: { type: 'value', name: opts.yName || '' },
      tooltip: { trigger: 'item' },
      series,
    };
    return mount(el, option, opts);
  }

  // ECharts has no built-in violin type; a `custom` series draws a mirrored
  // polygon per category from its histogram (density proxy), using
  // api.coord()/api.size() so it stays correct across resizes and zoom.
  function violin(el, { categories, distributions }, opts = {}) {
    const t = ensureTheme();
    const maxCount = Math.max(1, ...distributions.flatMap(d => d?.counts || []));
    const option = {
      ...chrome(opts), grid: grid(opts),
      xAxis: { type: 'category', data: categories, boundaryGap: true },
      yAxis: { type: 'value', name: opts.yName || '' },
      tooltip: { trigger: 'item', formatter: params => {
        const dist = distributions[params.dataIndex];
        return dist ? `${categories[params.dataIndex]}<br/>n = ${dist.counts.reduce((a, b) => a + b, 0)}` : '';
      } },
      series: [{
        type: 'custom', name: opts.name || 'distribution',
        renderItem: (params, api) => {
          const dist = distributions[params.dataIndex];
          if (!dist || !dist.counts?.length) return null;
          const catWidth = api.size([1, 0])[0];
          const halfW = Math.max(3, catWidth * 0.36);
          const left = [], right = [];
          for (let i = 0; i < dist.counts.length; i++) {
            const mid = (dist.edges[i] + dist.edges[i + 1]) / 2;
            const w = (dist.counts[i] / maxCount) * halfW;
            const p = api.coord([params.dataIndex, mid]);
            left.push([p[0] - w, p[1]]);
            right.unshift([p[0] + w, p[1]]);
          }
          return { type: 'polygon', shape: { points: [...left, ...right] },
                   style: api.style({ fill: t.accent, opacity: .3, stroke: t.accent, lineWidth: 1 }) };
        },
        data: categories.map((_, i) => i),
        encode: { x: -1, y: -1 },
      }],
    };
    return mount(el, option, opts);
  }

  function sparkline(el, values, color, opts = {}) {
    const t = ensureTheme();
    const option = {
      animation: !reducedMotion(), grid: { left: 1, right: 1, top: 3, bottom: 1 },
      xAxis: { type: 'category', show: false, boundaryGap: false, data: values.map((_, i) => i) },
      yAxis: { type: 'value', show: false, min: 'dataMin', max: 'dataMax' },
      tooltip: { show: false },
      series: [{ type: 'line', data: values, showSymbol: false, silent: true,
                lineStyle: { color: color || t.accent, width: 1.6 },
                areaStyle: { color: color || t.accent, opacity: .14 } }],
    };
    return mount(el, option, { ...opts, toolbox: false });
  }

  function _fmtAxisNum(v) {
    if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}k`;
    return Number.isInteger(v) ? String(v) : v.toFixed(v < 10 ? 2 : 1);
  }

  function connect(chartInstances) {
    window.echarts.connect(chartInstances.filter(c => c && !c.isDisposed()));
  }

  function dispose(el) {
    const instance = instances.get(el);
    if (instance && !instance.isDisposed()) {
      resizeObserver?.unobserve(el);
      instance.dispose();
    }
  }

  function exportPNG(instance, filename = 'lombolab-chart.png') {
    if (!instance || instance.isDisposed()) return;
    const url = instance.getDataURL({ pixelRatio: 2, backgroundColor: tokens().bg });
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
  }

  return Object.freeze({
    tokens, mount, line, timeline, bars, stackedBars, scatter, histogram,
    heatmap, radar, boxplot, violin, sparkline, connect, dispose, exportPNG,
  });
})();
