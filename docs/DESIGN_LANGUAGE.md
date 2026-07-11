# LomboLab — design language

LomboLab is a scientific workspace, not an analytics dashboard. Its primary unit
is evidence: a plot, trace, distribution, matrix, or comparison. Text identifies
the evidence and records an inference; it does not compete with the visualisation.

## Principles

Every research view should answer four questions in this order:

1. **What is happening?** Show the dominant signal and its direction.
2. **Where did it happen?** Keep layer, token, module, head, run, and condition visible.
3. **Why might it have happened?** Relate correlated measurements without claiming causality.
4. **What should I inspect next?** Turn anomalies and transitions into direct actions.

The interface optimises for long investigations: context persists, comparisons stay
aligned, and overview panels lead to specialised instruments instead of dead-end cards.

## Grid and density

- Base spacing unit: `4px`.
- Main canvas: fluid two-column research grid, `10px` gutters.
- Borders, not shadows, define most grouping. Floating inspectors may use shadow.
- Panels use `3px` radii; scientific plots and matrices remain visually rectangular.
- Default desktop shell: `214px` navigation rail, `48px` global bar, `66–74px` toolbars.
- Information density is high, but each panel has one primary reading path.

## Typography

- UI and explanations: Inter/Segoe UI/system sans.
- Measurements, axes, tokens, controls, IDs: JetBrains Mono/Cascadia Code/monospace.
- Display sizes stay restrained (`12–17px`); hierarchy comes from position, contrast,
  spacing, and type family rather than oversized headings.
- Uppercase mono eyebrow labels describe evidence type, never marketing categories.

## Colour

- Canvas `#0b0d0c`, panel `#111412`, raised panel `#181c19`.
- Primary text `#eceee9`; secondary text `#858b83`; structural lines `#292e29`.
- Amber `#e6a94c`: active focus, selected layer, primary experimental condition.
- Teal `#5eb7a6`: comparison condition or verified secondary signal.
- Violet `#9e8bd1`: uncertainty, entropy, or third series.
- Green `#8fb86c`: successful runtime/evaluation state. Red `#dc746b`: regression/error.
- Series colours are never used as decoration; labels and line styles reinforce meaning.

## Components

- **Persistent context:** project, model, runtime, active experiment, run/condition.
- **Instrument toolbar:** evidence scope selectors followed by one primary capture/run action.
- **Visualisation panel:** eyebrow, precise title, unit/method subtitle, interactive legend,
  plot, then an optional evidence footer with a next-step action.
- **Dense table:** mono numeric columns, quiet separators, colour only for meaningful deltas.
- **Inspector:** selection identity, four comparable measurements, distribution, action.
- **Status:** small dot plus explicit label; colour alone never carries the state.

## Chart system

- Shared graphite grid, mono axes, crosshair selection, and condition palette.
- Primary series: solid `1.7–2.2px`; comparison: solid or dashed; reference: `1px` dashed.
- Heatmaps use a near-black zero, continuous intensity, and a visible selected-cell inset.
- Tooltips show the exact layer/token/module first, then real (not normalised) values.
- Annotations identify transitions, divergence, emergence, stability, and outliers.
- Overview charts summarise; layer, activation, and benchmark views expose controls and detail.
- Correlation is labelled as such. Causal language is reserved for intervention results.

### Rendering layer

- General-purpose charts (line/area/timeline, bar/stacked bar, scatter, histogram, heatmap,
  radar, boxplot, violin, sparkline) render through **Apache ECharts**, vendored locally at
  `lombo/static/vendor/echarts.min.js` — no CDN, no build step. `lombo/static/charts.js` wraps
  it: a registered `lombo` theme reads the CSS custom properties above at call time, so chart
  colour always tracks the token palette instead of duplicating it, and every factory
  (`LomboCharts.line`, `.bars`, `.radar`, …) mounts with the shared crosshair tooltip, dataZoom
  where relevant, and a `saveAsImage` toolbox action for exporting a panel as PNG.
- Diagnostics (`diagnostics.js`) and the attention heatmap in the request drawer stay **bespoke
  SVG** — their per-cell token labels, replay scrubber, and synchronized crosshair predate the
  chart layer and don't map cleanly onto a general-purpose chart type. New instruments default
  to `LomboCharts`; a panel only earns bespoke SVG when a chart type can't express it (dense
  token-labelled grids, custom replay/scrub interaction).

## Motion

- Hover/focus response: `120ms`; panel/drawer transition: `180–220ms`.
- Motion communicates selection, loading, or trace progression; it is never ambient decoration.
- Heatmap cells brighten in place; plotted trajectories may draw once on new data.
- Active execution uses a low-frequency pulse (`2.6s`).
- `prefers-reduced-motion` disables all non-essential animation and transition.

