import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

// Load the actual browser script without firing its DOMContentLoaded boot.
const context = vm.createContext({
  window: {},
  document: {
    querySelector: () => ({}),
    addEventListener: () => {},
  },
});
const corePath = new URL('../../lombo/static/core.js', import.meta.url);
vm.runInContext(readFileSync(corePath, 'utf8'), context, { filename: corePath.pathname });
const { sessionTitleText, conditionText, humanLabelText } = context;

test('legacy default session titles display in English', () => {
  assert.equal(sessionTitleText('Sessão sem título'), 'Untitled session');
  assert.equal(sessionTitleText('Sessão 23/10/2026, 10:30:00'), 'Session 23/10/2026, 10:30:00');
  assert.equal(sessionTitleText('Sessão 10/23/2026, 10:30:00 AM'), 'Session 10/23/2026, 10:30:00 AM');
});

test('English session titles pass through unchanged', () => {
  for (const title of ['Untitled session', 'Session 23/10/2026, 10:30:00', 'My experiment']) {
    assert.equal(sessionTitleText(title), title);
  }
});

test('custom Portuguese session titles pass through unchanged', () => {
  for (const title of ['Sessão personalizada', 'Sessão 42', 'Minha sessão']) {
    assert.equal(sessionTitleText(title), title);
  }
});

test('legacy numbered conditions display in English', () => {
  assert.equal(conditionText('condição 1'), 'condition 1');
  assert.equal(conditionText('condição 12'), 'condition 12');
});

test('English and custom conditions pass through unchanged', () => {
  for (const condition of ['condition 1', 'condition 12', 'base', 'condição personalizada']) {
    assert.equal(conditionText(condition), condition);
  }
});

test('existing condition and human label display mappings preserve their input identifiers', () => {
  const record = { condition: 'prompt=variante', label: 'correto' };
  assert.equal(conditionText(record.condition), 'prompt=variant');
  assert.equal(conditionText('prompt=variante+prompt≠'), 'prompt=variant+prompt≠');
  assert.equal(humanLabelText(record.label), 'correct');
  assert.deepEqual(record, { condition: 'prompt=variante', label: 'correto' });
});
