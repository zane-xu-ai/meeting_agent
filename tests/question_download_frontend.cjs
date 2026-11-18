// Run the page setup and completion path with controlled browser/Vue dependencies.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require('node:path').join(__dirname, '../frontend/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

async function check(category, available) {
  let ui;
  const mounted = [];
  const opened = [];
  const calls = [];
  const context = {
    Vue: {
      createApp({setup}) { return {mount() { ui = setup(); }}; },
      ref(value) { return {value}; }, reactive(value) { return value; },
      computed(fn) { return {get value() { return fn(); }}; },
      onMounted(fn) { mounted.push(fn); }, onUnmounted() {},
    },
    window: {location: {search: '?task=completed-interview'}, open(url) { opened.push(url); }},
    document: {documentElement: {setAttribute() {}}},
    localStorage: {getItem() {return null;}, setItem() {}},
    URLSearchParams, console, setTimeout, clearTimeout, setInterval, clearInterval,
    fetch: async (url) => {
      calls.push(url);
      if (url.includes('/api/status/')) return {ok:true, json:async () => ({
        status:'completed', filename:'existing.m4a', progress:100,
        // Simulate completion status missing category. The result response is authoritative.
      })};
      if (url.includes('/api/result/')) return {ok:true, json:async () => ({
        content:'# Existing report', rec_type:category, questions_available:available,
      })};
      return {ok:true, json:async () => ({status:'ok', asr_models:[], llm_models:[]})};
    },
  };
  vm.runInNewContext(script, context);
  for (const callback of mounted) callback();
  for (let tick = 0; tick < 10 && ui.phase.value !== 'done'; tick++) {
    await new Promise(resolve => setImmediate(resolve));
  }
  assert.equal(ui.phase.value, 'done');
  assert.equal(ui.resultMd.value, '# Existing report');
  assert.equal(ui.meta.rec_type, category || '');
  assert.equal(ui.questionsAvailable.value, available);
  const visible = vm.runInNewContext(
    html.match(/v-if="([^"]+)"\s+@click="downloadQuestions"/)[1],
    {activeResultTab:ui.activeResultTab.value, meta:ui.meta, questionsAvailable:ui.questionsAvailable.value},
  );
  assert.equal(visible, category === 'interview' || available);
  if (visible) {
    ui.downloadQuestions();
    assert.deepEqual(opened, ['/api/questions/completed-interview/download']);
  }
  assert.ok(calls.every(url => !url.includes('/upload') && !url.includes('/submit-url')));
}
(async () => {
  await check('interview', true);
  await check('general', false);
  await check(null, true);
  console.log('Completed task restores report metadata and correct download actions.');
})().catch(error => {console.error(error); process.exit(1);});
