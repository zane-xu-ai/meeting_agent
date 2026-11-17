const el = (id) => document.getElementById(id);
const date = (value) => value ? new Date(value).toLocaleString() : '—';

async function request(url, options) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw Error(body.detail || response.status);
  return body;
}

function input(type, value, label) {
  const item = document.createElement('input');
  item.type = type;
  if (type === 'checkbox') item.checked = value;
  else item.value = value;
  item.setAttribute('aria-label', label);
  return item;
}

function button(label, action) {
  const item = document.createElement('button');
  item.textContent = label;
  item.onclick = async () => {
    item.disabled = true;
    try { await action(); } catch (error) { el('message').textContent = error.message; }
    finally { item.disabled = false; }
  };
  return item;
}

async function load() {
  const selected = el('stage').value;
  const data = await request('/api/cache?' + new URLSearchParams({stage: selected}));
  el('stage').replaceChildren(new Option('全部', ''));
  el('policies').replaceChildren();
  el('entries').replaceChildren();
  for (const [stage, policy] of Object.entries(data.policies)) {
    const label = data.labels[stage];
    el('stage').add(new Option(label, stage));
    const row = el('policies').insertRow();
    row.insertCell().textContent = label;
    const read = input('checkbox', policy.read, label + '读取');
    const write = input('checkbox', policy.write, label + '写入');
    const ttl = input('number', policy.ttl / 3600, label + '保存小时');
    ttl.min = .05;
    ttl.step = .05;
    [read, write, ttl].forEach((node) => row.insertCell().append(node));
    row.insertCell().append(
      button('保存', async () => {
        await request('/api/cache/policies/' + stage, {
          method: 'PUT', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({read: read.checked, write: write.checked, ttl: Math.round(Number(ttl.value) * 3600)}),
        });
        el('message').textContent = label + '策略已保存';
        await load();
      }),
      button('失效本层', async () => {
        if (!confirm('使本层缓存失效？')) return;
        await request('/api/cache/stages/' + stage, {method: 'DELETE'});
        await load();
      }),
    );
  }
  el('stage').value = selected;
  for (const entry of data.entries) {
    const row = el('entries').insertRow();
    const status = entry.invalidated_at ? '已失效' : new Date(entry.expires_at) < new Date() ? '已到期' : '可用';
    [data.labels[entry.stage], date(entry.created_at), date(entry.expires_at), entry.hit_count, status].forEach(
      (value) => row.insertCell().textContent = value,
    );
    const actions = row.insertCell();
    actions.append(
      button('预览', async () => {
        const preview = await request('/api/cache/entries/' + entry.id);
        el('preview').hidden = false;
        el('preview').textContent = preview.preview;
      }),
      button('失效', async () => {
        await request('/api/cache/entries/' + entry.id, {method: 'DELETE'});
        await load();
      }),
    );
    if (['source', 'preprocess'].includes(entry.stage)) {
      const file = document.createElement('a');
      file.href = '/api/cache/entries/' + entry.id + '/media';
      file.textContent = '媒体';
      actions.append(' ', file);
    }
  }
}

el('stage').onchange = () => load().catch((error) => el('message').textContent = error.message);
load().catch((error) => el('message').textContent = error.message);
