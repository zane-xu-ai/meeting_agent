const element = (id) => document.getElementById(id);
const labels = {pending:'排队',processing:'执行中',retrying:'重试等待',cancel_requested:'正在取消',
  cancelled:'已取消',completed:'完成',failed:'失败',waiting_gpu:'等待 GPU'};
const terminal = new Set(['completed', 'failed', 'cancelled']);
let timer;
let selectedBatch = new URLSearchParams(location.search).get('batch');
let requestKey;
let requestData;

async function request(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : '请求失败');
  return data;
}

function button(title, action) {
  const item = document.createElement('button');
  item.textContent = title;
  item.onclick = async () => {
    item.disabled = true;
    try { await action(); } catch (error) { element('message').textContent = error.message; }
    finally { item.disabled = false; }
  };
  return item;
}

async function poll() {
  clearTimeout(timer);
  if (!selectedBatch) return;
  try {
    const data = await request('/api/batches/' + encodeURIComponent(selectedBatch));
    element('tasks').replaceChildren();
    element('rejections').replaceChildren(...data.rejected.map(({index, detail}) => {
      const li = document.createElement('li');
      li.textContent = `第 ${index + 1} 项：${typeof detail === 'string' ? detail : detail.message || '提交失败'}`;
      return li;
    }));
    for (const task of data.items) {
      const row = element('tasks').insertRow();
      row.insertCell().textContent = task.filename || task.task_id;
      row.insertCell().textContent = labels[task.status] || task.status;
      row.insertCell().textContent = task.error || task.step || '—';
      const bar = document.createElement('progress');
      bar.max = 100;
      bar.value = task.progress || 0;
      bar.setAttribute('aria-label', '任务进度');
      row.insertCell().append(bar);
      const actions = row.insertCell();
      if (task.status === 'completed') {
        const link = document.createElement('a');
        link.href = '/api/result/' + encodeURIComponent(task.task_id) + '/download';
        link.textContent = '下载报告';
        actions.append(link);
      } else if (task.status === 'failed' || !terminal.has(task.status) && task.status !== 'cancel_requested') {
        const action = task.status === 'failed' ? 'resume' : 'cancel';
        actions.append(button(action === 'resume' ? '续跑' : '取消', async () => {
          await request('/api/status/' + encodeURIComponent(task.task_id) + '/' + action, {method:'POST'});
          await poll();
        }));
      }
    }
    if (data.items.some((item) => !terminal.has(item.status))) timer = setTimeout(poll, 1500);
  } catch (error) {
    element('message').textContent = error.message;
  }
}

element('batchForm').onchange = () => { requestKey = undefined; requestData = undefined; };
element('batchForm').onsubmit = async (event) => {
  event.preventDefault();
  element('submit').disabled = true;
  requestKey ||= crypto.randomUUID();
  requestData ||= new FormData(event.currentTarget);
  if (!requestData.get('files')?.name) requestData.delete('files');
  if (!requestData.get('asr_model')) requestData.delete('asr_model');
  if (!requestData.get('llm_model')) requestData.delete('llm_model');
  try {
    const result = await request('/api/batches', {method:'POST', headers:{'Idempotency-Key':requestKey}, body:requestData});
    selectedBatch = result.batch_id;
    history.replaceState(null, '', '?batch=' + encodeURIComponent(selectedBatch));
    element('message').textContent = '批次已提交';
    requestKey = requestData = undefined;
    await poll();
  } catch (error) {
    element('message').textContent = error.message;
  } finally {
    element('submit').disabled = false;
  }
};

request('/api/models').then(({asr_models, llm_models}) => {
  for (const [id, models] of [['asr', asr_models], ['llm', llm_models]]) {
    for (const model of models) element(id).add(new Option(model, model));
  }
}).catch(error => element('message').textContent = error.message);
poll();
