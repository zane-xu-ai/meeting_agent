const byId = (id) => document.getElementById(id);
const labels = { completed: '成功', processing: '执行中', pending: '待派发', failed: '失败',
  cancelled: '取消', meeting: '会议', interview: '面试', general: '通用', video: '视频' };
const label = (value) => labels[value] || value || '—';
let nextCursor = '';
let requestNumber = 0;
const date = (value) => value ? new Date(value).toLocaleString() : '—';
const cell = (row, value) => row.insertCell().textContent = value;

async function read(url) {
  const response = await fetch(url);
  const body = await response.json();
  if (!response.ok) throw Error(body.detail || response.status);
  return body;
}

async function load(append = false) {
  const request = ++requestNumber;
  const params = new URLSearchParams(new FormData(byId('filters')));
  if (append && nextCursor) params.set('cursor', nextCursor);
  byId('more').disabled = true;
  try {
    const data = await read('/api/history?' + params);
    if (request !== requestNumber) return;
    if (!append) byId('rows').replaceChildren();
    nextCursor = data.next_cursor;
    byId('more').hidden = !nextCursor;
    for (const item of data.items) {
      const row = byId('rows').insertRow();
      const button = document.createElement('button');
      button.textContent = item.title || item.state.filename || item.id;
      button.onclick = () => show(item.id).catch(error => byId('message').textContent = error.message);
      row.insertCell().append(button);
      cell(row, date(item.created_at));
      cell(row, label(item.state.rec_type));
      cell(row, label(item.state.status));
      const bar = document.createElement('progress');
      bar.max = 100;
      bar.value = item.state.progress || 0;
      bar.setAttribute('aria-label', '任务进度');
      row.insertCell().append(bar);
    }
    byId('message').textContent = byId('rows').rows.length ? '' : '暂无分析记录';
  } catch (error) {
    if (request === requestNumber) byId('message').textContent = error.message;
  } finally {
    if (request === requestNumber) byId('more').disabled = false;
  }
}

async function show(id) {
  const data = await read('/api/history/' + encodeURIComponent(id));
  byId('detail').hidden = false;
  byId('title').textContent = data.analysis?.title || data.source.title || data.task.filename;
  byId('meta').textContent = [
    data.source.original_filename || data.source.canonical_url,
    `创建 ${date(data.source.source_created_at)}`,
    `修改 ${date(data.source.source_modified_at)}`,
    data.media?.duration_ms ? `${Math.round(data.media.duration_ms / 1000)} 秒` : '',
  ].filter(Boolean).join(' · ');
  byId('stats').textContent = [
    label(data.task.status), label(data.task.rec_type), data.task.asr_model_used, data.analysis?.actual_model,
    data.task.cache_hit ? '缓存命中' : '',
    data.task.total_duration ? `耗时 ${data.task.total_duration} 秒` : '',
  ].filter(Boolean).join(' · ');
  byId('report').textContent = data.analysis?.report_md || data.task.error || data.task.step;
  const video = byId('media');
  video.hidden = !data.media?.artifact_id;
  if (!video.hidden) video.src = '/api/history/' + encodeURIComponent(id) + '/media';
  else { video.removeAttribute('src'); video.load(); }
  const links = byId('downloads');
  links.replaceChildren();
  if (data.analysis) {
    for (const [kind, name, enabled] of [
      ['result', '下载报告', true], ['asr', '下载转写', data.analysis.transcript_id],
      ['questions', '下载问题清单', data.analysis.questions_md],
    ]) {
      if (!enabled) continue;
      const link = document.createElement('a');
      link.href = `/api/${kind}/${encodeURIComponent(id)}/download`;
      link.textContent = name;
      links.append(link);
    }
  }
  byId('segments').replaceChildren(...data.segments.map(segment => {
    const item = document.createElement('li');
    const time = (segment.start_ms || 0) / 1000;
    if (!video.hidden) {
      const jump = document.createElement('button');
      jump.textContent = `${time.toFixed(1)} 秒`;
      jump.onclick = () => { video.currentTime = time; };
      item.append(jump);
    }
    item.append(' ' + (segment.speaker || '') + ' ' + segment.text);
    return item;
  }));
  byId('questionPanel').hidden = !data.questions.length;
  byId('questions').replaceChildren(...data.questions.map(question => {
    const item = document.createElement('li');
    item.textContent = `${question.question_text}  ${question.score ?? '—'}/100`;
    return item;
  }));
  byId('detail').scrollIntoView();
}

byId('filters').onsubmit = (event) => { event.preventDefault(); load(); };
byId('more').onclick = () => load(true);
load();
