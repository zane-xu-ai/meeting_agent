const $=id=>document.getElementById(id);
let offset=0;
let queryId;
const conversation=crypto.randomUUID();
async function call(path,options){
 const response=await fetch('/api/knowledge'+path,options);
 const data=await response.json();
 if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:'请求失败');
 return data;
}
const json=(method,body)=>({method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
async function load(reset=false){
 if(reset){offset=0;$('rows').replaceChildren();}
 const [list,stats]=await Promise.all([call('/sessions?limit=30&offset='+offset),call('/coverage')]);
 $('stats').textContent=`共 ${stats.sessions} 场事件，日期已确认 ${stats.confirmed_dates} 场，待确认 ${stats.unknown_or_candidate_dates} 场。`;
 for(const item of list.items){
  const row=$('rows').insertRow();
  [item.title,item.event_date||'未知',item.time_status==='confirmed'?'已确认':'待确认'].forEach(text=>row.insertCell().textContent=text);
  const button=document.createElement('button');button.textContent='查看与确认';
  button.onclick=()=>review(item).catch(error=>$('message').textContent=error.message);row.insertCell().append(button);
 }
 offset+=list.items.length;$('more').hidden=offset>=list.total;
}
async function review(item){
 const data=await call('/sessions/'+item.id);
 $('session').value=item.id;$('eventDate').value=item.event_date||'';
 $('names').value=data.participants.filter(p=>p.active&&p.status==='confirmed').map(p=>p.display_name).join('，');
 $('detail').textContent=data.participants.filter(p=>p.active).map(p=>`${p.display_name} · ${p.status}`).join('\n')||'尚无已提取的人员信息';
 $('confirm').hidden=false;$('confirm').scrollIntoView({block:'center'});
}
$('ask').onsubmit=async event=>{
 event.preventDefault();const button=event.currentTarget.querySelector('button');button.disabled=true;$('message').textContent='';
 try{
  const data=await call('/query',json('POST',{question:$('question').value,conversation}));
  queryId=data.query_id;$('reply').textContent=data.reply;
  $('coverage').textContent=data.coverage?`状态 ${data.status} · 已扫描 ${data.coverage.scanned} 场 · 匹配 ${data.coverage.matched} 场 · 日期未确认 ${data.coverage.unknown_dates} 场`:data.status;
  $('evidence').replaceChildren(...(data.evidence||[]).map((item,index)=>{
   const li=document.createElement('li');const a=document.createElement('a');a.href='/api/knowledge/evidence/'+item.id;a.target='_blank';a.rel='noopener';a.textContent=`来源 ${index+1}`;
   li.append(a,' · '+item.quote);return li;
  }));
  $('feedback').hidden=!queryId;
 }catch(error){$('message').textContent=error.message;}finally{button.disabled=false;}
};
$('confirm').onsubmit=async event=>{
 event.preventDefault();const button=event.currentTarget.querySelector('button');button.disabled=true;
 try{
  const names=$('names').value.trim();
  await call('/sessions/'+$('session').value+'/confirmation',json('PUT',{event_date:$('eventDate').value,names:names?names.split(/[,，]/).map(n=>n.trim()).filter(Boolean):null}));
  $('message').textContent='已确认，下次查询使用更新后的事实。';$('confirm').hidden=true;await load(true);
 }catch(error){$('message').textContent=error.message;}finally{button.disabled=false;}
};
$('more').onclick=()=>load().catch(error=>$('message').textContent=error.message);
for(const [id,helpful] of [['helpful',true],['incorrect',false]])$(id).onclick=async()=>{
 try{await call('/feedback',json('POST',{query_id:queryId,helpful}));$('message').textContent='反馈已记录';}catch(error){$('message').textContent=error.message;}
};
load().catch(error=>$('message').textContent=error.message);
