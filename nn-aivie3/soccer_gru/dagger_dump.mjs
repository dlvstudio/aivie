// DAgger CÂN 2 PHE: chạy GRU HIỆN-TẠI lái (cả Xanh lẫn Đỏ), ghi (obs camera GRU-ghé + nhãn THẦY) → gộp seqs_agg.json
import fs from 'fs';
let html=fs.readFileSync('robotball.html','utf8');
const wj=fs.readFileSync('gruw.json','utf8').trim();
html=html.replace('/*__GRUW__*/ null','/*__GRUW__*/ '+wj);
const script=html.split('<script>')[1].split('</'+'script>')[0];
const noop=()=>{};const ctx=new Proxy({},{get:(t,p)=>(['fillStyle','strokeStyle','globalAlpha','lineWidth'].includes(p)?'':noop),set:()=>true});
const els={};const el=id=>{if(els[id])return els[id];const e={id,width:720,height:490,checked:true,value:'34',textContent:'',classList:{toggle:noop},style:{},getContext:()=>ctx};for(const k of['onclick','onchange','oninput'])Object.defineProperty(e,k,{set:noop,get:()=>noop,configurable:true});els[id]=e;return e;};
globalThis.document={getElementById:el,documentElement:{}};globalThis.getComputedStyle=()=>({getPropertyValue:()=>'#888'});
new Function(script)();const A=globalThis.__api;
const T=120, STEPS=5000; const chunks=[];
// CÂN 2 PHE: net-Xanh vs luật-Đỏ, luật-Xanh vs net-Đỏ, net vs net → GRU lộ trạng-thái CẢ hai phía tấn-công
const modes=[['net','rule'],['net','rule'],['rule','net'],['rule','net'],['net','net'],['net','net']];
for(let g=0;g<modes.length;g++){ A.reset(); A.sides={blue:modes[g][0],red:modes[g][1]}; const seq=[[],[]];
  for(let i=0;i<STEPS;i++){ A.step(); A.robs.forEach((r,ri)=>{ if(r._x&&r._y) seq[ri].push([r._x.map(v=>+v.toFixed(4)), [ +r._y[0].toFixed(4), +r._y[1].toFixed(4) ]]); }); }
  for(const sq of seq) for(let i=0;i+T<=sq.length;i+=T) chunks.push(sq.slice(i,i+T));
  process.stdout.write(`ván ${g+1}/${modes.length} [${modes[g]}] · chunk mới ${chunks.length}\r`);
}
const IN=chunks[0][0][0].length;
const agg=JSON.parse(fs.readFileSync('seqs_agg.json','utf8'));
agg.chunks.push(...chunks);
fs.writeFileSync('seqs_agg.json', JSON.stringify(agg));
console.log(`\nDAgger +${chunks.length} chunk (GRU-ghé, cân phe) → tổng ${agg.chunks.length} · IN=${IN}`);
