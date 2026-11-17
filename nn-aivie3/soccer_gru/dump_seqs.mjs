// Dump chuỗi (obs camera 51 + nhãn thầy [steer,thr]) từ sim JS → seqs.json cho PyTorch BPTT. THẦY vs THẦY (cân 2 phe).
import fs from 'fs';
const html=fs.readFileSync('robotball.html','utf8');const script=html.split('<script>')[1].split('</'+'script>')[0];
const noop=()=>{};const ctx=new Proxy({},{get:(t,p)=>(['fillStyle','strokeStyle','globalAlpha','lineWidth'].includes(p)?'':noop),set:()=>true});
const els={};const el=id=>{if(els[id])return els[id];const e={id,width:720,height:490,checked:true,value:'34',textContent:'',classList:{toggle:noop},style:{},getContext:()=>ctx};for(const k of['onclick','onchange','oninput'])Object.defineProperty(e,k,{set:noop,get:()=>noop,configurable:true});els[id]=e;return e;};
globalThis.document={getElementById:el,documentElement:{}};globalThis.getComputedStyle=()=>({getPropertyValue:()=>'#888'});
new Function(script)();const A=globalThis.__api;
const T=120, GAMES=10, STEPS=5000; const chunks=[];
for(let g=0;g<GAMES;g++){ A.reset(); A.sides={blue:'rule',red:'rule'}; const seq=[[],[]];
  for(let i=0;i<STEPS;i++){ A.step(); A.robs.forEach((r,ri)=>{ if(r._x&&r._y) seq[ri].push([r._x.map(v=>+v.toFixed(4)), [ +r._y[0].toFixed(4), +r._y[1].toFixed(4) ]]); }); }
  for(const sq of seq) for(let i=0;i+T<=sq.length;i+=T) chunks.push(sq.slice(i,i+T));
  process.stdout.write(`game ${g+1}/${GAMES} · chunks ${chunks.length}\r`);
}
fs.writeFileSync('seqs.json', JSON.stringify({T, IN:chunks[0][0][0].length, chunks}));
console.log(`\nĐÃ DUMP ${chunks.length} chunk × T=${T} · IN=${chunks[0][0][0].length} · ${(fs.statSync('seqs.json').size/1e6).toFixed(1)}MB`);
