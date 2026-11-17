// So các checkpoint GRU + tracker(DRVW) cùng harness. Đối số: file gruw (hoặc 'tracker' = không nhúng GRU → net=DRVW).
import fs from 'fs';
const which=process.argv[2]||'gruw.json', G=+(process.argv[3]||8);
let html=fs.readFileSync('robotball.html','utf8');
if(which!=='tracker'){ const wj=fs.readFileSync(which,'utf8').trim(); html=html.replace('/*__GRUW__*/ null','/*__GRUW__*/ '+wj); }
const script=html.split('<script>')[1].split('</'+'script>')[0];
const noop=()=>{};const ctx=new Proxy({},{get:(t,p)=>(['fillStyle','strokeStyle','globalAlpha','lineWidth'].includes(p)?'':noop),set:()=>true});
const els={};const el=id=>{if(els[id])return els[id];const e={id,width:720,height:490,checked:true,value:'34',textContent:'',classList:{toggle:noop},style:{},getContext:()=>ctx};for(const k of['onclick','onchange','oninput'])Object.defineProperty(e,k,{set:noop,get:()=>noop,configurable:true});els[id]=e;return e;};
globalThis.document={getElementById:el,documentElement:{}};globalThis.getComputedStyle=()=>({getPropertyValue:()=>'#888'});
new Function(script)();const A=globalThis.__api;
function m(b,r){let bl=0,rd=0;for(let g=0;g<G;g++){A.reset();A.sides={blue:b,red:r};const b0=A.blue,r0=A.red;for(let i=0;i<12000;i++)A.step();bl+=A.blue-b0;rd+=A.red-r0;}return [bl,rd];}
const NET = which==='tracker' ? 'net' : 'gru';   // 'net'=tracker+MLP, 'gru'=GRU
const [xb,xr]=m(NET,'rule'), [rb,rr]=m('rule',NET), [nb,nr]=m(NET,NET);
const scored=xb+rr, conceded=xr+rb;   // tổng bàn net GHI (2 phe) vs net THỦNG
console.log(`${which}: net ghi ${scored} / thủng ${conceded} (Xanh ${xb}-${xr}, Đỏ ghi ${rr} thủng ${rb}) · net-v-net ${nb}-${nr} · ${G} ván/cặp`);
