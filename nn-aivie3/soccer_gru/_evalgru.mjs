// Đo GRU cả 2 phe: nhúng gruw.json vào __GRUW__ rồi chạy các cặp đấu. Đối số: số ván mỗi cặp (mặc-định 6).
import fs from 'fs';
const G=+(process.argv[2]||6);
let html=fs.readFileSync('robotball.html','utf8');
const wj=fs.readFileSync('gruw.json','utf8').trim();
html=html.replace('/*__GRUW__*/ null','/*__GRUW__*/ '+wj);
const script=html.split('<script>')[1].split('</'+'script>')[0];
const noop=()=>{};const ctx=new Proxy({},{get:(t,p)=>(['fillStyle','strokeStyle','globalAlpha','lineWidth'].includes(p)?'':noop),set:()=>true});
const els={};const el=id=>{if(els[id])return els[id];const e={id,width:720,height:490,checked:true,value:'34',textContent:'',classList:{toggle:noop},style:{},getContext:()=>ctx};for(const k of['onclick','onchange','oninput'])Object.defineProperty(e,k,{set:noop,get:()=>noop,configurable:true});els[id]=e;return e;};
globalThis.document={getElementById:el,documentElement:{}};globalThis.getComputedStyle=()=>({getPropertyValue:()=>'#888'});
new Function(script)();const A=globalThis.__api;
console.log('GRUW nạp:',!!A.GRUW,'· ván/cặp =',G);
function m(b,r){let bl=0,rd=0,tc=0;for(let g=0;g<G;g++){A.reset();A.sides={blue:b,red:r};const b0=A.blue,r0=A.red,t0=A.touch;for(let i=0;i<12000;i++)A.step();bl+=A.blue-b0;rd+=A.red-r0;tc+=A.touch-t0;}return `${bl}-${rd} (chạm ${tc})`;}
console.log('🧠GRU(Xanh) v ⚙luật :',m('net','rule'));
console.log('⚙luật v 🧠GRU(Đỏ)   :',m('rule','net'));
console.log('🧠GRU v 🧠GRU        :',m('net','net'));
console.log('⚙luật v ⚙luật (mốc) :',m('rule','rule'));
