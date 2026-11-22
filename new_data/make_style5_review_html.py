"""Build click-to-label page for the blind review (no answers embedded)."""
import csv, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'new_data/style5-criteria/review-v1'
paths = {r['image_id']: r['image_path'] for r in csv.DictReader(open(ROOT/'data/furniture-v3-style5/train.csv', encoding='utf-8'))}
items = [dict(no=int(r['no']), id=r['image_id'], type=r['type'],
              src='../../../data/furniture-v3-style5/'+paths[r['image_id']])
         for r in csv.DictReader(open(OUT/'review.csv', encoding='utf-8-sig'))]
html = '''<!doctype html><html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Style review</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#f4f4f4;color:#222;text-align:center}
#top{padding:10px;background:#fff;border-bottom:1px solid #ddd;position:sticky;top:0}
#bar{height:8px;background:#ddd;border-radius:4px;margin-top:6px}#fill{height:8px;background:#2a7;border-radius:4px;width:0}
img{max-width:min(560px,92vw);max-height:52vh;background:#fff;margin:12px auto 4px;display:block;border:1px solid #ddd}
.btns{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;padding:6px 10px}
button{font-size:17px;padding:12px 16px;border:2px solid #888;border-radius:8px;background:#fff;cursor:pointer}
button.sel{background:#2a7;color:#fff;border-color:#175}button.u.sel{background:#e90;border-color:#a60}
.nav button{font-size:15px;padding:8px 14px}small{color:#666}
</style></head><body>
<div id="top"><b id="pos"></b> &nbsp; <span id="type"></span> &nbsp; <small id="cnt"></small><div id="bar"><div id="fill"></div></div></div>
<img id="im" alt="">
<div class="btns" id="g"></div>
<div class="btns nav"><button onclick="go(-1)">&larr; ย้อนกลับ</button><button onclick="go(1)">ข้าม &rarr;</button><button onclick="exportCsv()">ดาวน์โหลดผล (CSV)</button></div>
<small>ปุ่มลัด: 1-5 เลือกกลุ่ม, 0 = ไม่แน่ใจ, ลูกศรซ้าย/ขวา = ย้อน/ข้าม (บันทึกอัตโนมัติ ปิดแล้วเปิดต่อได้)</small>
<script>
const ITEMS=__ITEMS__;
const GROUPS=['Minimal','Modern','Traditional','Luxury','Industrial-Rustic'];
const KEY='style5_review_v1';let ans={};try{ans=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
let i=0;while(i<ITEMS.length-1&&ans[ITEMS[i].no])i++;
function save(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}}
function render(){const it=ITEMS[i];document.getElementById('im').src=it.src;
 document.getElementById('pos').textContent='#'+String(it.no).padStart(3,'0')+' / '+ITEMS.length;
 document.getElementById('type').textContent=it.type;
 const n=Object.keys(ans).length;document.getElementById('cnt').textContent='ตอบแล้ว '+n+' / '+ITEMS.length;
 document.getElementById('fill').style.width=(100*n/ITEMS.length)+'%';
 const g=document.getElementById('g');g.innerHTML='';
 GROUPS.forEach((n,k)=>{const b=document.createElement('button');b.textContent=(k+1)+'  '+n;if(ans[it.no]===n)b.className='sel';b.onclick=()=>pick(n);g.appendChild(b)});
 const u=document.createElement('button');u.textContent='0  ไม่แน่ใจ';u.className='u'+(ans[it.no]==='unsure'?' sel':'');u.onclick=()=>pick('unsure');g.appendChild(u)}
function pick(v){ans[ITEMS[i].no]=v;save();if(i<ITEMS.length-1)i++;render()}
function go(d){i=Math.max(0,Math.min(ITEMS.length-1,i+d));render()}
document.addEventListener('keydown',e=>{if(e.key>='1'&&e.key<='5')pick(GROUPS[+e.key-1]);else if(e.key==='0')pick('unsure');
 else if(e.key==='ArrowLeft')go(-1);else if(e.key==='ArrowRight')go(1)});
function exportCsv(){const rows=['no,image_id,type,your_group'].concat(ITEMS.map(it=>[it.no,it.id,it.type,ans[it.no]||''].join(',')));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\ufeff'+rows.join('\n')],{type:'text/csv'}));a.download='review-answers.csv';a.click()}
render();
</script></body></html>'''
(OUT/'review.html').write_text(html.replace('__ITEMS__', json.dumps(items)), encoding='utf-8')
print(len(items), (OUT/'review.html'))
