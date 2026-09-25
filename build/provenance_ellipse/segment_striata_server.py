#!/usr/bin/env python
"""3D striatal segmentation: per-side ELLIPSOID + ONE SHARED THRESHOLD, tri-planar browser UI.

    conda activate paire_training && python segment_striata_server.py --uids meta/control105.csv
    open http://localhost:8766   (ssh -L 8766:localhost:8766 <server>)

v2 (2026-08-17, user's correction): a SINGLE threshold for both striata -- the aim is separating signal
from BACKGROUND, which is common to the scan; per-side relative cuts let the diseased side's lower peak
drop its cut level and hide the asymmetry being measured. The shared rule matches what the expert
implicitly did on the first 105 (cut/better-side-max = 0.494 +- 0.039):
    cut = thr x max(in-ellipsoid maxima of BOTH sides)          thr slider, default 0.50
Workflow: place/adjust ellipsoids (auto-init at each hemisphere's bounded smoothed peak), tune ONE
threshold until contours sit on the background edge. Wheel/hover+arrows scroll slices. SAVE writes
meta/striatal_seg.csv (thr column = shared value on both rows) + meta/striatal_masks/<uid>.npz.
Axes 0=L-R, 1=A-P, 2=S-I; volumes whole-brain-mean normalised.
"""
import argparse, csv, io, json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np, pandas as pd
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ap = argparse.ArgumentParser()
ap.add_argument("--uids", default="meta/control105.csv")
ap.add_argument("--port", type=int, default=8766)
args = ap.parse_args()
ISO = "/ssd/datasets/DAT_SCAN/iso"
UIDS = pd.read_csv(args.uids)["uid"].tolist()
OUTC, OUTM = "meta/striatal_seg.csv", "meta/striatal_masks"
os.makedirs(OUTM, exist_ok=True)
CMAPS = sorted(plt.colormaps())
SC = 3
_cache = {}

def vol(uid):
    if uid not in _cache:
        if len(_cache) > 12: _cache.pop(next(iter(_cache)))
        v = np.asarray(np.load(f"{ISO}/{uid}.npy"), dtype=np.float32)
        v = np.clip(v / max(v[v > 0.05 * v.max()].mean(), 1e-6), 0, 12)
        _cache[uid] = v
    return _cache[uid]

def defaults(uid):
    v = vol(uid); s = ndi.uniform_filter(v, 5); sh = v.shape
    y0, y1 = int(0.45*sh[1]), int(0.75*sh[1]); z0, z1 = int(0.13*sh[2]), int(0.62*sh[2])
    out = {}
    for side, (f0, f1) in (("L", (0.28, 0.53)), ("R", (0.49, 0.69))):   # expert-derived bounds (no parotids)
        x0, x1 = int(f0*sh[0]), int(f1*sh[0])
        sub = s[x0:x1, y0:y1, z0:z1]
        c = np.unravel_index(np.argmax(sub), sub.shape)
        out[side] = {"c": [int(c[0])+x0, int(c[1])+y0, int(c[2])+z0], "r": [10, 16, 10]}
    return out

def ell_mask(shape, c, r):
    gx, gy, gz = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((gx-c[0])**2/r[0]**2 + (gy-c[1])**2/r[1]**2 + (gz-c[2])**2/r[2]**2) <= 1.0

def joint_seg(v, sides, thr):
    """Shared cut = thr x max over BOTH ellipsoids."""
    es = {s: ell_mask(v.shape, p["c"], p["r"]) for s, p in sides.items()}
    mxs = {s: (float(v[e].max()) if e.any() else 0.0) for s, e in es.items()}
    cut = thr * max(mxs.values()) if mxs else 0.0
    return {s: es[s] & (v >= cut) for s in es}, cut, mxs

def parse_side(q, k):
    if k not in q: return None
    p = [float(x) for x in q[k].split(",")]
    return {"c": p[0:3], "r": p[3:6]}

AXF = {"ax": 2, "co": 1, "sa": 0}
REM = {"ax": (0, 1), "co": (0, 2), "sa": (1, 2)}

def slice_png(uid, plane, idx, cmap, vmax, sides, thr):
    v = vol(uid); f = AXF[plane]; a, b = REM[plane]
    idx = int(np.clip(idx, 0, v.shape[f] - 1))
    base = np.take(v, idx, axis=f)
    segs, cut, _ = joint_seg(v, sides, thr) if sides else ({}, 0, {})
    fig = plt.figure(figsize=(base.shape[0]*SC/100, base.shape[1]*SC/100), dpi=100)
    axp = fig.add_axes([0, 0, 1, 1]); axp.axis("off")
    axp.imshow(base.T, origin="lower", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", aspect="auto")
    for side, col in (("L", "#00e5ff"), ("R", "#ffd600")):
        s = sides.get(side)
        if not s: continue
        c, r = s["c"], s["r"]
        rem = 1.0 - ((idx - c[f]) / r[f]) ** 2
        if rem > 0:
            axp.add_patch(Ellipse((c[a], c[b]), 2*r[a]*np.sqrt(rem), 2*r[b]*np.sqrt(rem),
                                  fill=False, color=col, lw=1.2, ls="--"))
            sl = np.take(segs[side], idx, axis=f)
            if sl.any():
                axp.contour(sl.T.astype(float), levels=[0.5], colors=[col], linewidths=1.6,
                            origin="lower", extent=(-0.5, base.shape[0]-0.5, -0.5, base.shape[1]-0.5))
    axp.set_xlim(-0.5, base.shape[0]-0.5); axp.set_ylim(-0.5, base.shape[1]-0.5)
    bio = io.BytesIO(); fig.savefig(bio, format="png"); plt.close(fig)
    return bio.getvalue()

def saved():
    d = {}
    if os.path.exists(OUTC):
        for r in csv.DictReader(open(OUTC)): d.setdefault(r["uid"], set()).add(r["side"])
    return {u: sorted(s) for u, s in d.items()}

PAGE = r"""<!doctype html><meta charset=utf-8><title>striatal seg (shared thr)</title>
<style>body{font:13px sans-serif;margin:10px;background:#111;color:#ddd}
.pl{display:inline-block;vertical-align:top;margin-right:8px;text-align:center}
.pl img{border:1px solid #444;cursor:crosshair;display:block}
.pl img.act{border:1px solid #6cf;box-shadow:0 0 6px #269}
input[type=range]{vertical-align:middle}button{margin:2px;padding:5px 9px}.on{background:#2a6}
.sideL{color:#00e5ff}.sideR{color:#ffd600}#stats{margin:6px 0;font-family:monospace}
td{padding:1px 6px}</style>
<div id=top></div>
<div id=planes></div>
<div>cmap <select id=cm></select> vmax <input type=range id=vm min=1 max=12 step=.5 value=8><span id=vmn>8</span>
 &nbsp; mode <button id=mnav class=on>navigate</button><button id=mL class=sideL>place L</button>
 <button id=mR class=sideR>place R</button></div>
<table><tr><th></th><th>rx(LR)</th><th>ry(AP)</th><th>rz(SI)</th></tr>
<tr class=sideL><td><b>L</b></td>
<td><input type=range id=Lrx min=3 max=25 value=10><span id=Lrxn></span></td>
<td><input type=range id=Lry min=3 max=30 value=16><span id=Lryn></span></td>
<td><input type=range id=Lrz min=3 max=25 value=10><span id=Lrzn></span></td></tr>
<tr class=sideR><td><b>R</b></td>
<td><input type=range id=Rrx min=3 max=25 value=10><span id=Rrxn></span></td>
<td><input type=range id=Rry min=3 max=30 value=16><span id=Rryn></span></td>
<td><input type=range id=Rrz min=3 max=25 value=10><span id=Rrzn></span></td></tr></table>
<div><b>SHARED threshold</b> (x better-side max)
 <input type=range id=thr min=0.10 max=0.95 step=0.01 value=0.50 style="width:300px"><span id=thrn>0.50</span></div>
<div id=stats></div>
<div><button id=save style="background:#264">SAVE both sides</button>
 <button id=reset>reset to auto</button> <button id=prev>&lt; prev</button><button id=next>next &gt;</button>
 <span id=prog></span></div>
<div id=list style="margin-top:8px;font-size:12px;line-height:1.7"></div>
<script>
let S={uids:[],done:{},i:0,cmap:'gray',vmax:8,mode:'nav',shape:null,idx:{ax:0,co:0,sa:0},
       seg:{L:null,R:null},thr:0.50,active:null};
const $=id=>document.getElementById(id);
const PLANES=['ax','co','sa'],TITLE={ax:'axial',co:'coronal',sa:'sagittal'};
function q(side){const s=S.seg[side];return s?[...s.c,...s.r].join(','):''}
async function init(){
 const r=await (await fetch('/init')).json();
 S.uids=r.uids;S.done=r.done;
 r.cmaps.forEach(c=>{const o=document.createElement('option');o.text=c;$('cm').add(o)});
 $('cm').value='gray';
 S.i=S.uids.findIndex(u=>!(S.done[u]||[]).length);if(S.i<0)S.i=0;
 $('planes').innerHTML=PLANES.map(p=>`<div class=pl><div>${TITLE[p]} <span id=${p}n></span></div>
   <img id=im_${p}><input type=range id=sl_${p} style="width:90%"></div>`).join('');
 bind();load();}
function bind(){
 $('cm').onchange=()=>{S.cmap=$('cm').value;drawAll()};
 $('vm').oninput=()=>{S.vmax=+$('vm').value;$('vmn').textContent=S.vmax;drawAll()};
 $('thr').oninput=()=>{S.thr=+$('thr').value;$('thrn').textContent=S.thr;drawAll();stats()};
 [['mnav','nav'],['mL','L'],['mR','R']].forEach(([id,m])=>$(id).onclick=()=>{S.mode=m;
   ['mnav','mL','mR'].forEach(x=>$(x).classList.toggle('on',x===id))});
 PLANES.forEach(p=>{
  $('sl_'+p).oninput=()=>{S.idx[p]=+$('sl_'+p).value;draw(p)};
  $('im_'+p).onclick=e=>{setActive(p);click(p,e)};
  $('im_'+p).onmouseenter=()=>setActive(p);
  $('im_'+p).addEventListener('wheel',e=>{e.preventDefault();step(p,e.deltaY<0?1:-1)},{passive:false});});
 document.addEventListener('keydown',e=>{
  if(!S.active)return;
  if(e.key==='ArrowUp'||e.key==='ArrowRight'){e.preventDefault();step(S.active,1)}
  if(e.key==='ArrowDown'||e.key==='ArrowLeft'){e.preventDefault();step(S.active,-1)}
  if(e.key==='PageUp'){e.preventDefault();step(S.active,5)}
  if(e.key==='PageDown'){e.preventDefault();step(S.active,-5)}});
 ['L','R'].forEach(sd=>['rx','ry','rz'].forEach((k,j)=>{
  $(sd+k).oninput=()=>{const v=+$(sd+k).value;$(sd+k+'n').textContent=v;S.seg[sd].r[j]=v;drawAll();stats();};}));
 $('save').onclick=save;$('reset').onclick=()=>loadSeg(null,0.50);
 $('prev').onclick=()=>{S.i=Math.max(0,S.i-1);load()};
 $('next').onclick=()=>{S.i=Math.min(S.uids.length-1,S.i+1);load()};}
function setActive(p){S.active=p;
 PLANES.forEach(pp=>$('im_'+pp).classList.toggle('act',pp===p));}
function step(p,d){
 const n=+$('sl_'+p).max;
 S.idx[p]=Math.max(0,Math.min(n,S.idx[p]+d));
 $('sl_'+p).value=S.idx[p];draw(p);}
function click(p,e){
 if(S.mode==='nav'){navClick(p,e);return}
 const im=$('im_'+p),cols={ax:[0,1],co:[0,2],sa:[1,2]}[p],fix={ax:2,co:1,sa:0}[p];
 const dA=S.shape[cols[0]],dB=S.shape[cols[1]];
 const ia=Math.round(e.offsetX/im.width*dA),ib=Math.round((im.height-e.offsetY)/im.height*dB);
 const c=S.seg[S.mode].c;c[cols[0]]=ia;c[cols[1]]=ib;c[fix]=S.idx[p];
 drawAll();stats();}
function navClick(p,e){
 const im=$('im_'+p),cols={ax:[0,1],co:[0,2],sa:[1,2]}[p];
 const ia=Math.round(e.offsetX/im.width*S.shape[cols[0]]);
 const ib=Math.round((im.height-e.offsetY)/im.height*S.shape[cols[1]]);
 const pos=[0,0,0];pos[cols[0]]=ia;pos[cols[1]]=ib;pos[{ax:2,co:1,sa:0}[p]]=S.idx[p];
 S.idx.ax=pos[2];S.idx.co=pos[1];S.idx.sa=pos[0];
 PLANES.forEach(pp=>{$('sl_'+pp).value=S.idx[pp]});drawAll();}
function loadSeg(d,thr){
 S.seg=d||JSON.parse(JSON.stringify(S.auto));
 S.thr=thr??0.50;$('thr').value=S.thr;$('thrn').textContent=S.thr;
 ['L','R'].forEach(sd=>{const s=S.seg[sd];
  [['rx',s.r[0]],['ry',s.r[1]],['rz',s.r[2]]].forEach(([k,v])=>{$(sd+k).value=v;$(sd+k+'n').textContent=v;});});
 drawAll();stats();}
async function load(){
 const uid=S.uids[S.i];
 const m=await (await fetch('/meta?uid='+uid)).json();
 S.shape=m.shape;S.auto=m.auto;
 S.idx={ax:m.auto.L.c[2],co:m.auto.L.c[1],sa:m.auto.L.c[0]};
 PLANES.forEach(p=>{const n=S.shape[{ax:2,co:1,sa:0}[p]];
  $('sl_'+p).max=n-1;$('sl_'+p).value=S.idx[p];});
 loadSeg(m.saved,m.saved_thr);
 $('top').innerHTML=`<b>scan ${S.i+1}/${S.uids.length}</b> uid ${uid} ${
  (S.done[uid]||[]).length?'<span style=color:#6a6>(saved)</span>':''}`;
 list();}
function draw(p){
 const uid=S.uids[S.i];$(p+'n').textContent=S.idx[p];
 $('im_'+p).src=`/slice?uid=${uid}&plane=${p}&idx=${S.idx[p]}&cmap=${S.cmap}&vmax=${S.vmax}`+
  `&L=${q('L')}&R=${q('R')}&thr=${S.thr}&_=${Date.now()%1e6}`;}
function drawAll(){PLANES.forEach(draw)}
let _st=null;
function stats(){clearTimeout(_st);_st=setTimeout(async()=>{
 const r=await (await fetch(`/stats?uid=${S.uids[S.i]}&L=${q('L')}&R=${q('R')}&thr=${S.thr}`)).json();
 $('stats').innerHTML=`cut ${r.cut} &nbsp;|&nbsp; L: ${r.L.vox} vox, mean ${r.L.mean} &nbsp;|&nbsp; `+
  `R: ${r.R.vox} vox, mean ${r.R.mean} &nbsp;|&nbsp; vol-asym ${r.vasym}`;},250);}
async function save(){
 await fetch('/save',{method:'POST',body:JSON.stringify({uid:S.uids[S.i],seg:S.seg,thr:S.thr})});
 S.done[S.uids[S.i]]=['L','R'];load();}
function list(){
 $('list').innerHTML=S.uids.map((u,k)=>`<a href=# style="color:${k===S.i?'#6cf':
  (S.done[u]||[]).length?'#6a6':'#888'}" onclick="S.i=${k};load();return false">${u}</a>`).join(' &nbsp;');}
init();
</script>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, body, ctype="application/json"):
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/": return self._send(PAGE.encode(), "text/html")
        if u.path == "/init":
            return self._send(json.dumps({"uids": UIDS, "cmaps": CMAPS, "done": saved()}).encode())
        uid = q.get("uid", "")
        if uid not in UIDS: return self._send(b"{}")
        if u.path == "/meta":
            v = vol(uid); sv, thr = None, None
            f = f"{OUTM}/{uid}.shared.json"
            if os.path.exists(f):
                d = json.load(open(f)); sv, thr = d["seg"], d["thr"]
            return self._send(json.dumps({"shape": list(v.shape), "auto": defaults(uid),
                                          "saved": sv, "saved_thr": thr}).encode())
        sides = {s: p for s in ("L", "R") if (p := parse_side(q, s))}
        thr = float(q.get("thr", 0.5))
        if u.path == "/slice":
            cmap = q.get("cmap", "gray")
            if cmap not in CMAPS: cmap = "gray"
            return self._send(slice_png(uid, q.get("plane", "ax"), int(float(q.get("idx", 0))),
                                        cmap, float(q.get("vmax", 8)), sides, thr), "image/png")
        if u.path == "/stats":
            v = vol(uid)
            segs, cut, _ = joint_seg(v, sides, thr)
            out = {"cut": round(cut, 3)}
            for s in ("L", "R"):
                m = segs[s]
                out[s] = {"vox": int(m.sum()), "mean": round(float(v[m].mean()), 3) if m.any() else 0}
            vL, vR = out["L"]["vox"], out["R"]["vox"]
            out["vasym"] = round(abs(vL - vR) / (vL + vR + 1e-9), 3)
            return self._send(json.dumps(out).encode())
        self._send(b"{}")
    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        uid = d.get("uid")
        if uid not in UIDS: return self._send(b"{}")
        v = vol(uid); thr = float(d["thr"])
        segs, cut, mxs = joint_seg(v, d["seg"], thr)
        np.savez_compressed(f"{OUTM}/{uid}.npz", maskL=segs["L"], maskR=segs["R"])
        json.dump({"seg": d["seg"], "thr": thr}, open(f"{OUTM}/{uid}.shared.json", "w"))
        rows = []
        for s in ("L", "R"):
            p, m = d["seg"][s], segs[s]
            rows.append([uid, s] + [round(float(x), 3) for x in p["c"] + p["r"] + [thr]]
                        + [int(m.sum()), round(float(v[m].mean()), 4) if m.any() else 0, round(mxs[s], 4)])
        hdr_rows = []
        if os.path.exists(OUTC):
            hdr_rows = [r for r in csv.reader(open(OUTC))][1:]
            hdr_rows = [r for r in hdr_rows if r and r[0] != uid]
        with open(OUTC, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["uid", "side", "cx", "cy", "cz", "rx", "ry", "rz", "thr", "vox", "mean", "max"])
            for r in hdr_rows + rows: w.writerow(r)
        self._send(b"{}")

print(f"{len(UIDS)} scans ({args.uids}) | SHARED-threshold protocol | http://localhost:{args.port}")
ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
