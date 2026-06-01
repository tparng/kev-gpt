"""Kevin-on-Kria live chat — Stage 4, early.

A tiny Flask front end that runs the FP char-level GPT on the A53 and lets you
talk to Kevin over the LAN. The model is a continuation model trained only on
Kevinised TinyStories, so "chat" means: you type a seed, Kevin rambles on in
telegraphic toddler-speak. The dumbness is the point.

Tokens stream as they are generated, and each reply reports TTFT (time to first
token) and tok/s measured on the A53 — the numbers the whole project is about.

Run on the board:
    KEV_PASSWORD=secret ~/kevweb/venv/bin/python app.py
Then browse to http://<board-ip>:8080 and log in.
"""

from __future__ import annotations

import json
import os
import threading
import time

import torch
from flask import Flask, request, session, redirect, Response, render_template_string

from gpt import GPT, GPTConfig

CKPT = os.environ.get("KEV_CKPT", "ckpt.fp.pt")
PASSWORD = os.environ.get("KEV_PASSWORD", "kevin")
PORT = int(os.environ.get("KEV_PORT", "8080"))

# ---- load the model once, at startup -------------------------------------
torch.set_num_threads(os.cpu_count() or 4)
_device = "cpu"
_ck = torch.load(CKPT, map_location=_device, weights_only=False)
_model = GPT(GPTConfig(**_ck["cfg"])).to(_device)
_model.load_state_dict(_ck["model"])
_model.eval()
_meta = _ck["meta"]
_stoi = _meta["stoi"]
_itos = _meta["itos"]
_block = _model.cfg.block_size
_lock = threading.Lock()  # one CPU model, serialise requests
_SENTINEL = ""      # not in the char vocab; separates text from metrics JSON
print(f"loaded {CKPT}: {_model.num_params()/1e6:.2f}M params, "
      f"vocab {len(_stoi)}, block {_block}, iter {_ck.get('iter')}, val {_ck.get('val')}")


@torch.no_grad()
def stream_reply(prompt: str, n_tokens: int = 160, temperature: float = 0.85,
                 top_k: int = 40):
    """Yield Kevin-speak char by char, then a final SENTINEL+JSON with metrics."""
    seed = (prompt or "").strip().lower()
    ids = [_stoi[c] for c in seed if c in _stoi]
    if not ids:
        ids = [_stoi.get("\n", 0)]
    idx = torch.tensor([ids[-_block:]], dtype=torch.long)
    with _lock:
        t0 = time.perf_counter()
        ttft = None
        count = 0
        for _ in range(n_tokens):
            cond = idx[:, -_block:]
            logits, _ = _model(cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
            if ttft is None:
                ttft = time.perf_counter() - t0
            ch = _itos[str(int(nxt))]
            if ch == "\n":      # stop at a story boundary -> one utterance
                break
            count += 1
            yield ch
        total = time.perf_counter() - t0
        tok_s = count / total if total > 0 else 0.0
        yield _SENTINEL + json.dumps({
            "ttft_ms": round((ttft or total) * 1000, 1),
            "tok_s": round(tok_s, 1),
            "n": count,
        })


app = Flask(__name__)
app.secret_key = os.urandom(24)

_PAGE = """
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kevin on Kria</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;font-family:system-ui,sans-serif;background:#0d1117;color:#e6edf3;
      display:flex;flex-direction:column;height:100vh}
 header{padding:12px 16px;background:#161b22;border-bottom:1px solid #30363d;
        font-weight:600;display:flex;justify-content:space-between;align-items:center}
 header small{font-weight:400;color:#8b949e}
 #log{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:6px}
 .msg{max-width:80%;padding:9px 13px;border-radius:14px;line-height:1.45;white-space:pre-wrap}
 .you{align-self:flex-end;background:#1f6feb;color:#fff;border-bottom-right-radius:4px;margin-top:8px}
 .kev{align-self:flex-start;background:#21262d;border:1px solid #30363d;border-bottom-left-radius:4px}
 .meta{align-self:flex-start;font-size:11px;color:#8b949e;font-variant-numeric:tabular-nums;
       padding-left:6px;letter-spacing:.02em}
 .meta b{color:#3fb950;font-weight:600}
 form{display:flex;gap:8px;padding:12px 16px;background:#161b22;border-top:1px solid #30363d}
 input[type=text]{flex:1;padding:11px 13px;border-radius:10px;border:1px solid #30363d;
      background:#0d1117;color:#e6edf3;font-size:15px}
 button{padding:11px 18px;border:0;border-radius:10px;background:#238636;color:#fff;
        font-weight:600;cursor:pointer;font-size:15px}
 button:disabled{opacity:.5;cursor:default}
 .cursor::after{content:"\\258c";animation:b 1s steps(2) infinite;color:#8b949e}
 @keyframes b{50%{opacity:0}}
</style></head><body>
<header><span>🧒 Kevin <small>on Kria · A53 · {{params}}M params · fp32 · CPU</small></span>
 <a href="/logout" style="color:#8b949e;font-size:13px">log out</a></header>
<div id="log">
 <div class="msg kev">few word do trick. type thing. me finish it.</div>
</div>
<form id="f" onsubmit="return send(event)">
 <input id="p" type="text" autocomplete="off" placeholder="once upon time..." autofocus>
 <button id="b" type="submit">Send</button>
</form>
<script>
const log=document.getElementById('log'),inp=document.getElementById('p'),btn=document.getElementById('b');
function add(cls,txt){const d=document.createElement('div');d.className='msg '+cls;d.textContent=txt||'';
 log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
async function send(e){e.preventDefault();const q=inp.value.trim();if(!q)return false;
 add('you',q);inp.value='';btn.disabled=true;
 const bubble=add('kev','');bubble.classList.add('cursor');
 const meta=add('meta','');
 try{
   const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},
     body:JSON.stringify({prompt:q})});
   const reader=r.body.getReader(),dec=new TextDecoder();let buf='';
   while(true){const {done,value}=await reader.read();if(done)break;
     buf+=dec.decode(value,{stream:true});
     const i=buf.indexOf('\\u0001');
     if(i>=0){bubble.textContent=buf.slice(0,i);bubble.classList.remove('cursor');
       const m=JSON.parse(buf.slice(i+1));
       if(m.error){bubble.textContent='['+m.error+']';}
       else{meta.innerHTML='TTFT <b>'+m.ttft_ms+' ms</b> · <b>'+m.tok_s+' tok/s</b> · '+m.n+' tok · A53 fp32';}
     }else{bubble.textContent=buf;}
     log.scrollTop=log.scrollHeight;}
 }catch(err){bubble.classList.remove('cursor');bubble.textContent='[network error]';}
 btn.disabled=false;inp.focus();return false;}
</script></body></html>
"""

_LOGIN = """
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Kevin on Kria</title>
<style>body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
 background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif}
 form{background:#161b22;border:1px solid #30363d;padding:28px;border-radius:14px;width:280px;text-align:center}
 h1{font-size:20px;margin:0 0 4px}p{color:#8b949e;margin:0 0 18px;font-size:13px}
 input{width:100%;box-sizing:border-box;padding:11px;margin-bottom:12px;border-radius:9px;
 border:1px solid #30363d;background:#0d1117;color:#e6edf3;font-size:15px}
 button{width:100%;padding:11px;border:0;border-radius:9px;background:#238636;color:#fff;
 font-weight:600;font-size:15px;cursor:pointer}.err{color:#f85149;font-size:13px;margin-bottom:10px}</style>
</head><body><form method="post"><h1>🧒 Kevin on Kria</h1><p>tiny model. big claim. log in.</p>
{% if error %}<div class="err">wrong word, try again</div>{% endif %}
<input type="password" name="password" placeholder="password" autofocus>
<button type="submit">Enter</button></form></body></html>
"""


@app.before_request
def _gate():
    if request.endpoint in ("login", "static"):
        return None
    if not session.get("auth"):
        return redirect("/login")
    return None


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("password") == PASSWORD:
            session["auth"] = True
            return redirect("/")
        return render_template_string(_LOGIN, error=True)
    if session.get("auth"):
        return redirect("/")
    return render_template_string(_LOGIN, error=False)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/")
def index():
    return render_template_string(_PAGE, params=round(_model.num_params() / 1e6, 2))


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", ""))[:512]
    n = int(data.get("n", 160))
    temperature = float(data.get("temperature", 0.85))
    top_k = int(data.get("top_k", 40))

    def gen():
        try:
            for chunk in stream_reply(prompt, n, temperature, top_k):
                yield chunk
        except Exception as e:  # noqa: BLE001 - surface to the UI
            yield _SENTINEL + json.dumps({"error": str(e)})

    return Response(gen(), mimetype="text/plain; charset=utf-8",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
