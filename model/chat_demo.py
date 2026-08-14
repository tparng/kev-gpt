"""Interactive chat demo for the mamba2 checkpoints (doc 8 — "see how it talks").

Runs the model on CPU/GPU with the recurrent state held across turns, so the
multi-turn memory (the point of the branch) is live: tell it your name, chat,
ask later. Speaks the rawchat register the corpus taught it:
``user: <msg> kevin: <reply>``.

    python -m model.chat_demo                          # serves http://localhost:8017
    python -m model.chat_demo data/ckpt.mamba2.rawchat3.pt --cli
    python -m model.chat_demo --device cpu --port 9000

Not the fabric path — this is the reference model in torch, for auditioning
quality, not speed.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F

from .mamba2 import Mamba2, Mamba2Config
from .train import pick_device

DEFAULT_CKPT = "data/ckpt.mamba2.bpe1024.pt"
USER_MARK = "user:"
KEVIN_MARK = "kevin:"
_STORY = re.compile(r'(^|[.!?]["\']?\s+)([A-Z])')


def load(path: str, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    assert ck.get("arch") == "mamba2", f"{path} is not a mamba2 checkpoint"
    model = Mamba2(Mamba2Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["meta"], ck.get("iter"), ck.get("val")


class Codec:
    """Encode/decode for either the BPE or the char meta."""

    def __init__(self, meta: dict):
        self.bpe = bool(meta.get("bpe"))
        if self.bpe:
            from tokenizers import Tokenizer
            self.tok = Tokenizer.from_file(meta["tokenizer_file"])
        else:
            self.stoi, itos = meta["stoi"], meta["itos"]
            self.itos = {int(k): v for k, v in itos.items()}

    def encode(self, text: str) -> list[int]:
        if self.bpe:
            return self.tok.encode(text).ids
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids: list[int]) -> str:
        if self.bpe:
            return self.tok.decode(ids)
        return "".join(self.itos.get(i, "?") for i in ids)


class ChatSession:
    """One conversation: the model's recurrent state plus turn bookkeeping."""

    def __init__(self, model: Mamba2, codec: Codec, device: str):
        self.model, self.codec, self.device = model, codec, device
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.states = self.model.alloc_state(1, self.device)
        self.logits = None
        self.fed_any = False        # anything in the state yet
        self.user_mark_fed = False  # model already emitted "user:" itself
        self.turns = 0

    def _feed(self, ids: list[int]):
        for t in ids:
            idx = torch.tensor([t], dtype=torch.long, device=self.device)
            self.logits = self.model.step(idx, self.states)
        self.fed_any = self.fed_any or bool(ids)

    @torch.no_grad()
    def stream(self, msg: str, temperature=0.8, top_k=40, max_new=110):
        """Yield reply text chunks. Caller holds no lock; we do."""
        with self.lock:
            msg = " ".join(msg.strip().split()).lower()
            # corpus user turns always end with punctuation; without it the
            # model doesn't register the turn boundary and keeps writing the
            # user's turn itself
            if msg and msg[-1] not in ".!?":
                msg += "."
            if self.user_mark_fed:
                prefix = " " + msg + " kevin:"
            else:
                prefix = ("" if not self.fed_any else " ") + "user: " + msg + " kevin:"
            self.user_mark_fed = False
            self._feed(self.codec.encode(prefix))

            out_ids: list[int] = []
            sent = ""
            t0 = time.perf_counter()
            for _ in range(max_new):
                lg = self.logits / max(temperature, 1e-6)
                if top_k:
                    v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
                    lg[lg < v[:, [-1]]] = -float("inf")
                nxt = int(torch.multinomial(F.softmax(lg, dim=-1), 1).item())
                self._feed([nxt])
                out_ids.append(nxt)
                text = self.codec.decode(out_ids)

                # end of reply: the model starts the next user turn, restates
                # its own speaker tag, opens a new story line, or drifts into
                # TinyStories register (capital opening a sentence — dialogue
                # is all-lowercase)
                stops = [(p, is_user) for mark, is_user in
                         ((USER_MARK, True), (KEVIN_MARK, False))
                         if (p := text.find(mark)) >= 0]
                if (nl := text.find("\n")) >= 0:
                    stops.append((nl, False))
                if m := _STORY.search(text):
                    stops.append((m.start(2), False))
                if stops:
                    cut, self.user_mark_fed = min(stops)
                    done = True
                else:
                    cut, done = len(text), False

                clean = text[:cut].lstrip()
                # hold back a partial speaker-tag match at the tail so we
                # never stream half a marker to the client
                if not done:
                    for mark in (USER_MARK, KEVIN_MARK):
                        for k in range(min(len(mark) - 1, len(clean)), 0, -1):
                            if clean.endswith(mark[:k]):
                                clean = clean[:-k]
                                break
                if clean.startswith(sent):
                    delta = clean[len(sent):]
                else:  # tokenizer rewrote the prefix (rare); resend from scratch
                    yield {"replace": clean}
                    sent, delta = clean, ""
                if delta:
                    sent = clean
                    yield {"text": delta}
                if done:
                    break
            self.turns += 1
            dt = time.perf_counter() - t0
            yield {"done": True, "tokens": len(out_ids),
                   "tok_s": round(len(out_ids) / dt, 1) if dt > 0 else None}


PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>kevin (mamba2)</title>
<style>
  :root { --bg:#101312; --panel:#181c1a; --ink:#d8e0da; --dim:#7c8781;
          --user:#8fb8e8; --kev:#8fd8a8; --line:#252b28; }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--ink); height:100dvh; display:flex;
         flex-direction:column; font:15px/1.5 ui-monospace,Menlo,Consolas,monospace; }
  header { padding:10px 16px; border-bottom:1px solid var(--line); display:flex;
           gap:14px; align-items:baseline; flex-wrap:wrap; }
  header b { color:var(--kev); font-weight:600; }
  header small, #stat { color:var(--dim); font-size:12px; }
  #log { flex:1; overflow-y:auto; padding:18px 16px; }
  .msg { max-width:52em; margin:0 auto 12px; white-space:pre-wrap; word-break:break-word; }
  .msg .who { font-weight:600; }
  .msg.u .who { color:var(--user); } .msg.k .who { color:var(--kev); }
  .msg .meta { color:var(--dim); font-size:11px; margin-left:8px; }
  .cursor { display:inline-block; width:.6em; background:var(--kev); opacity:.7;
            animation:bl 1s steps(1) infinite; }
  @keyframes bl { 50% { opacity:0; } }
  form { display:flex; gap:8px; padding:12px 16px; border-top:1px solid var(--line);
         background:var(--panel); }
  form > div { display:flex; gap:8px; max-width:52em; margin:0 auto; flex:1; }
  input[type=text] { flex:1; background:var(--bg); color:var(--ink); border:1px solid
         var(--line); border-radius:6px; padding:9px 12px; font:inherit; outline:none; }
  input[type=text]:focus { border-color:var(--kev); }
  button { background:var(--kev); color:#0c1410; border:0; border-radius:6px;
           padding:9px 16px; font:inherit; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.4; cursor:default; }
  button.ghost { background:transparent; color:var(--dim); border:1px solid var(--line); }
  #knobs { padding:6px 16px 10px; background:var(--panel); }
  #knobs > div { max-width:52em; margin:0 auto; display:flex; gap:18px; color:var(--dim);
                 font-size:12px; align-items:center; flex-wrap:wrap; }
  #knobs input[type=range] { width:110px; accent-color:var(--kev); vertical-align:middle; }
</style>
<header><b>kevin</b> <small id="info">loading…</small> <span id="stat"></span></header>
<div id="log"></div>
<form id="f"><div>
  <input id="box" type="text" autocomplete="off" spellcheck="false"
         placeholder="say something to kevin…" autofocus>
  <button id="send">send</button>
  <button id="reset" class="ghost" type="button" title="wipe the recurrent state">forget</button>
</div></form>
<div id="knobs"><div>
  <label>temp <input id="temp" type="range" min="0.1" max="1.3" step="0.05" value="0.8">
    <span id="tv">0.8</span></label>
  <label>top-k <input id="topk" type="range" min="1" max="200" step="1" value="40">
    <span id="kv">40</span></label>
  <label>max tokens <input id="maxn" type="range" min="20" max="400" step="10" value="110">
    <span id="nv">110</span></label>
</div></div>
<script>
const log = document.getElementById('log'), box = document.getElementById('box');
const f = document.getElementById('f'), send = document.getElementById('send');
const stat = document.getElementById('stat');
for (const [r, s] of [['temp','tv'], ['topk','kv'], ['maxn','nv']]) {
  const el = document.getElementById(r);
  el.oninput = () => document.getElementById(s).textContent = el.value;
}
fetch('/info').then(r => r.json()).then(d => {
  document.getElementById('info').textContent =
    `${d.ckpt} · ${d.params_m}M params · val ${d.val} · ${d.device}` +
    (d.bpe ? ` · bpe-${d.vocab}` : ' · char');
});
function add(who, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.innerHTML = `<span class="who">${who}:</span> <span class="body"></span>`;
  log.appendChild(div); log.scrollTop = log.scrollHeight;
  return div;
}
f.onsubmit = async (e) => {
  e.preventDefault();
  const msg = box.value.trim();
  if (!msg || send.disabled) return;
  box.value = ''; send.disabled = true;
  add('you', 'u').querySelector('.body').textContent = msg;
  const div = add('kevin', 'k'), body = div.querySelector('.body');
  const cur = document.createElement('span'); cur.className = 'cursor';
  cur.textContent = '\\u00a0'; div.appendChild(cur);
  try {
    const r = await fetch('/chat', { method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ msg,
        temperature: +document.getElementById('temp').value,
        top_k: +document.getElementById('topk').value,
        max_new: +document.getElementById('maxn').value }) });
    const rd = r.body.getReader(), dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const {value, done} = await rd.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let i;
      while ((i = buf.indexOf('\\n')) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.text) body.textContent += ev.text;
        if (ev.replace !== undefined) body.textContent = ev.replace;
        if (ev.done) {
          const m = document.createElement('span');
          m.className = 'meta';
          m.textContent = `${ev.tokens} tok${ev.tok_s ? ' · ' + ev.tok_s + ' tok/s' : ''}`;
          div.appendChild(m);
        }
        log.scrollTop = log.scrollHeight;
      }
    }
  } catch (err) { body.textContent += ' [error: ' + err + ']'; }
  cur.remove();
  if (!body.textContent.trim()) body.textContent = '…';
  send.disabled = false; box.focus();
};
document.getElementById('reset').onclick = async () => {
  await fetch('/reset', {method: 'POST'});
  log.innerHTML = ''; stat.textContent = 'state wiped';
  setTimeout(() => stat.textContent = '', 1500); box.focus();
};
</script>
"""


def make_handler(session: ChatSession, info: dict):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype="text/html; charset=utf-8"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE)
            elif self.path == "/info":
                self._send(200, json.dumps(info), "application/json")
            else:
                self._send(404, "not found")

        def do_POST(self):
            if self.path == "/reset":
                with session.lock:
                    session.reset()
                self._send(200, "{}", "application/json")
                return
            if self.path != "/chat":
                self._send(404, "not found")
                return
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            msg = (req.get("msg") or "").strip()
            if not msg:
                self._send(400, '{"error":"empty"}', "application/json")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                for ev in session.stream(
                        msg,
                        temperature=float(req.get("temperature", 0.8)),
                        top_k=int(req.get("top_k", 40)),
                        max_new=min(int(req.get("max_new", 110)), 1000)):
                    self.wfile.write((json.dumps(ev) + "\n").encode())
                    self.wfile.flush()
            except BrokenPipeError:
                pass

    return H


def run_cli(session: ChatSession):
    print("type a message ('/reset' wipes the state, ctrl-d quits)\n")
    while True:
        try:
            msg = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not msg:
            continue
        if msg == "/reset":
            with session.lock:
                session.reset()
            print("(state wiped)")
            continue
        print("kevin: ", end="", flush=True)
        for ev in session.stream(msg):
            if ev.get("text"):
                print(ev["text"], end="", flush=True)
            if ev.get("done"):
                print(f"\n  ({ev['tokens']} tok, {ev['tok_s']} tok/s)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="model.chat_demo",
                                description="Chat with a mamba2 checkpoint.")
    p.add_argument("ckpt", nargs="?", default=DEFAULT_CKPT)
    p.add_argument("--device", default="auto")
    p.add_argument("--port", type=int, default=8017)
    p.add_argument("--cli", action="store_true", help="terminal REPL, no server")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    model, meta, it, val = load(args.ckpt, device)
    codec = Codec(meta)
    session = ChatSession(model, codec, device)
    info = {
        "ckpt": args.ckpt, "iter": it, "val": round(val, 4) if val else None,
        "params_m": round(model.num_params() / 1e6, 2), "device": device,
        "bpe": codec.bpe, "vocab": meta.get("vocab_size"),
        "state_kb": round(model.state_bytes(4) / 1024, 1),
    }
    print(f"# {args.ckpt}  iter {it}  val {val:.4f}  "
          f"{info['params_m']}M params  {device}"
          f"  ({'bpe-' + str(info['vocab']) if codec.bpe else 'char'})")

    if args.cli:
        run_cli(session)
        return
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session, info))
    print(f"# chat at http://localhost:{args.port}  (ctrl-c to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
