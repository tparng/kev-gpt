# Putting Kevin online: chat.mikeayles.com (+ the dashboard)

Two public endpoints, **two different mechanisms** — this trips people up:

| endpoint | what it is | how it ships |
|---|---|---|
| `chat.mikeayles.com` | the chat UI **+** WebSocket | **cloudflared tunnel** → the i7 server |
| dashboard (e.g. `dash.mikeayles.com` or `*.workers.dev`) | the load dashboard | **Cloudflare Worker** (`wrangler deploy`) |

The chat is a long-lived WebSocket server talking to the FPGA over GigE, so it
**cannot** be a Worker — it lives behind a tunnel. Only the dashboard is a Worker.
`mikeayles.com` is already on Cloudflare (nameservers `kyrie/kay.ns.cloudflare.com`),
so both DNS records can be created from the CLI.

Steps marked **(you)** need an interactive Cloudflare browser login — run them
yourself (in this session, prefix with `!`, e.g. `!cloudflared tunnel login`).

---

## A. The chat at chat.mikeayles.com (cloudflared tunnel)

Run these on the **serving box** (the i7 laptop that runs the server).

```bash
# 1. install cloudflared
#    Windows:  winget install Cloudflare.cloudflared
#    macOS:    brew install cloudflared
#    Linux:    see https://pkg.cloudflare.com  (cloudflared deb/rpm)

# 2. (you) authenticate — opens a browser, pick the mikeayles.com zone
cloudflared tunnel login

# 3. create the tunnel (prints a <TUNNEL_ID> and writes a creds .json)
cloudflared tunnel create kevin-chat

# 4. point chat.mikeayles.com at it (creates the proxied CNAME on Cloudflare)
cloudflared tunnel route dns kevin-chat chat.mikeayles.com

# 5. config: copy deploy/cloudflared-config.example.yml to ~/.cloudflared/config.yml
#    and fill in <TUNNEL_ID> + the credentials-file path from step 3.

# 6. start the chat server (serves client.html + the WS on one port)
python -m webchat.demo.server --backend stub --port 8090
#    real-fabric path instead of stub: --backend pl --lanes 128 --fclk 200e6
#    (needs the TcpPLBackend adapter to the A53 daemon — still TODO; stub is
#     telegraphic gibberish but exercises the whole plumbing + telemetry)

# 7. run the tunnel (separate terminal / a service)
cloudflared tunnel run kevin-chat
```

Then open **https://chat.mikeayles.com** — the page loads from the tunnel and the
client auto-connects to `wss://chat.mikeayles.com` (its default is
`proto://location.host`, no config needed). The server's `process_request`
distinguishes a browser GET (serves the page) from a WS upgrade (handshake).

Caveats from the PRD worth checking under real load:
- WebSockets over Tunnel work by default; if Cloudflare's bot-fight / DDoS
  mitigation throttles a traffic spike, relax protection on the `chat` route.
- Pin `cloudflared` and the server to separate cores so the tunnel doesn't
  steal cycles from batch assembly (keep the death fabric-bound, not box-bound).

---

## B. The dashboard (Cloudflare Worker, wrangler)

```bash
cd webchat/demo/dashboard
npm i -g wrangler            # or: npx wrangler ...
# (you) authenticate
wrangler login
# deploy the Worker (bundles dashboard.html as a text module — no build step)
wrangler deploy
# set the ingest secret the server signs telemetry POSTs with
wrangler secret put INGEST_TOKEN
```

Heads-up: the dashboard Worker uses a **Durable Object** (holds latest record +
rolling history so it fans out from the edge and survives the box dying). Durable
Objects require the **Workers Paid plan** ($5/mo). If you'd rather stay on the
free plan, the DO can be swapped for Workers KV (last record only, no per-second
history) — say the word and it's a small worker.js change.

Point the server at the deployed Worker so it pushes 1 Hz telemetry:

```bash
python -m webchat.demo.server --backend stub --port 8090 \
  --push-url https://kevin-kria-dashboard.<your-subdomain>.workers.dev/ingest \
  --push-token <the-same-INGEST_TOKEN>
```

Optional custom domain for the dashboard (`dash.mikeayles.com`): add a route in
`wrangler.toml` or map it in the Cloudflare dashboard → Workers → Custom Domains.

---

## What I (Claude) can vs can't do here

- **Can, offline:** install `wrangler` locally, write/adjust `worker.js` /
  `wrangler.toml` / the cloudflared config, dry-run everything against the stub,
  wire the real-fabric `TcpPLBackend`.
- **Can't:** `cloudflared tunnel login` / `wrangler login` (interactive browser
  auth on your Cloudflare account), and I won't create DNS records or deploy to
  your domain without you driving the auth. Those are the **(you)** steps above.
