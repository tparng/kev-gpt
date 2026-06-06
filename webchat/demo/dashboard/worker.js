/*
 * worker.js — Cloudflare Worker for the live load dashboard.
 *
 * Two jobs (the telemetry plane, kept off the box so it survives the death):
 *   1. POST /ingest   <- the i7 server pushes a ~1 Hz telemetry JSON blob here.
 *                        We stash the latest in a Durable Object so fan-out and
 *                        last-known/RIP rendering are Cloudflare's problem, not
 *                        the box's. Optional bearer-token auth on this route.
 *   2. GET  /         -> serves the single-file dashboard (DASHBOARD_HTML inlined
 *                        at deploy by a tiny build step, or fetched from KV/asset).
 *      GET  /latest   -> the most recent telemetry record as JSON (the page polls
 *                        this ~1 Hz; no WebSocket needed for the dashboard).
 *
 * "Survives death": /latest always returns the last record plus an `age_s` and a
 * `rip` flag the page uses to switch to the RIP/last-known state when the feed
 * stops (no fresh POST for > RIP_AFTER_S seconds). Stats keep rendering; the page
 * makes clear the box is gone.
 *
 * Deployment is the human's job: `wrangler deploy`. See README. The dashboard
 * HTML is read from the bundled asset `dashboard.html` (wrangler [assets] or a
 * text import) — here we import it as a module text binding for a no-build path.
 */

import DASHBOARD_HTML from "./dashboard.html";

const RIP_AFTER_S = 5;       // no POST for this long => box presumed dead
const HISTORY = 300;         // keep ~5 min of 1 Hz records for the charts

export class TelemetryStore {
  constructor(state, env) { this.state = state; this.env = env; }

  async fetch(request) {
    const url = new URL(request.url);
    if (request.method === "POST" && url.pathname === "/ingest") {
      const rec = await request.json();
      rec._recv = Date.now();
      await this.state.storage.put("latest", rec);
      let hist = (await this.state.storage.get("history")) || [];
      hist.push(rec);
      if (hist.length > HISTORY) hist = hist.slice(hist.length - HISTORY);
      await this.state.storage.put("history", hist);
      return new Response("ok");
    }
    if (url.pathname === "/latest") {
      const latest = (await this.state.storage.get("latest")) || null;
      const hist = (await this.state.storage.get("history")) || [];
      let age_s = null, rip = true;
      if (latest && latest._recv) {
        age_s = (Date.now() - latest._recv) / 1000;
        rip = age_s > RIP_AFTER_S;
      }
      return Response.json({ latest, age_s, rip, history: hist });
    }
    return new Response("not found", { status: 404 });
  }
}

function stub(env) {
  const id = env.TELEMETRY.idFromName("singleton");
  return env.TELEMETRY.get(id);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/ingest" && request.method === "POST") {
      // optional shared-secret auth so randoms can't poison the dashboard
      if (env.INGEST_TOKEN) {
        const auth = request.headers.get("Authorization") || "";
        if (auth !== `Bearer ${env.INGEST_TOKEN}`) {
          return new Response("unauthorized", { status: 401 });
        }
      }
      return stub(env).fetch(new Request("https://do/ingest", {
        method: "POST", body: await request.text(),
        headers: { "Content-Type": "application/json" },
      }));
    }

    if (url.pathname === "/latest") {
      return stub(env).fetch(new Request("https://do/latest"));
    }

    if (url.pathname === "/" || url.pathname === "/index.html") {
      return new Response(DASHBOARD_HTML, {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      });
    }
    return new Response("not found", { status: 404 });
  },
};
