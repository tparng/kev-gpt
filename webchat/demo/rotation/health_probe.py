#!/usr/bin/env python3
"""Loopback coherence probe for the active fabric model. rc 0=coherent, 2=garbage, 3=error."""
import socket, struct, json, sys
try:
    s = socket.create_connection(("127.0.0.1", 9099), timeout=15)
    b = json.dumps({"prompts": ["once upon"]}).encode()
    s.sendall(struct.pack(">I", len(b)) + b)
    h = b""
    while len(h) < 4: h += s.recv(4 - len(h))
    n = struct.unpack(">I", h)[0]; buf = b""
    while len(buf) < n: buf += s.recv(n - len(buf))
    c = json.loads(buf).get("completions", [""])[0]
    frac = sum(ch.isalpha() or ch == " " for ch in c) / max(len(c), 1)
    ok = bool(c) and frac > 0.7
    print(("OK " if ok else "GARBAGE ") + repr(c[:70]))
    sys.exit(0 if ok else 2)
except Exception as e:
    print("ERR " + repr(e)); sys.exit(3)
