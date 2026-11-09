#!/usr/bin/env python3
"""Static file server with Range support, a /sha256?path= endpoint and a /ls?path= endpoint (pod side).

python3 scripts/runpod_range_server.py <directory> [port]
Serves <directory> on the port (default 8765). Paths given to /sha256 and /ls are relative to the directory.
Used by the RunPod boot scripts so that multi-GB result archives download with resume and are verified.
"""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import re
import socketserver
import sys
import urllib.parse

ROOT_DIR = os.path.realpath(sys.argv[1]) if len(sys.argv) > 1 else os.getcwd()
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8765


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT_DIR, **kw)

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        if u.path in ("/sha256", "/ls"):
            q = urllib.parse.parse_qs(u.query).get("path", [""])[0]
            real = os.path.realpath(os.path.join(ROOT_DIR, q.lstrip("/")))
            if not real.startswith(ROOT_DIR):
                return self._json({"error": "path escapes the served directory"}, 400)
            if u.path == "/ls":
                if not os.path.isdir(real):
                    return self._json({"error": "not a directory", "path": q}, 404)
                rows = []
                for n in sorted(os.listdir(real)):
                    fp = os.path.join(real, n)
                    try:
                        st = os.stat(fp)
                        rows.append({"name": n, "dir": os.path.isdir(fp), "size": st.st_size})
                    except OSError:
                        pass
                return self._json({"path": q, "entries": rows})
            if not os.path.isfile(real):
                return self._json({"error": "not a file", "path": q}, 404)
            h = hashlib.sha256()
            with open(real, "rb") as f:
                for b in iter(lambda: f.read(1 << 24), b""):
                    h.update(b)
            return self._json({"path": q, "size": os.path.getsize(real), "sha256": h.hexdigest()})
        return super().do_GET()

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404)
            return None
        size = os.fstat(f.fileno()).st_size
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range") or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            if start >= size:
                f.close()
                self.send_error(416)
                return None
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        else:
            self.send_response(200)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        f.seek(start)
        self._left = end - start + 1
        return f

    def copyfile(self, src, dst):
        n = getattr(self, "_left", None)
        if n is None:
            return super().copyfile(src, dst)
        while n > 0:
            b = src.read(min(1 << 20, n))
            if not b:
                break
            dst.write(b)
            n -= len(b)
        self._left = None

    def log_message(self, *a):
        pass


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    S(("", PORT), H).serve_forever()
