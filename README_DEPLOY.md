# NARCBench Core demo — Zeabur deploy

Weekend science review (Transfer-stable, 2026-10-01): see [README.md](README.md#weekend-review).

Public walkthrough of **Qwen3-32B NARCBench-Core** multi-round transcripts (model picker API kept). **No GPU** — static + `app/server.py` only.

Suggested subdomain: **`narcbench-demo.zeabur.app`** (same pattern as `age18-kit.zeabur.app`).

After the URL is live, put it in `PUBLIC_URL.txt` (replace `PENDING`).

---

## What’s in this folder

| Path | Role |
|------|------|
| `Dockerfile` | Python 3.12-slim, expose **8080**, `CMD python3 app/server.py` |
| `zbpack.json` | `start_command` hint for Zeabur |
| `app/` | `server.py` + `viewer.html` (binds `0.0.0.0`, honors `PORT`) |
| `scripts/build_index.py` | Index builder imported by the server |
| `data/` | Prebuilt `demo_index*.json` + `models_index.json` |
| `upstream/scenarios/qwen3_32b/core/` | ~16MB official Core transcripts (100 runs / 50 pairs) |
| `upstream/scenarios/gemma2_*/core/` | Empty stubs until regen lands |

Local presenter at the **repo root** still defaults to **`:8765`** (`PORT` unset). This package defaults image `PORT=8080` for Zeabur.

---

## Option A — GitHub → Zeabur (preferred)

### 1. Create repo under `andylikescodes`

If not already created:

1. GitHub → **New repository** → name `narcbench-core-demo` (public or private).
2. From this folder (the deploy root **is** the git root for Zeabur):

```bash
cd /workspace/narcbench-core-demo/zeabur-deploy
git init -b main
git add .
git commit -m "NARCBench Core demo — Zeabur package"
git remote add origin https://github.com/andylikescodes/narcbench-core-demo.git
git push -u origin main
```

Or push only this subdirectory from the parent monorepo with `git subtree` / a dedicated remote.

### 2. Zeabur click-path (Andy / Ada)

1. Open [Zeabur Dashboard](https://dash.zeabur.com/) → sign in with Andy’s account.
2. **Create Project** (or open an existing one, e.g. next to age18-kit).
3. **Add Service** → **Git** → connect GitHub if needed → select **`andylikescodes/narcbench-core-demo`**.
4. Root directory: **`/`** (this folder is the repo root). If the monorepo was pushed instead, set root to `zeabur-deploy/`.
5. Zeabur detects the **Dockerfile** → build & deploy (Python slim image).
6. Service → **Networking / Domains** → **Generate Domain** → set subdomain  
   **`narcbench-demo`** → `https://narcbench-demo.zeabur.app`  
   (or attach a custom domain / subpath on an existing project gateway if you prefer).
7. Confirm health: `https://<domain>/api/health` → `"ok": true`, `active_model` ≈ `qwen3_32b`.
8. Paste the URL into `PUBLIC_URL.txt` and commit.

**Port:** leave Zeabur default; image listens on `PORT` (8080). Do not force a GPU plan.

---

## Option B — Upload / deploy without Git

1. Zip this folder:

```bash
cd /workspace/narcbench-core-demo
zip -r narcbench-core-demo-zeabur.zip zeabur-deploy -x '*/__pycache__/*'
```

2. Zeabur → **Add Service** → **Upload** / **Dockerfile** (UI label varies) → upload the zip or point at the Dockerfile tree.
3. Same domain step as Option A §6–8.

---

## Local smoke (this package)

```bash
cd /workspace/narcbench-core-demo/zeabur-deploy
PORT=8080 python3 app/server.py
# → http://127.0.0.1:8080/  and  /api/health
```

Or Docker:

```bash
docker build -t narcbench-core-demo .
docker run --rm -p 8080:8080 narcbench-core-demo
```

Root app (unchanged presenter):

```bash
cd /workspace/narcbench-core-demo
python3 app/server.py   # still http://127.0.0.1:8765/
```

---

## Notes

- Gemma model picker entries stay **stubs** until real `core/` runs exist; Qwen is fully available.
- No RunPod / GPU needed for this host — transcripts are baked into the image.
- Viewer shows official **`system_prompt`** from `agent_prompts.json` (Show system prompts). Round 0 = private back-channel.
- Attribution: Rose et al. NARCBench; HF `aaronrose227/narcbench`.
