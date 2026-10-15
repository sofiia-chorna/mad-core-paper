import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://archive.materialscloud.org/api/records/{rid}/draft"
FIRST = ["README.md", "LICENSES.md", "example.py", "dataset_metadata.csv", "madcore_0-2pow18.extxyz.gz"]
PREVIEW = "README.md"


def token():
    path = os.path.expanduser("~/.mcloud_token")
    if (os.stat(path).st_mode & 0o077) != 0:
        raise SystemExit(f"{path} must be readable only by you (chmod 600)")
    return open(path).read().strip()


def call(method, url, tok, data=None, content_type="application/json", timeout=3600, length=None):
    headers = {"Authorization": f"Bearer {tok}", "Content-Type": content_type}
    if length is not None:
        headers["Content-Length"] = str(length)
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")[:500]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(64 << 20), b""):
            h.update(block)
    return h.hexdigest()


def remote_files(rid, tok):
    status, body = call("GET", API.format(rid=rid) + "/files", tok)
    if status != 200:
        raise SystemExit(f"cannot list draft files: {status} {body}")
    return {e["key"]: e for e in body.get("entries", [])}


def upload_one(rid, tok, path, name, state):
    base = API.format(rid=rid) + "/files"
    local = md5(path)
    existing = remote_files(rid, tok).get(name)
    if existing and existing.get("status") == "completed" and existing.get("checksum") == f"md5:{local}":
        print(f"skip {name}: already complete, checksum matches", flush=True)
        state["uploaded_files"][name] = {"md5": local, "bytes": os.path.getsize(path), "status": "verified"}
        return
    if existing:
        print(f"replace {name}: remote {existing.get('status')} {existing.get('checksum')}", flush=True)
        status, body = call("DELETE", f"{base}/{name}", tok)
        if status not in (200, 204):
            raise SystemExit(f"delete {name}: {status} {body}")
    status, body = call("POST", base, tok, json.dumps([{"key": name}]).encode())
    if status not in (200, 201):
        raise SystemExit(f"init {name}: {status} {body}")
    t0 = time.time()
    with open(path, "rb") as fh:
        status, body = call("PUT", f"{base}/{name}/content", tok, fh, "application/octet-stream",
                            length=os.path.getsize(path))
    if status not in (200, 201):
        raise SystemExit(f"content {name}: {status} {body}")
    status, body = call("POST", f"{base}/{name}/commit", tok)
    if status not in (200, 201):
        raise SystemExit(f"commit {name}: {status} {body}")
    remote = body.get("checksum")
    ok = remote == f"md5:{local}" and body.get("size") == os.path.getsize(path)
    state["uploaded_files"][name] = {"md5": local, "bytes": os.path.getsize(path), "status": "verified" if ok else f"CHECKSUM MISMATCH {remote}"}
    print(f"{'ok' if ok else 'MISMATCH'} {name} {os.path.getsize(path)} B in {time.time() - t0:.0f}s", flush=True)
    if not ok:
        raise SystemExit(f"checksum mismatch on {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    parser.add_argument("--files", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--first-only", action="store_true")
    args = parser.parse_args()
    tok = token()
    status, draft = call("GET", API.format(rid=args.record), tok)
    if status != 200:
        raise SystemExit(f"cannot read draft: {status} {draft}")
    print(f"draft: {draft['metadata'].get('title')!r} published={draft.get('is_published')} status={draft.get('status')}", flush=True)
    if draft.get("is_published"):
        raise SystemExit("record is published; refusing to touch it")
    state = json.load(open(args.state))
    state.update({"draft_record_id": args.record, "status": "in_progress"})
    state.setdefault("uploaded_files", {})
    if isinstance(state["uploaded_files"], list):
        state["uploaded_files"] = {}
    names = FIRST if args.first_only else FIRST + sorted(n for n in os.listdir(args.files) if n not in FIRST)
    for stale in sorted(set(remote_files(args.record, tok)) - set(os.listdir(args.files))):
        status, body = call("DELETE", f"{API.format(rid=args.record)}/files/{stale}", tok)
        print(f"{'removed' if status in (200, 204) else 'FAILED to remove'} stale remote file {stale} ({status})", flush=True)
        state["uploaded_files"].pop(stale, None)
    try:
        for name in names:
            upload_one(args.record, tok, os.path.join(args.files, name), name, state)
    finally:
        json.dump(state, open(args.state, "w"), indent=1)
    status, draft = call("GET", API.format(rid=args.record), tok)
    if status == 200 and draft["files"].get("default_preview") != PREVIEW:
        draft["files"]["default_preview"] = PREVIEW
        status, body = call("PUT", API.format(rid=args.record), tok, json.dumps(draft).encode())
        print(f"default_preview set to {PREVIEW}: {status}", flush=True)

    remote = remote_files(args.record, tok)
    local = set(os.listdir(args.files))
    complete = {k for k, e in remote.items() if e.get("status") == "completed" and e.get("checksum") == f"md5:{md5(os.path.join(args.files, k))}"} if not args.first_only else set()
    if not args.first_only:
        state["status"] = "complete" if complete == local and set(remote) == local else "incomplete"
        state["remote_files"] = len(remote)
        json.dump(state, open(args.state, "w"), indent=1)
        print(f"remote {len(remote)} files, verified {len(complete)}/{len(local)}; state {state['status']}")


if __name__ == "__main__":
    main()
