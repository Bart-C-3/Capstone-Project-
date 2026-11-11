"""
repo_sync.py - send scripts and results to the project's GitHub repo
====================================================================
Repo: https://github.com/Bart-C-3/Capstone-Project-

Every script in this project calls push_run() at the end, so the script
itself and its small result files land in the repo automatically:

    scripts/<script name>.py
    scripts/repo_sync.py
    results/<run name>/<result files>

Design notes
  * Uses only the Python standard library (no pip installs, no git needed).
  * The token is read from the GITHUB_TOKEN environment variable. It is never
    written into a file, never printed, and never sent anywhere except GitHub.
  * It can never crash a training run: if there is no token, no internet, or
    GitHub rejects the upload, it prints a warning and your files stay local.
  * Files whose content is unchanged are skipped, so re-runs don't spam commits.
  * Model weights (.pt/.pth/.bin/.safetensors/.ckpt) and anything over
    MAX_FILE_MB are NOT uploaded unless you explicitly allow weights.
  * The repo is PUBLIC, so only send files you are happy to share.

One-time setup (Windows PowerShell)
  1. GitHub -> Settings -> Developer settings -> Personal access tokens ->
     Fine-grained tokens -> Generate new token.
       Repository access : Only select repositories -> Capstone-Project-
       Permissions       : Repository permissions -> Contents: Read and write
  2. Set it for the current terminal:
         $env:GITHUB_TOKEN = "github_pat_xxxxxxxx"
     or permanently (then restart VS Code):
         setx GITHUB_TOKEN "github_pat_xxxxxxxx"
  Never paste the token into a script, a chat or a commit.

Usage inside a script
    from repo_sync import push_run
    push_run(__file__, "lstm-hybrid", [out_dir / "test_results.json"],
             message="LSTM detector run")
"""

import base64
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OWNER = "Bart-C-3"
REPO = "Capstone-Project-"
DEFAULT_BRANCH = "main"
MAX_FILE_MB = 50
WEIGHT_EXTENSIONS = {".pt", ".pth", ".bin", ".safetensors", ".ckpt"}
BLOCKED_NAMES = {".env"}


# ─────────────────────────────────────────────
# Low-level GitHub API helpers
# ─────────────────────────────────────────────
def _api_base():
    return os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")


def _request(method, path, token, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"{_api_base()}{path}", data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "capstone-repo-sync")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
        return resp.status, (json.loads(raw) if raw else {})


def _blob_sha(content: bytes) -> str:
    """The SHA GitHub reports for a file (git blob hash), computed locally."""
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def _explain_http_error(e: urllib.error.HTTPError) -> str:
    try:
        detail = json.loads(e.read().decode("utf-8")).get("message", "")
    except Exception:
        detail = ""
    hints = {
        401: "the token was rejected. Check GITHUB_TOKEN is correct and has not expired.",
        403: "the token lacks permission. It needs 'Contents: Read and write' on "
             f"{OWNER}/{REPO} (or you hit a rate limit).",
        404: "repo not found, or the token has no access to it. Fine-grained tokens "
             "report 'no access' as 404. If the repo is brand new and empty, create "
             "a README on github.com first, then re-run.",
        409: "conflict with a newer version of the file. Re-run to retry.",
        422: "GitHub refused the request (often a wrong branch name; set GITHUB_BRANCH).",
    }
    return f"HTTP {e.code} {detail}".strip() + " - " + hints.get(e.code, "unexpected error.")


def _default_branch(token):
    override = os.environ.get("GITHUB_BRANCH", "").strip()
    if override:
        return override
    try:
        _, info = _request("GET", f"/repos/{OWNER}/{REPO}", token)
        return info.get("default_branch") or DEFAULT_BRANCH
    except Exception:
        return DEFAULT_BRANCH


def _upload_one(token, branch, local: Path, repo_path: str, message: str) -> str:
    content = local.read_bytes()
    quoted = urllib.parse.quote(repo_path, safe="/")

    sha = None
    try:
        _, existing = _request(
            "GET",
            f"/repos/{OWNER}/{REPO}/contents/{quoted}?ref={urllib.parse.quote(branch)}",
            token)
        if isinstance(existing, dict):
            sha = existing.get("sha")
    except urllib.error.HTTPError as e:
        if e.code != 404:       # 404 just means "new file"
            raise

    if sha and sha == _blob_sha(content):
        return "unchanged"

    body = {"message": message,
            "content": base64.b64encode(content).decode("ascii"),
            "branch": branch}
    if sha:
        body["sha"] = sha
    _request("PUT", f"/repos/{OWNER}/{REPO}/contents/{quoted}", token, body)
    return "updated" if sha else "created"


# ─────────────────────────────────────────────
# Public function
# ─────────────────────────────────────────────
def push_run(script_file, run_name, result_files=(), message=None, allow_weights=False):
    """
    Upload the calling script, this helper, and the given result files.

    script_file  : pass __file__ from the calling script
    run_name     : folder name under results/ (e.g. "lstm-hybrid")
    result_files : iterable of local file paths to upload
    message      : commit message (a default is used if None)
    allow_weights: set True to also upload model weight files (<= MAX_FILE_MB)

    Never raises: any problem is printed as a warning.
    """
    try:
        _push_run(script_file, run_name, result_files, message, allow_weights)
    except urllib.error.HTTPError as e:
        print(f"[repo_sync] Upload failed: {_explain_http_error(e)}")
        print("[repo_sync] Your files are saved locally; nothing was lost.")
    except Exception as e:
        print(f"[repo_sync] Upload failed ({type(e).__name__}: {e}).")
        print("[repo_sync] Your files are saved locally; nothing was lost.")


def _push_run(script_file, run_name, result_files, message, allow_weights):
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("[repo_sync] GITHUB_TOKEN is not set, so nothing was uploaded.")
        print("[repo_sync] Setup steps are at the top of repo_sync.py. "
              "Your files are saved locally.")
        return

    script_path = Path(script_file).resolve()
    run_name = str(run_name).strip().strip("/\\").replace(" ", "-").replace("\\", "/")

    items = [(script_path, f"scripts/{script_path.name}")]
    helper = Path(__file__).resolve()
    if helper != script_path:
        items.append((helper, f"scripts/{helper.name}"))
    for f in result_files:
        p = Path(f).resolve()
        items.append((p, f"results/{run_name}/{p.name}"))

    # Safety filters
    to_send, skipped = [], []
    for local, repo_path in items:
        if not local.is_file():
            skipped.append((repo_path, "file not found"))
        elif local.name in BLOCKED_NAMES:
            skipped.append((repo_path, "blocked file name"))
        elif local.suffix.lower() in WEIGHT_EXTENSIONS and not allow_weights:
            skipped.append((repo_path, "model weights are not uploaded by default"))
        elif local.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            skipped.append((repo_path, f"larger than {MAX_FILE_MB} MB"))
        else:
            to_send.append((local, repo_path))

    if not to_send:
        print("[repo_sync] Nothing to upload.")
        for repo_path, why in skipped:
            print(f"  skipped   {repo_path}  ({why})")
        return

    branch = _default_branch(token)
    message = message or f"Add results for {run_name}"
    print(f"[repo_sync] Uploading to github.com/{OWNER}/{REPO} (branch '{branch}') ...")

    counts = {"created": 0, "updated": 0, "unchanged": 0}
    for local, repo_path in to_send:
        status = _upload_one(token, branch, local, repo_path, message)
        counts[status] += 1
        print(f"  {status:<9} {repo_path}")
    for repo_path, why in skipped:
        print(f"  skipped   {repo_path}  ({why})")

    print(f"[repo_sync] Done: {counts['created']} created, {counts['updated']} updated, "
          f"{counts['unchanged']} unchanged, {len(skipped)} skipped.")
    print(f"[repo_sync] https://github.com/{OWNER}/{REPO}/tree/{branch}")
