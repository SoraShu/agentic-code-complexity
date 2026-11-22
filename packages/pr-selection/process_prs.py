#!/usr/bin/env python3
"""从 jsonl 提取 PR 指标，输出到另一个 jsonl。

用法:
    export GITHUB_TOKEN=ghp_xxxx
    python3 process_prs.py -i arrow.jsonl -o arrow_metrics.jsonl

无需第三方依赖，仅使用 Python 标准库。
"""

import argparse
import json
import os
import http.client
import sys
import time
import urllib.error
import urllib.request

MAX_RETRIES = 5

HEADERS = {}


def init_headers(token: str):
    global HEADERS
    HEADERS = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "pr-metrics-collector",
    }


def request_with_retry(url: str) -> tuple[int, bytes, dict]:
    """发送 GET 请求，遇到 rate limit 或网络错误时自动重试。

    返回 (status_code, body_bytes, response_headers)。
    """
    backoff = 1
    for attempt in range(1, MAX_RETRIES + 1):
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
                hdrs = dict(resp.headers)
                return resp.status, body, hdrs

        except urllib.error.HTTPError as e:
            status = e.code
            hdrs = dict(e.headers) if e.headers else {}

            # Rate limit
            if status in (403, 429):
                reset_ts = hdrs.get("X-RateLimit-Reset")
                retry_after = hdrs.get("Retry-After")
                if retry_after:
                    wait = int(retry_after) + 1
                elif reset_ts:
                    wait = max(int(reset_ts) - int(time.time()), 0) + 5
                else:
                    wait = 60
                print(f"  Rate limited ({status}). Waiting {wait}s ...")
                time.sleep(wait)
                continue

            # Server errors
            if status >= 500:
                print(
                    f"  Server error {status}, retry {attempt}/{MAX_RETRIES} in {backoff}s ..."
                )
                time.sleep(backoff)
                backoff *= 2
                continue

            # Other client errors — no point retrying
            raise

        except (
            urllib.error.URLError,
            OSError,
            TimeoutError,
            http.client.IncompleteRead,
        ) as e:
            if attempt == MAX_RETRIES:
                raise
            print(
                f"  Network error: {e}, retry {attempt}/{MAX_RETRIES} in {backoff}s ..."
            )
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {url}")


def fetch_json(url: str) -> dict:
    _, body, _ = request_with_retry(url)
    return json.loads(body)


def fetch_bytes(url: str) -> bytes:
    _, body, _ = request_with_retry(url)
    return body


def parse_patch(patch_text: str):
    """Historical screening count; also excludes content starting +++ or ---.

    Preserve this rule to reproduce the saved 100-PR cohort. It is distinct
    from the regression feature that includes all added/deleted content lines.
    """
    files_changed = 0
    lines_added = 0
    lines_deleted = 0

    for line in patch_text.split("\n"):
        if line.startswith("diff --git "):
            files_changed += 1
        elif line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_deleted += 1

    return files_changed, lines_added + lines_deleted


def load_done(output_file: str) -> set:
    """读取已有输出文件，返回已处理的 PR number 集合（用于断点续传）。"""
    done = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["number"])
    return done


def parse_args():
    parser = argparse.ArgumentParser(description="从 jsonl 提取 PR 指标")
    parser.add_argument("-i", "--input", required=True, help="输入 jsonl 文件")
    parser.add_argument("-o", "--output", required=True, help="输出 jsonl 文件")
    return parser.parse_args()


def main():
    args = parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("Error: 请设置环境变量 GITHUB_TOKEN", file=sys.stderr)
        sys.exit(1)

    init_headers(token)

    # 读取所有 PR
    with open(args.input, "r", encoding="utf-8") as f:
        prs = [json.loads(line) for line in f if line.strip()]

    total = len(prs)
    print(f"共 {total} 个 PR")

    done = load_done(args.output)
    if done:
        print(f"已处理 {len(done)} 个，从断点继续")

    with open(args.output, "a", encoding="utf-8") as out:
        for i, pr in enumerate(prs, 1):
            number = pr["number"]
            if number in done:
                continue

            print(f"[{i}/{total}] PR #{number} ...", end=" ", flush=True)

            # 请求 1: PR detail API → num_commits, num_discussions
            detail = fetch_json(pr["url"])
            num_commits = detail.get("commits", 0)
            num_discussions = detail.get("comments", 0)

            # 请求 2: patch → patch_size, files_changed, lines_changed
            try:
                patch_bytes = fetch_bytes(pr["patch_url"])
                patch_size = len(patch_bytes)
                patch_text = patch_bytes.decode("utf-8", errors="replace")
                files_changed, lines_changed = parse_patch(patch_text)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print("SKIP (patch 404)", end=" ")
                    patch_size = 0
                    files_changed = 0
                    lines_changed = 0
                else:
                    raise

            record = {
                "number": number,
                "lines_changed": lines_changed,
                "files_changed": files_changed,
                "patch_size": patch_size,
                "num_commits": num_commits,
                "num_discussions": num_discussions,
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            print(
                f"OK  (lines={lines_changed}, files={files_changed}, "
                f"patch={patch_size}B, commits={num_commits}, discussions={num_discussions})"
            )

    print("Done.")


if __name__ == "__main__":
    main()
