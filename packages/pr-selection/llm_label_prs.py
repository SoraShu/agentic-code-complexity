#!/usr/bin/env python3
"""

Usage:
    export GITHUB_TOKEN=ghp_xxxx
    export LLM_API_KEY=sk-xxxx
    export LLM_BASE_URL=https://api.openai.com/v1       # OpenAI compatible API
    export LLM_MODEL=gpt-4o                               # Optional, default gpt-4o
    python3 llm_label_prs.py -i arrow_metrics.jsonl -p arrow.jsonl -k lines_changed -n 50 -o arrow_llm_labels.jsonl --merged-only
"""

import argparse
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request

from openai import OpenAI

MAX_RETRIES = 5
DIFF_MAX_CHARS = 8000

GITHUB_HEADERS = {}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_system_prompt() -> str:
    prompt_path = os.path.join(SCRIPT_DIR, "label.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def init_github_headers(token: str):
    global GITHUB_HEADERS
    GITHUB_HEADERS = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "pr-llm-labeler",
    }


def github_request(url: str) -> tuple[int, bytes, dict]:
    backoff = 1
    for attempt in range(1, MAX_RETRIES + 1):
        req = urllib.request.Request(url, headers=GITHUB_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                reset_ts = (e.headers or {}).get("X-RateLimit-Reset")
                retry_after = (e.headers or {}).get("Retry-After")
                if retry_after:
                    wait = int(retry_after) + 1
                elif reset_ts:
                    wait = max(int(reset_ts) - int(time.time()), 0) + 5
                else:
                    wait = 60
                print(f"  GitHub rate limited. Waiting {wait}s ...")
                time.sleep(wait)
                continue
            if e.code >= 500:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
        except (
            urllib.error.URLError,
            OSError,
            TimeoutError,
            http.client.IncompleteRead,
        ) as e:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(f"GitHub request failed after {MAX_RETRIES} retries: {url}")


def fetch_github_json(url: str):
    _, body, _ = github_request(url)
    return json.loads(body)


def fetch_github_text(url: str) -> str:
    try:
        _, body, _ = github_request(url)
        return body.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return ""
        raise


def fetch_diff(diff_url: str) -> str:
    text = fetch_github_text(diff_url)
    if len(text) > DIFF_MAX_CHARS:
        return text[:DIFF_MAX_CHARS] + f"\n\n... [truncated, {len(text)} chars total]"
    return text


def fetch_comments(comments_url: str) -> list[dict]:
    """Fetch comments and extract relevant information."""
    data = fetch_github_json(comments_url)
    if not isinstance(data, list):
        return []
    result = []
    for c in data:
        result.append(
            {
                "user": c.get("user", {}).get("login", ""),
                "body": c.get("body", ""),
                "created_at": c.get("created_at", ""),
            }
        )
    return result


def call_llm(
    client: OpenAI, model: str, system_prompt: str, user_content: str
) -> dict:
    """Call OpenAI compatible API and return the parsed JSON response."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    return {"label": -1, "reason": f"Unparseable response: {text[:200]}"}


def build_user_prompt(title: str, body: str, diff: str, comments: list[dict]) -> str:
    parts = [f"# PR Title\n{title}"]

    if body:
        b = body if len(body) <= 2000 else body[:2000] + "\n... [truncated]"
        parts.append(f"# PR Description\n{b}")

    if diff:
        parts.append(f"# Diff\n```\n{diff}\n```")

    if comments:
        comment_lines = []
        for c in comments:
            comment_lines.append(
                f"- **{c['user']}** ({c['created_at']}): {c['body'][:500]}"
            )
        parts.append("# Comments\n" + "\n".join(comment_lines))

    parts.append(
        "\nBased on the above PR information, is this PR interesting for a coding benchmark? Reply with JSON only."
    )
    return "\n\n".join(parts)


def load_prs(pr_file: str) -> dict:
    """Load PRs from a JSONL file and return a mapping from PR number to PR data."""
    mapping = {}
    with open(pr_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pr = json.loads(line)
                mapping[pr["number"]] = pr
    return mapping


def load_done(output_file: str) -> set:
    """Load completed PR numbers from the output file."""
    done = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["number"])
    return done


def main():
    parser = argparse.ArgumentParser(description="Automatically label PRs using LLM")
    parser.add_argument("-i", "--input", required=True, help="metrics jsonl file")
    parser.add_argument("-p", "--prs", required=True, help="original PR jsonl file")
    parser.add_argument("-k", "--key", required=True, help="sorting metric name")
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=50,
        help="number of items to process (default: 50)",
    )
    parser.add_argument("-o", "--output", required=True, help="output file for labels")
    parser.add_argument(
        "--merged-only", action="store_true", help="process only merged PRs"
    )
    args = parser.parse_args()

    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        print("Error: Please set GITHUB_TOKEN", file=sys.stderr)
        sys.exit(1)
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        print("Error: Please set LLM_API_KEY", file=sys.stderr)
        sys.exit(1)
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL", "gpt-4o")

    system_prompt = load_system_prompt()
    init_github_headers(github_token)
    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=MAX_RETRIES)

    # Load metrics
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records or args.key not in records[0]:
        print(f"No records or key '{args.key}' not found.")
        return

    # Load original PR data
    pr_map = load_prs(args.prs)

    records.sort(key=lambda r: r.get(args.key, 0), reverse=True)

    if args.merged_only:
        records = [r for r in records if pr_map.get(r["number"], {}).get("merged_at")]

    tail = records[: args.number]

    done = load_done(args.output)
    remaining = [r for r in tail if r["number"] not in done]

    print(f"Total {len(tail)} PRs, {len(done)} labeled, {len(remaining)} remaining")
    print(f"LLM: {base_url}  model: {model}")
    print()

    with open(args.output, "a", encoding="utf-8") as out:
        for idx, r in enumerate(remaining, 1):
            number = r["number"]
            pr = pr_map.get(number, {})
            title = pr.get("title", "")
            body = pr.get("body", "") or ""
            diff_url = pr.get("diff_url", "")
            comments_href = pr.get("_links", {}).get("comments", {}).get("href", "")
            html_url = pr.get("html_url", "")

            print(f"[{idx}/{len(remaining)}] PR #{number}: {title}")

            # Fetch diff
            print("  Fetching diff ...", end=" ", flush=True)
            diff = fetch_diff(diff_url) if diff_url else ""
            print(f"{len(diff)} chars")

            # Fetch comments
            print("  Fetching comments ...", end=" ", flush=True)
            comments = fetch_comments(comments_href) if comments_href else []
            print(f"{len(comments)} comments")

            # Build prompt and call LLM
            user_prompt = build_user_prompt(title, body, diff, comments)
            print(f"  Calling LLM ({len(user_prompt)} chars) ...", end=" ", flush=True)
            result = call_llm(client, model, system_prompt, user_prompt)
            label = result.get("label", -1)
            reason = result.get("reason", "")
            print(f"label={label}  reason={reason}")

            record = {
                **r,
                "html_url": html_url,
                "label": label,
                "reason": reason,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print()

    print("Done.")


if __name__ == "__main__":
    main()
