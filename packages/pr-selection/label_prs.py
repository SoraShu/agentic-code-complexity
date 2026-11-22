#!/usr/bin/env python3
"""遍历 metrics tail，逐个打开 PR 网页，交互式标注感兴趣(1)/不感兴趣(0)。

用法:
    python3 label_prs.py -i arrow_metrics.jsonl -p arrow.jsonl -k lines_changed -n 20 -o labels.jsonl

流程：对 tail 中每个 PR，自动打开浏览器，等待用户输入 1/0/s(skip)/q(quit)。
支持断点续传：已标注的 PR 自动跳过。
"""

import argparse
import json
import os
import subprocess
import sys


def load_pr_info(pr_file: str) -> tuple[dict, set]:
    """从原始 PR jsonl 构建 number -> html_url 映射，以及已 merge 的 number 集合。"""
    url_map = {}
    merged = set()
    with open(pr_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pr = json.loads(line)
                url_map[pr["number"]] = pr["html_url"]
                if pr.get("merged_at"):
                    merged.add(pr["number"])
    return url_map, merged


def load_done(output_file: str) -> set:
    done = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["number"])
    return done


def open_url(url: str):
    if sys.platform == "darwin":
        subprocess.Popen(
            ["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    elif sys.platform.startswith("linux"):
        subprocess.Popen(
            ["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    else:
        subprocess.Popen(
            ["start", url],
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main():
    parser = argparse.ArgumentParser(description="交互式标注 PR")
    parser.add_argument("-i", "--input", required=True, help="metrics jsonl 文件")
    parser.add_argument(
        "-p", "--prs", required=True, help="原始 PR jsonl 文件（含 html_url）"
    )
    parser.add_argument("-k", "--key", required=True, help="排序的指标名")
    parser.add_argument(
        "-n", "--number", type=int, default=10, help="tail 条数 (默认: 10)"
    )
    parser.add_argument("-o", "--output", required=True, help="标注结果输出文件")
    parser.add_argument(
        "--merged-only", action="store_true", help="仅包含已 merge 的 PR"
    )
    args = parser.parse_args()

    # 加载 metrics
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("No records.")
        return

    if args.key not in records[0]:
        print(f"Key '{args.key}' not found. Available: {list(records[0].keys())}")
        return

    records.sort(key=lambda r: r.get(args.key, 0), reverse=True)

    # 加载 PR URL 映射与 merge 状态
    url_map, merged_set = load_pr_info(args.prs)

    if args.merged_only:
        records = [r for r in records if r["number"] in merged_set]

    tail = records[: args.number]

    # 断点续传
    done = load_done(args.output)
    if done:
        print(f"已标注 {len(done)} 个，跳过")

    remaining = [r for r in tail if r["number"] not in done]
    if not remaining:
        print("所有 PR 已标注完毕。")
        return

    print(f"共 {len(remaining)} 个待标注  (1=感兴趣, 0=不感兴趣, s=跳过, q=退出)")
    print()

    with open(args.output, "a", encoding="utf-8") as out:
        for idx, r in enumerate(remaining, 1):
            number = r["number"]
            url = url_map.get(number, "")

            # 显示信息
            print(f"[{idx}/{len(remaining)}] PR #{number}  {args.key}={r[args.key]}")
            for k, v in r.items():
                if k != "number":
                    print(f"  {k}: {v}")
            if url:
                print(f"  url: {url}")
                open_url(url)
            else:
                print("  url: (未找到)")

            # 等待输入
            while True:
                choice = input("  标注 (1/0/s/q): ").strip().lower()
                if choice in ("1", "0"):
                    label = int(choice)
                    record = {**r, "html_url": url, "label": label}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"  -> {'感兴趣' if label else '不感兴趣'}")
                    break
                elif choice == "s":
                    print("  -> 跳过")
                    break
                elif choice == "q":
                    print("退出。")
                    return
                else:
                    print("  无效输入，请输入 1, 0, s 或 q")
            print()

    print("标注完毕。")


if __name__ == "__main__":
    main()
