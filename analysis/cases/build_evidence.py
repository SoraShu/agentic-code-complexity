"""Extract existing case evidence; no agent, repository test, or model execution."""
import ast
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import textwrap

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--private-root',type=Path,required=True)
parser.add_argument('--tasks',type=Path,required=True,help='Retrieved Verified task directories containing tests/config.json and instruction.md')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
ROOT = args.private_root.resolve()/'workspace'
HERE = args.output.resolve()
RUN = json.loads((ROOT/'private-config.json').read_text())['case_run_id']
RAW = ROOT/'evalagent-play/test/.msbenchRun'/RUN
TASKS = args.tasks.resolve()
PAPER = HERE
CASES = {
    "T1": ("django__django-16502", [62, 63, 75, 77, 78, 79], 2),
    "T2": ("sympy__sympy-20916", [15, 27, 28, 31, 32, 39, 44, 45], 3),
    "T3": ("django__django-15098", [11, 13, 19, 24, 30, 32, 34, 45, 47, 49, 52, 68, 69], 2),
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def printable(text):
    # Preserve Unicode semantics explicitly in Python/log excerpts.
    return "".join(c if ord(c) < 128 else
                   (f"\\u{ord(c):04x}" if ord(c) <= 0xffff else f"\\U{ord(c):08x}")
                   for c in text)


def main():
    PAPER.mkdir(parents=True,exist_ok=False)
    csvs = sorted((ROOT / "localdataset/results/swe-bench-verified").glob("report-*.csv"))
    outcomes = collections.defaultdict(list)
    protected = list(csvs)
    for f in csvs:
        for row in csv.DictReader(f.open()):
            outcomes[row["InstanceId"]].append((row["RunId"], row["Resolved"]))
    cases, snippets = {}, {}
    adopted=ROOT/'experiments/regression/results/verified-boolean-outcomes-2026-11-09/verified'
    dataset=json.loads((adopted/'dataset.json').read_text())
    quantitative={row['task_id']:row for row in dataset['rows']}
    oof={row['task_id']:row for row in csv.DictReader((adopted/'oof_predictions.csv').open())}
    metric_rows=[]
    for label, (instance, records, success) in CASES.items():
        folder = RAW / f"sweb.eval.x86_64.{instance}-output/output"
        q=quantitative[instance];pred=oof[instance]
        metric_rows.append(dict(case=label,instance_id=instance,**dict(zip(dataset['feature_names'],q['features'])),
                               complexity=q['complexity'],lasso=float(pred['lasso']),xgboost=float(pred['xgboost']),outer_fold=int(pred['outer_fold'])))
        files = {name: folder / name for name in
                 ["trajectories/trajectory.json", "changes.patch", "100_eval.log", "eval.json", "metadata.json"]}
        config_path = TASKS / instance / "tests/config.json"
        protected += [*files.values(), config_path, TASKS / instance / "instruction.md"]
        cfg = json.loads(config_path.read_text())
        for key,suffix in [('problem_statement','problem.txt'),('patch','reference.patch'),('test_patch','tests.patch')]:
            (PAPER/f'{label.lower()}-{suffix}').write_text(printable(cfg[key]).rstrip()+'\n')
        (PAPER/f'{label.lower()}-candidate.patch').write_text(printable(files['changes.patch'].read_text()).rstrip()+'\n')
        trajectory = json.loads(files["trajectories/trajectory.json"].read_text())
        grade = json.loads(files["eval.json"].read_text())[instance]
        meta = json.loads(files["metadata.json"].read_text())
        patch = files["changes.patch"].read_text()
        log = files["100_eval.log"].read_text()
        assert cfg["problem_statement"] == meta["problem_statement"]
        assert grade["resolved"] is False and grade["patch_successfully_applied"] is True
        assert len(outcomes[instance]) == 26
        assert sum(v == "True" for _, v in outcomes[instance]) == success
        assert (RUN, "False") in outcomes[instance]
        assert f"git checkout {cfg['base_commit']} " in log
        selected = []
        for n in records:
            r = trajectory[n - 1]
            selected.append(dict(record=n, turn=r["turn"], tool=r["tool"],
                                 action=r["action"], observation=r["observation"]))
        cases[label] = dict(instance_id=instance, run_id=RUN, base_commit=cfg["base_commit"],
            successes=success, attempts=26, complexity=1-success/26,
            trajectory_records=len(trajectory), selected_records=selected,
            evaluation=grade, sources={k: str(v) for k, v in files.items()})

        def excerpt(name, text, source, locator):
            rendered = printable(text.strip("\n")) + "\n"
            (PAPER / name).write_text(rendered)
            snippets[name] = dict(source=source, locator=locator, original_text=text,
                                  rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest())

        if label == "T1":
            code = json.loads(trajectory[62]["action"])["new_str"]
            code = textwrap.dedent(code[code.index("    def write(self, data):"):])
            assert 'self.bytes_sent = len(data)' in code
            excerpt("t1-write.txt", code, "trajectory", "R63.action.new_str, write method")
            tp = cfg["test_patch"]
            head = tp[tp.index('+        # The body is not returned in a HEAD response.'):]
            lines = [s[1:] for s in head.splitlines() if s.startswith('+')]
            excerpt("t1-test-contract.txt", textwrap.dedent("\n".join(lines)),
                    "benchmark test_patch", "HEAD-response assertions")
            observation = trajectory[77]["observation"]
            text = observation[observation.index("Content-Length: 45"):observation.index("GET Response:")]
            excerpt("t1-local-output.txt", text, "trajectory", "R78.observation, HEAD response tail")
            start = log.index("FAIL: test_no_body_returned_for_head_requests")
            end = log.index("AssertionError: True is not False", start) + len("AssertionError: True is not False")
            excerpt("t1-evaluation.txt", log[start:end], "100_eval.log", f"lines {log[:start].count(chr(10))+1}-{log[:end].count(chr(10))+1}")
            assert "Ran 34 tests" in trajectory[76]["observation"]
            assert "OK (skipped=2)" in trajectory[76]["observation"]
        elif label == "T2":
            obs = trajectory[26]["observation"]
            excerpt("t2-local-inputs.txt", "\n".join(obs.splitlines()[:4]),
                    "trajectory", "R27.observation, first four lines")
            regex = "\n".join(s for s in cfg["patch"].splitlines()
                              if s.startswith(("-_name_with_digits_p", "+_name_with_digits_p")))
            excerpt("t2-reference-regex.txt", regex, "benchmark patch", "regex replacement")
            start = log.index("Traceback (most recent call last):", log.index("test_super_sub F"))
            end = log.index("AssertionError", start) + len("AssertionError")
            excerpt("t2-evaluation.txt", log[start:end], "100_eval.log", f"lines {log[:start].count(chr(10))+1}-{log[:end].count(chr(10))+1}")
            assert "142 passed" in trajectory[43]["observation"]
            assert "2 passed, 1 xfailed" in trajectory[44]["observation"]
            assert re.findall(r"^diff --git a/(\S+)", patch, re.M) == ["sympy/printing/pretty/pretty_symbology.py"]
        else:
            first = json.loads(trajectory[10]["action"])["new_str"]
            assert first in cfg["patch"]
            excerpt("t3-initial-regex.txt", first, "trajectory", "R11.action.new_str")
            final = json.loads(trajectory[44]["action"])["new_str"]
            final = final[final.index("language_code_prefix_re ="):]
            excerpt("t3-final-regex.txt", final, "trajectory", "R45.action.new_str, prefix regex")
            assert "self.assertEqual(g('/de-simple-page/'), 'de')" in trajectory[29]["action"]
            assert "self.assertIsNone(g('/de-simple-page/'))" in json.loads(trajectory[48]["action"])["new_str"]
            assert "test_page_with_dash" in trajectory[33]["observation"]
            assert "Ran 19 tests" in trajectory[51]["observation"]
            assert "Ran 20 tests" in trajectory[67]["observation"]
            assert "Ran 104 tests" in trajectory[68]["observation"]
            start = log.index("FAIL: test_get_language_from_path_real")
            end = log.index("FAILED (failures=2, errors=1)", start) + len("FAILED (failures=2, errors=1)")
            excerpt("t3-evaluation.txt", log[start:end], "100_eval.log", f"lines {log[:start].count(chr(10))+1}-{log[:end].count(chr(10))+1}")
            assert "ModuleNotFoundError: No module named 'tests'" in log
            # Static checks of the recorded expression, not a repository test rerun.
            tree = ast.parse(final)
            pattern = ast.literal_eval(tree.body[0].value.args[0])
            rx = re.compile(pattern, re.IGNORECASE)
            examples = {x: bool(rx.match('/' + x + '/')) for x in
                        ["en-latn-us", "de-ch-1901", "i-mingo", "kl-tunumiit", "nan-hani-tw"]}
            assert examples == {"en-latn-us": True, "de-ch-1901": False, "i-mingo": False,
                                "kl-tunumiit": True, "nan-hani-tw": True}
            cases[label]["static_regex_matches"] = examples
    checksums = {str(p): sha(p) for p in protected}
    manifest = HERE / "source-sha256.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == checksums, "Original evidence changed"
    else:
        manifest.write_text(json.dumps(checksums, indent=2) + "\n")
    (HERE / "evidence.json").write_text(json.dumps(
        dict(cases=cases, snippets=snippets), ensure_ascii=False, indent=2) + "\n")
    summary = dict(run_id=RUN, cases=3, protected_sources=len(checksums),
                   original_sources_unchanged=True, excerpt_files=len(snippets),
                   model_calls=0, repository_test_runs=0)
    (HERE / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (PAPER/'case-metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metric_rows[0]));writer.writeheader();writer.writerows(metric_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
