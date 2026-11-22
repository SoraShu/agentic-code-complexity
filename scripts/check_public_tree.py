# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = []
# ///

"""Check the publication tree without staging files or making a commit."""
from pathlib import Path
import argparse,re,subprocess
ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN=('archive/','paper-assets/','provenance/','packages/arrow-builder/','packages/harbor-agent/',
           'data/benchmarks/','data/pro-upstream/','data/workspace/','analysis/historical/')
SECRET=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?!ecdsa-sha2|ssh-ed25519)[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')


def main(require_no_commits=False):
    files=set(filter(None,subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')))
    for name in sorted(files):
        p=ROOT/name
        assert not name.startswith(FORBIDDEN),name
        assert not p.is_symlink(),name
        assert not {'work','.venv','private','.private','chat_log','chatlogs','.aws','.copilot'}.intersection(Path(name).parts),name
        assert p.suffix not in {'.gz','.zip','.pdf','.png','.pem','.key'},name
        assert not re.search(r'(?:report-.*\.csv|sources\.jsonl\.gz|trajectory\.json)$',name),name
        assert not SECRET.search(p.read_bytes()),f'Credential-shaped content: {name}'
    # Internal code must not survive in reachable public history either.
    if require_no_commits:
        assert not subprocess.check_output(['git','rev-list','--all'],cwd=ROOT).strip(), 'Unexpected commits'
    history=subprocess.check_output(['git','log','--all','--pretty=format:','--name-only'],cwd=ROOT,text=True)
    assert not any(name.startswith(FORBIDDEN) for name in history.splitlines() if name),'Excluded paths survive in Git history'
    print(f'Checked {len(files)} publication files; no internal construction code, raw dataset bundles or duplicate paper assets.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--require-no-commits',action='store_true')
    main(p.parse_args().require_no_commits)
