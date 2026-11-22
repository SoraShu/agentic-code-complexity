"""Reproduce selected Appendix B excerpts without changing original evidence."""
from pathlib import Path
import hashlib
import json

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--materials',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();MATERIALS=args.materials.resolve();OUTPUT=args.output.resolve()

# One-based inclusive ranges in the archived, unabridged display materials.
# Omission markers are editorial; retained lines are copied byte-for-byte.
SELECTIONS = {
    't1-problem-excerpt.txt': ('t1-problem.txt', [(1, 1), (4, 4), (14, 18), (21, 24), (27, 30)]),
    't1-tests-excerpt.patch': ('t1-tests.patch', [(1, 3), (6, 6), (14, 17), (50, 57), (70, 71), (81, 84), (86, 101)]),
    't1-candidate-excerpt.patch': ('t1-candidate.patch', [(1, 32), (33, 33), (35, 36), (52, 56), (59, 60), (64, 71), (79, 83)]),
    't3-problem-excerpt.txt': ('t3-problem.txt', [(1, 1), (4, 11), (17, 22), (48, 49)]),
    't3-evaluation-excerpt.txt': ('t3-evaluation.txt', [(1, 1), (6, 6), (9, 9), (14, 14)]),
    't3-tests-excerpt.patch': ('t3-tests.patch', [(1, 21), (30, 31), (37, 40), (43, 51), (56, 61)]),
}

def main():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    manifest = {}
    for target, (source, ranges) in SELECTIONS.items():
        lines = (MATERIALS / source).read_text().splitlines(keepends=True)
        chunks = []
        last = 0
        for start, end in ranges:
            assert last < start <= end <= len(lines)
            if start > last + 1:
                chunks.append('[... omitted ...]\n')
            chunks.extend(lines[start-1:end])
            last = end
        if last < len(lines):
            chunks.append('[... omitted ...]\n')
        text = ''.join(chunks)
        (OUTPUT / target).write_text(text)
        manifest[target] = {'source': source,
                            'ranges_inclusive': ranges,
                            'sha256': hashlib.sha256(text.encode()).hexdigest()}
    (OUTPUT / 'compact-excerpts.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Reproduced', len(manifest), 'excerpts; no experiments run.')

if __name__ == '__main__':
    main()
