import sys

# Usage: python fix_readme.py <source_readme> <target_readme>
# Example: python fix_readme.py /path/to/GENESIS_SPECTRAL_README.md experiments/genesis/README.md
if len(sys.argv) != 3:
    print("Usage: fix_readme.py <source> <target>")
    sys.exit(1)

source = sys.argv[1]
target = sys.argv[2]

with open(source, 'r', encoding='utf-8') as f:
    content = f.read()

prefix = """## Replication via `replicate.py`

This folder contains a fully unified orchestrator to replicate all phases of testing (from 0A up to 10O). 

**Usage:**

To list all available phases, what they test, and why:
```bash
python replicate.py --list
```

To view detailed info and dependencies for a phase (e.g., 9C):
```bash
python replicate.py --info 9C
```

To run a specific phase:
```bash
python replicate.py 10O
```

To run everything in sequence:
```bash
python replicate.py all
```

"""

with open(target, 'w', encoding='utf-8') as f:
    f.write(prefix + content)

