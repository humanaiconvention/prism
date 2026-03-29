import os

source = 'D:/Genesis/GENESIS_SPECTRAL_README.md'
target = 'D:/spectral_microscope_public/experiments/genesis_152m/README.md'

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

