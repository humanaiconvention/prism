# Model Provenance in PRISM

> Integration design doc for the `prism.provenance` submodule introduced
> in PRISM 1.2.0.  Audience: contributors and users who want to add
> "where did this model come from?" alongside PRISM's geometry/NLA scans.

---

## 1. The technique

A **Model Provenance Kit (MPK) fingerprint** is a set of five weight-level
statistics extracted from a transformer that, in aggregate, identify the
model's lineage with high recall.  Cisco released MPK on **2026-05-04**:

- Blog: <https://blogs.cisco.com/ai/model-provenance-kit>
- Code (Apache-2.0): <https://github.com/cisco-ai-defense/model-provenance-kit>
- Reference data (CC BY 4.0): <https://huggingface.co/datasets/cisco-ai/model-provenance-kit>

### The five signals

| Signal | What it measures | Survives |
|---|---|---|
| **EAS** Embedding Anchor Similarity | Geometric relationships between token embeddings. | Re-tokenisation, head replacement. |
| **END** Embedding Norm Distribution | Distribution of per-token embedding magnitudes (encodes word-frequency structure). | Fine-tuning. |
| **NLF** Norm Layer Fingerprint | Statistics of normalisation-layer weights. | Most fine-tuning — these layers move very little. |
| **LEP** Layer Energy Profile | Normalised energy curve across network depth. | LoRA / adapter fine-tuning. |
| **WVC** Weight-Value Cosine | Cosine similarity between corresponding layers' weight tensors. | Identical-architecture comparisons; degrades with quantisation. |

These combine into a composite score in `[0.0, 1.0]`.  MPK's documented
threshold is **0.70**.  Cisco reports **100% recall and 100% specificity
on a 111-pair benchmark** (4 misclassifications on extreme architectural
transformations).

### Hard limits — read carefully

1. **Statistical, not cryptographic.**  MPK's README is explicit: the
   output is "strong evidence … but not absolute proof."  Do not treat
   a positive result as a signature.
2. **Cannot disambiguate identical-template fine-tunes.**  When two
   models share an architecture, MPK cannot tell whether the weights
   were copied or independently trained.
3. **Coverage gaps.**  MPK's reference database covers ~150 base models
   across ~45 families.  Brand-new architectures (e.g. `gemma-4`,
   released after MPK 1.0.0 was cut) may not be fingerprinted yet.
4. **MPK is new.**  Version 1.0.0 shipped on 2026-05-04.  Treat its
   ground-truth claims with the deference that maturity earns.

PRISM surfaces these caveats by setting `not_cryptographic=True` on every
:class:`ProvenanceResult` and including the flag verbatim in the
audit-dict serialisation.

---

## 2. PRISM integration surface

```python
from prism.provenance import (
    scan_model_provenance,         # database lookup, top-k matches
    compare_models,                 # pairwise comparison
    ProvenanceResult,
    ProvenanceMatch,
    ProvenanceSignals,
    MPKBackend,                     # real backend (lazy provenancekit)
    mock_compare, mock_scan,        # offline mock for tests / CI
    DEFAULT_PROVENANCE_THRESHOLD,   # 0.70, from MPK's calibration
)
```

### Pairwise comparison

```python
result = compare_models("haic-gemma4-v42", "google/gemma-4-e2b-it")
# result.is_match         : bool
# result.composite_score  : float in [0, 1]
# result.top_match.signals: ProvenanceSignals (EAS/END/NLF/LEP/WVC)
# result.method           : "mpk" (or "mock" under test)
# result.not_cryptographic: True (always — surfaced in audit dicts)
```

### Database scan

```python
result = scan_model_provenance("haic-gemma4-v42", top_k=3)
# result.matches: top-k ProvenanceMatch sorted by composite_score desc.
```

### Audit serialisation

```python
result.as_audit_dict()
# {
#     "model": "haic-gemma4-v42",
#     "method": "mpk",
#     "method_version": "mpk-1.0.0",
#     "threshold": 0.70,
#     "is_match": True,
#     "not_cryptographic": True,        # ← cannot be silently dropped
#     "composite_score": 0.91,
#     "top_match_asset": "google/gemma-4-e2b-it",
#     "top_match_family": "gemma-4",
#     "top_match_signals": {"eas": 0.90, "end": 0.85, "nlf": 0.95, ...},
#     "n_matches": 1,
#     "metadata": {"mode": "compare"},
# }
```

---

## 3. Backends

PRISM ships two:

| Backend | When to use |
|---|---|
| `MPKBackend` | Default.  Lazily requires `pip install provenancekit`.  Downloads the ~908 MB Parquet fingerprint dataset on first call.  Produces `method="mpk"` results. |
| Mock (`mock_compare`, `mock_scan`) | Tests, CI, demo notebooks where the 908 MB dataset is a non-starter.  Deterministic — same inputs always return the same output.  Produces `method="mock"` results. |

You can also inject your own backend for tests:

```python
from prism.provenance import MPKBackend

class FakeScanner:
    def compare(self, a, b): ...
    def scan(self, model_id, *, top_k=None, threshold=None): ...

backend = MPKBackend(scanner=FakeScanner())
result = scan_model_provenance("foo/bar", backend=backend)
```

This is how `tests/test_provenance.py` exercises the MPK adapter without
installing MPK itself.

---

## 4. When to use provenance vs geometry vs NLA

| Question | Tool |
|---|---|
| "Will this layer survive 4-bit quantisation?" | `prism.geometry.scan_model_geometry` |
| "How did fine-tuning reshape the residual manifold?" | `scan_model_geometry` across checkpoints |
| "What is this layer thinking on this prompt?" | `prism.nla` (with the confabulation caveat) |
| "Did this model actually descend from the base it claims?" | `prism.provenance.compare_models` |
| "Given an unknown model, what's the closest known base?" | `prism.provenance.scan_model_provenance` |

Provenance is the only one of these four that's about *identity*.  The
other three describe properties of the model in front of you; provenance
relates that model back to a catalogue of known lineages.

The signals are complementary: a quantisation-hostile geometry alongside
a strong provenance match to a known base tells you the geometry shift
came from fine-tuning *of that base*, not from a different base
masquerading.

---

## 5. The HAIC connection (see also: `D:/humanai-convention/HAIC_PROVENANCE.md`)

The HumanAI Convention's value proposition is *training data provenance* —
verifiable, consented, Merkle-auditable contributions.  MPK is *model
weight provenance* — given a trained artifact, does it derive from what
the producer claims?

These two questions stack:

```
   consented training data (HAIC Merkle receipt)
                │
                ▼
        fine-tune run
                │
                ▼
   resulting adapter (MPK fingerprint says: derives from google/gemma-4-e2b-it)
                │
                ▼
        audit log entry binds both
```

PRISM 1.2.0 supplies the model-side primitive.  The HAIC improvement
pipeline (`tools/improvement_pipeline.py` in the monorepo) is the
consumer: when a freshly fine-tuned adapter is promoted, the pipeline
calls `compare_models(new_adapter, claimed_base)`, attaches the
`as_audit_dict()` output to the promotion event, and refuses to promote
if `is_match` is `False`.

---

## 6. Versioning and cost

* **PRISM:** introduced in 1.2.0.  Back-compatible — no PRISM call site
  outside this submodule needs to know it exists.
* **MPK:** 1.0.0, released 2026-05-04.  Pin a version in your own
  `pyproject.toml` rather than relying on `>=`.
* **Dataset:** ~908 MB on first use.  Cache under
  `~/.cache/huggingface/datasets/cisco-ai___model-provenance-kit/` by
  default; override via `MPKBackend(cache_dir=...)`.
* **Compute:** CPU-only.  Architectural matches resolve in
  milliseconds; weight-level analysis on small/medium models runs in
  seconds.  Models >20 GB are streamed.

---

## 7. Failure modes you will see

| Symptom | Cause | What to do |
|---|---|---|
| `ImportError: 'provenancekit' package` | Default backend constructed without injection and MPK not installed. | `pip install provenancekit` or use the mock backend. |
| `composite_score < 0.70` on a known fine-tune | MPK doesn't yet fingerprint that family. | Note it in the audit log; don't fail the promotion automatically — operator judgement call. |
| Two `is_match=True` results to different bases | Architectures share enough structure that the signals can't disambiguate. | Use PRISM's geometry signal to choose between them. |
| `composite_score == 1.0` when models are not actually identical | Mock backend; check `result.method`. | Switch to `MPKBackend()`. |

---

## 8. License

MPK is Apache-2.0; its reference dataset is CC BY 4.0.  PRISM stays
under CC BY 4.0.  No code from MPK is vendored — only its public Python
API surface is consumed.
