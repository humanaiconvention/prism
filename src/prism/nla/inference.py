"""HTTP client for a remote NLA inference server.

PRISM does not host NLA weights. Instead it talks to an SGLang server
running the AR (Activation Reconstructor) — that's the serving model
described in kitft/natural_language_autoencoders's ``nla_inference.py``.
The AR is invoked with ``input_embeds`` (the activation tensor is injected
directly into the residual stream rather than tokenised), and it returns:

  * the AV (Activation Verbalizer) generated explanation text, and
  * an FVE score over the AR's reconstruction of the activation.

This module wraps that contract behind :class:`NLAExplainer` so the
PRISM-side caller never touches HTTP plumbing. The transport is
pluggable — pass a ``transport=`` callable (or any object with a
``post(url, json)`` method) to inject a fake during testing.

Choice of HTTP-only path
------------------------
A direct in-process PyTorch path would also work — load the AR + AV
locally and run forward passes — but it would force every PRISM user to
pull tens of GB of weights and a GPU. The HTTP path matches kitft's
released serving design, keeps weights on the model server, and keeps
the wire payload tiny (one ``float32`` vector per call). Users who need
in-process inference can subclass :class:`NLAExplainer` and override
:meth:`NLAExplainer._call_remote`.

License
-------
The kitft inference contract is Apache-2.0; PRISM remains CC BY 4.0.
Only the wire format is reproduced here, not any code from that repo.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Union

import numpy as np

from .registry import NLACheckpoint, get_checkpoint
from .types import NLAExplanation


class _HasPost(Protocol):
    def post(self, url: str, json: Dict[str, Any]) -> Any: ...


TransportLike = Union[str, Callable[[Dict[str, Any]], Dict[str, Any]], _HasPost]


def _default_http_transport(url: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Build a tiny ``requests``-based transport bound to *url*.

    Imported lazily so PRISM does not gain a hard dependency on ``requests``
    just for users who only ever touch the mock or pass their own transport.
    """

    def _post(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import requests  # noqa: WPS433 — intentional lazy import
        except ImportError as exc:
            raise ImportError(
                "prism.nla's default HTTP transport requires the 'requests' package. "
                "Install it with `pip install requests`, or pass a custom transport "
                "via NLAExplainer(transport=...)."
            ) from exc
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    return _post


class NLAExplainer:
    """Client for a remote NLA inference server.

    Args:
        checkpoint: The :class:`NLACheckpoint` describing which target model
            and layer the server is configured for. Used for metadata and
            shape validation; the server itself is the source of truth.
        server_url: HTTP endpoint to POST activation requests to. Required
            when *transport* is ``None`` (the default).
        transport: Optional transport override. May be either a callable
            ``payload_dict -> response_dict`` (preferred for tests) or an
            object exposing ``post(url, json)`` returning a ``.json()``-able
            response. When ``None`` and *server_url* is set, a default
            ``requests``-backed transport is constructed lazily.
    """

    def __init__(
        self,
        checkpoint: NLACheckpoint,
        *,
        server_url: Optional[str] = None,
        transport: Optional[TransportLike] = None,
    ) -> None:
        if server_url is None and transport is None:
            raise ValueError(
                "NLAExplainer requires either server_url= (HTTP endpoint) or "
                "transport= (callable/object for tests)."
            )
        self.checkpoint = checkpoint
        self.server_url = server_url
        self._transport = transport
        self.d_model = checkpoint.d_model
        self.layer_idx = checkpoint.target_layer
        self.model_id = checkpoint.target_model

    # ------------------------------------------------------------------ factory

    @classmethod
    def from_pretrained(
        cls,
        nla_id: str,
        *,
        server_url: Optional[str] = None,
        transport: Optional[TransportLike] = None,
    ) -> "NLAExplainer":
        """Look up *nla_id* in :mod:`prism.nla.registry` and build an explainer.

        Raises:
            KeyError: if *nla_id* is not in the registry.
            ValueError: if neither *server_url* nor *transport* is supplied.
        """
        ckpt = get_checkpoint(nla_id)
        if ckpt is None:
            raise KeyError(
                f"No NLA checkpoint registered under {nla_id!r}. "
                "Call prism.nla.list_checkpoints() to see registered IDs, "
                "or use prism.nla.mock_explainer() for offline testing."
            )
        return cls(ckpt, server_url=server_url, transport=transport)

    # ----------------------------------------------------------------- transport

    def _call_remote(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send *payload* via the configured transport and return the JSON response."""
        t = self._transport
        if callable(t):
            return t(payload)
        if t is not None and hasattr(t, "post"):
            resp = t.post(self.server_url or "", json=payload)
            data = resp.json() if hasattr(resp, "json") else resp
            return data
        # Fall back to the default requests-backed HTTP path.
        if self.server_url is None:
            raise RuntimeError(
                "NLAExplainer has no transport configured. This should be unreachable "
                "given __init__'s validation."
            )
        return _default_http_transport(self.server_url)(payload)

    # ------------------------------------------------------------------- public

    def explain(self, activation_vector: Sequence[float] | np.ndarray) -> NLAExplanation:
        arr = self._coerce(activation_vector)
        payload = {
            "activation_vector": arr.tolist(),
            "nla_id": self.checkpoint.nla_id,
            "layer_idx": self.layer_idx,
        }
        resp = self._call_remote(payload)
        return self._parse_response(resp)

    def explain_batch(
        self, activation_vectors: Sequence[Sequence[float] | np.ndarray]
    ) -> List[NLAExplanation]:
        if len(activation_vectors) == 0:
            return []
        batch = np.stack([self._coerce(v) for v in activation_vectors], axis=0)
        payload = {
            "activation_vectors": batch.tolist(),
            "nla_id": self.checkpoint.nla_id,
            "layer_idx": self.layer_idx,
        }
        resp = self._call_remote(payload)
        results = resp.get("results")
        if results is None:
            raise ValueError(
                "Batch response from NLA server missing 'results' key. "
                f"Got keys: {sorted(resp.keys())!r}"
            )
        return [self._parse_response(r) for r in results]

    # ----------------------------------------------------------------- internals

    def _coerce(self, activation_vector: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(activation_vector, dtype=np.float32).ravel()
        if arr.shape[0] != self.d_model:
            raise ValueError(
                f"NLAExplainer for {self.checkpoint.nla_id!r} expects "
                f"d_model={self.d_model}, but received vector of length {arr.shape[0]}."
            )
        return arr

    def _parse_response(self, resp: Dict[str, Any]) -> NLAExplanation:
        if "text" not in resp:
            raise ValueError(
                f"NLA server response missing required 'text' field. Got: "
                f"{sorted(resp.keys())!r}"
            )
        text = str(resp["text"])
        fve = float(resp.get("reconstruction_fve", resp.get("fve", 0.0)))
        rec = resp.get("reconstructed_vector")
        rec_arr: Optional[np.ndarray] = None
        if rec is not None:
            rec_arr = np.asarray(rec, dtype=np.float32).ravel()
        metadata = {
            "model_id": self.model_id,
            "layer_idx": self.layer_idx,
            "d_model": self.d_model,
            "backend": "http",
            "nla_id": self.checkpoint.nla_id,
        }
        extra = resp.get("metadata")
        if isinstance(extra, dict):
            metadata.update(extra)
        return NLAExplanation(
            text=text,
            reconstruction_fve=fve,
            reconstructed_vector=rec_arr,
            metadata=metadata,
        )
