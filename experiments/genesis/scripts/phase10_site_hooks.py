import torch

from scripts.run_phase9_token_position_steering import resolve_position_index


SITE_SPECS = {
    "block_input": {"kind": "pre", "path": ()},
    "attn_norm": {"kind": "forward", "path": ("attn_norm",)},
    "q_proj": {"kind": "forward", "path": ("attn", "q_proj")},
    "v_proj": {"kind": "forward", "path": ("attn", "v_proj")},
    "o_proj_input": {"kind": "pre", "path": ("attn", "o_proj")},
    "fgate_proj_input": {"kind": "pre", "path": ("attn", "fgate_proj")},
    "fgate_proj": {"kind": "forward", "path": ("attn", "fgate_proj")},
    "ogate_proj_input": {"kind": "pre", "path": ("attn", "ogate_proj")},
    "ogate_proj": {"kind": "forward", "path": ("attn", "ogate_proj")},
    "o_proj": {"kind": "forward", "path": ("attn", "o_proj")},
    "attn_output": {"kind": "forward", "path": ("attn",)},
    "ffn_norm": {"kind": "forward", "path": ("ffn_norm",)},
    "ffn_output": {"kind": "forward", "path": ("ffn",)},
    "block_output": {"kind": "forward", "path": ()},
}

DEFAULT_L11_SITES = [
    "block_input",
    "attn_norm",
    "q_proj",
    "v_proj",
    "o_proj",
    "attn_output",
    "ffn_norm",
    "ffn_output",
    "block_output",
]


def parse_site_list(raw_sites):
    sites = [site.strip() for site in raw_sites.split(",") if site.strip()]
    unknown = [site for site in sites if site not in SITE_SPECS]
    if unknown:
        raise ValueError(f"Unknown steering sites: {unknown}. Valid sites: {sorted(SITE_SPECS)}")
    return sites


def extract_output_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def replace_output_tensor(output, new_tensor):
    if isinstance(output, tuple):
        return (new_tensor, *output[1:])
    return new_tensor


def match_norm_to_reference(source, reference):
    source_norm = torch.norm(source, dim=-1, keepdim=True)
    reference_norm = torch.norm(reference, dim=-1, keepdim=True)
    scale = reference_norm / (source_norm + 1e-10)
    return source * scale


def resolve_site_module(model, layer, site):
    spec = SITE_SPECS[site]
    module = model.blocks[int(layer)]
    for attr in spec["path"]:
        module = getattr(module, attr)
    return module, spec["kind"]


def apply_residual_style_intervention(x, vector, alpha, mode, position_fraction):
    pos = resolve_position_index(x.shape[1], position_fraction)
    x_mod = x.clone()
    vec = vector.unsqueeze(0).to(device=x.device, dtype=x.dtype)
    if mode == "add":
        x_mod[:, pos, :] = x_mod[:, pos, :] + (float(alpha) * vec)
        return x_mod
    if mode == "ablate":
        token_slice = x_mod[:, pos, :]
        coeff = torch.sum(token_slice * vec, dim=-1, keepdim=True)
        proj = coeff * vec
        x_mod[:, pos, :] = token_slice - (float(alpha) * proj)
        return x_mod
    raise ValueError(f"Unsupported intervention mode: {mode}")


def resolve_answer_window_positions(seq_len, answer_offset=1, window_size=1):
    seq_len = int(seq_len)
    if seq_len <= 1:
        return [0]
    offset = max(1, int(answer_offset))
    size = max(1, int(window_size))
    end_idx = max(0, seq_len - offset)
    start_idx = max(0, end_idx - size + 1)
    return list(range(start_idx, end_idx + 1))


def apply_residual_style_intervention_to_positions(x, vector, alpha, mode, positions):
    x_mod = x.clone()
    if not positions:
        return x_mod
    vec = vector.unsqueeze(0).to(device=x.device, dtype=x.dtype)
    pos = [int(idx) for idx in positions]
    token_slice = x_mod[:, pos, :]
    if mode == "add":
        x_mod[:, pos, :] = token_slice + (float(alpha) * vec.unsqueeze(1))
        return x_mod
    if mode == "ablate":
        coeff = torch.sum(token_slice * vec.unsqueeze(1), dim=-1, keepdim=True)
        proj = coeff * vec.unsqueeze(1)
        x_mod[:, pos, :] = token_slice - (float(alpha) * proj)
        return x_mod
    raise ValueError(f"Unsupported intervention mode: {mode}")


class TensorSiteInterventionHook:
    def __init__(self, vector, alpha=0.0, mode="add", position_fraction=1.0):
        self.vector = vector
        self.alpha = float(alpha)
        self.mode = mode
        self.position_fraction = float(position_fraction)
        self.enabled = True
        self.handle = None

    def _pre_hook(self):
        def hook_fn(module, args):
            if not self.enabled or self.vector is None or self.alpha == 0.0:
                return None
            x = args[0]
            x_mod = apply_residual_style_intervention(x, self.vector, self.alpha, self.mode, self.position_fraction)
            return (x_mod, *args[1:])

        return hook_fn

    def _forward_hook(self):
        def hook_fn(module, args, output):
            if not self.enabled or self.vector is None or self.alpha == 0.0:
                return output
            x = extract_output_tensor(output)
            x_mod = apply_residual_style_intervention(x, self.vector, self.alpha, self.mode, self.position_fraction)
            return replace_output_tensor(output, x_mod)

        return hook_fn

    def attach(self, model, layer, site):
        module, kind = resolve_site_module(model, layer, site)
        if kind == "pre":
            self.handle = module.register_forward_pre_hook(self._pre_hook())
            return
        self.handle = module.register_forward_hook(self._forward_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class TensorSiteAnswerWindowInterventionHook:
    def __init__(self, vector, alpha=0.0, mode="add", answer_offset=1, window_size=1):
        self.vector = vector
        self.alpha = float(alpha)
        self.mode = mode
        self.answer_offset = int(answer_offset)
        self.window_size = int(window_size)
        self.enabled = True
        self.handle = None

    def _apply(self, x):
        positions = resolve_answer_window_positions(
            x.shape[1],
            answer_offset=self.answer_offset,
            window_size=self.window_size,
        )
        return apply_residual_style_intervention_to_positions(
            x,
            self.vector,
            self.alpha,
            self.mode,
            positions,
        )

    def _pre_hook(self):
        def hook_fn(module, args):
            if not self.enabled or self.vector is None or self.alpha == 0.0:
                return None
            x_mod = self._apply(args[0])
            return (x_mod, *args[1:])

        return hook_fn

    def _forward_hook(self):
        def hook_fn(module, args, output):
            if not self.enabled or self.vector is None or self.alpha == 0.0:
                return output
            x_mod = self._apply(extract_output_tensor(output))
            return replace_output_tensor(output, x_mod)

        return hook_fn

    def attach(self, model, layer, site):
        module, kind = resolve_site_module(model, layer, site)
        if kind == "pre":
            self.handle = module.register_forward_pre_hook(self._pre_hook())
            return
        self.handle = module.register_forward_hook(self._forward_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class TensorSiteAnswerWindowCaptureHook:
    def __init__(self, answer_offset=1, window_size=1):
        self.answer_offset = int(answer_offset)
        self.window_size = int(window_size)
        self.handle = None
        self.captured = None
        self.positions = None

    def _capture_tensor(self, x):
        self.positions = resolve_answer_window_positions(
            x.shape[1],
            answer_offset=self.answer_offset,
            window_size=self.window_size,
        )
        self.captured = x[:, self.positions, :].detach().clone()

    def _pre_hook(self):
        def hook_fn(module, args):
            self._capture_tensor(args[0])
            return None

        return hook_fn

    def _forward_hook(self):
        def hook_fn(module, args, output):
            self._capture_tensor(extract_output_tensor(output))
            return output

        return hook_fn

    def attach(self, model, layer, site):
        module, kind = resolve_site_module(model, layer, site)
        if kind == "pre":
            self.handle = module.register_forward_pre_hook(self._pre_hook())
            return
        self.handle = module.register_forward_hook(self._forward_hook())

    def get_captured(self):
        return self.captured

    def clear(self):
        self.captured = None
        self.positions = None

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class TensorSiteCaptureHook:
    def __init__(self, position_fraction=1.0):
        self.position_fraction = float(position_fraction)
        self.handle = None
        self.captured = None

    def _capture_tensor(self, x):
        pos = resolve_position_index(x.shape[1], self.position_fraction)
        self.captured = x[:, pos, :].detach().clone()

    def _pre_hook(self):
        def hook_fn(module, args):
            self._capture_tensor(args[0])
            return None

        return hook_fn

    def _forward_hook(self):
        def hook_fn(module, args, output):
            self._capture_tensor(extract_output_tensor(output))
            return output

        return hook_fn

    def attach(self, model, layer, site):
        module, kind = resolve_site_module(model, layer, site)
        if kind == "pre":
            self.handle = module.register_forward_pre_hook(self._pre_hook())
            return
        self.handle = module.register_forward_hook(self._forward_hook())

    def clear(self):
        self.captured = None

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class TensorSiteSwapHook:
    def __init__(self, source_tensor, position_fraction=1.0, norm_match=True):
        self.source_tensor = source_tensor
        self.position_fraction = float(position_fraction)
        self.norm_match = bool(norm_match)
        self.enabled = True
        self.handle = None

    def _apply_swap(self, x):
        pos = resolve_position_index(x.shape[1], self.position_fraction)
        current = x[:, pos, :]
        source = self.source_tensor.to(device=x.device, dtype=x.dtype)
        if source.dim() == 1:
            source = source.unsqueeze(0).expand_as(current)
        elif source.shape[0] == 1 and current.shape[0] != 1:
            source = source.expand_as(current)
        if self.norm_match:
            source = match_norm_to_reference(source, current)
        x_mod = x.clone()
        x_mod[:, pos, :] = source
        return x_mod

    def _pre_hook(self):
        def hook_fn(module, args):
            if not self.enabled or self.source_tensor is None:
                return None
            return (self._apply_swap(args[0]), *args[1:])

        return hook_fn

    def _forward_hook(self):
        def hook_fn(module, args, output):
            if not self.enabled or self.source_tensor is None:
                return output
            x_mod = self._apply_swap(extract_output_tensor(output))
            return replace_output_tensor(output, x_mod)

        return hook_fn

    def attach(self, model, layer, site):
        module, kind = resolve_site_module(model, layer, site)
        if kind == "pre":
            self.handle = module.register_forward_pre_hook(self._pre_hook())
            return
        self.handle = module.register_forward_hook(self._forward_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class TensorSiteDirectionalCoeffInterchangeHook:
    def __init__(self, direction, donor_coeff, alpha=1.0, position_fraction=1.0):
        self.direction = direction
        self.donor_coeff = donor_coeff
        self.alpha = float(alpha)
        self.position_fraction = float(position_fraction)
        self.enabled = True
        self.handle = None

    def _expand_coeff(self, coeff, reference):
        if not torch.is_tensor(coeff):
            coeff = torch.tensor(coeff, device=reference.device, dtype=reference.dtype)
        coeff = coeff.to(device=reference.device, dtype=reference.dtype)
        if coeff.dim() == 0:
            coeff = coeff.view(1, 1).expand(reference.shape[0], 1)
        elif coeff.dim() == 1:
            if coeff.shape[0] == 1:
                coeff = coeff.view(1, 1).expand(reference.shape[0], 1)
            elif coeff.shape[0] == reference.shape[0]:
                coeff = coeff.view(reference.shape[0], 1)
            else:
                raise ValueError(f"Unexpected donor_coeff batch shape: {tuple(coeff.shape)}")
        elif coeff.dim() == 2 and coeff.shape == (reference.shape[0], 1):
            pass
        else:
            raise ValueError(f"Unexpected donor_coeff shape: {tuple(coeff.shape)}")
        return coeff

    def _apply_interchange(self, x):
        pos = resolve_position_index(x.shape[1], self.position_fraction)
        current = x[:, pos, :]
        direction = self.direction.to(device=x.device, dtype=x.dtype)
        if direction.dim() == 1:
            direction = direction.unsqueeze(0).expand_as(current)
        elif direction.shape[0] == 1 and current.shape[0] != 1:
            direction = direction.expand_as(current)
        direction = direction / torch.clamp(torch.norm(direction, dim=-1, keepdim=True), min=1e-10)
        donor_coeff = self._expand_coeff(self.donor_coeff, current)
        current_coeff = torch.sum(current * direction, dim=-1, keepdim=True)
        delta = (donor_coeff - current_coeff) * direction
        x_mod = x.clone()
        x_mod[:, pos, :] = current + (self.alpha * delta)
        return x_mod

    def _pre_hook(self):
        def hook_fn(module, args):
            if not self.enabled or self.direction is None or self.alpha == 0.0:
                return None
            return (self._apply_interchange(args[0]), *args[1:])

        return hook_fn

    def _forward_hook(self):
        def hook_fn(module, args, output):
            if not self.enabled or self.direction is None or self.alpha == 0.0:
                return output
            x_mod = self._apply_interchange(extract_output_tensor(output))
            return replace_output_tensor(output, x_mod)

        return hook_fn

    def attach(self, model, layer, site):
        module, kind = resolve_site_module(model, layer, site)
        if kind == "pre":
            self.handle = module.register_forward_pre_hook(self._pre_hook())
            return
        self.handle = module.register_forward_hook(self._forward_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class TensorSiteOrthogonalResidualInterchangeHook:
    def __init__(self, direction, donor_tensor, alpha=1.0, position_fraction=1.0, donor_norm_match=True):
        self.direction = direction
        self.donor_tensor = donor_tensor
        self.alpha = float(alpha)
        self.position_fraction = float(position_fraction)
        self.donor_norm_match = bool(donor_norm_match)
        self.enabled = True
        self.handle = None

    def _expand_tensor(self, tensor, reference):
        if not torch.is_tensor(tensor):
            tensor = torch.tensor(tensor, device=reference.device, dtype=reference.dtype)
        tensor = tensor.to(device=reference.device, dtype=reference.dtype)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0).expand_as(reference)
        elif tensor.dim() == 2 and tensor.shape[0] == 1 and reference.shape[0] != 1:
            tensor = tensor.expand_as(reference)
        elif tensor.shape != reference.shape:
            raise ValueError(f"Unexpected donor_tensor shape: {tuple(tensor.shape)} vs reference {tuple(reference.shape)}")
        return tensor

    def _expand_direction(self, direction, reference):
        direction = direction.to(device=reference.device, dtype=reference.dtype)
        if direction.dim() == 1:
            direction = direction.unsqueeze(0).expand_as(reference)
        elif direction.dim() == 2 and direction.shape[0] == 1 and reference.shape[0] != 1:
            direction = direction.expand_as(reference)
        elif direction.shape != reference.shape:
            raise ValueError(f"Unexpected direction shape: {tuple(direction.shape)} vs reference {tuple(reference.shape)}")
        return direction / torch.clamp(torch.norm(direction, dim=-1, keepdim=True), min=1e-10)

    def _apply_interchange(self, x):
        pos = resolve_position_index(x.shape[1], self.position_fraction)
        current = x[:, pos, :]
        donor = self._expand_tensor(self.donor_tensor, current)
        direction = self._expand_direction(self.direction, current)

        current_coeff = torch.sum(current * direction, dim=-1, keepdim=True)
        donor_coeff = torch.sum(donor * direction, dim=-1, keepdim=True)
        current_orth = current - (current_coeff * direction)
        donor_orth = donor - (donor_coeff * direction)
        if self.donor_norm_match:
            donor_orth = match_norm_to_reference(donor_orth, current_orth)
        updated = (current_coeff * direction) + current_orth + (self.alpha * (donor_orth - current_orth))
        x_mod = x.clone()
        x_mod[:, pos, :] = updated
        return x_mod

    def _pre_hook(self):
        def hook_fn(module, args):
            if not self.enabled or self.direction is None or self.donor_tensor is None or self.alpha == 0.0:
                return None
            return (self._apply_interchange(args[0]), *args[1:])

        return hook_fn

    def _forward_hook(self):
        def hook_fn(module, args, output):
            if not self.enabled or self.direction is None or self.donor_tensor is None or self.alpha == 0.0:
                return output
            x_mod = self._apply_interchange(extract_output_tensor(output))
            return replace_output_tensor(output, x_mod)

        return hook_fn

    def attach(self, model, layer, site):
        module, kind = resolve_site_module(model, layer, site)
        if kind == "pre":
            self.handle = module.register_forward_pre_hook(self._pre_hook())
            return
        self.handle = module.register_forward_hook(self._forward_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class TensorSiteSubspaceOverwriteHook:
    def __init__(self, site, mean, basis, donor_tensor, alpha=1.0, position_fraction=1.0, donor_norm_match=False):
        self.site = site
        self.mean = mean
        self.basis = basis
        self.donor_tensor = donor_tensor
        self.alpha = float(alpha)
        self.position_fraction = float(position_fraction)
        self.donor_norm_match = bool(donor_norm_match)
        self.enabled = True
        self.handle = None

    def _project(self, x, mean, basis):
        centered = x - mean.unsqueeze(0)
        coeff = centered @ basis
        return coeff @ basis.transpose(0, 1)

    def _apply_overwrite(self, x):
        pos = resolve_position_index(x.shape[1], self.position_fraction)
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        basis = self.basis.to(device=x.device, dtype=x.dtype)
        current = x[:, pos, :]
        donor = self.donor_tensor.to(device=x.device, dtype=x.dtype)
        if donor.dim() == 1:
            donor = donor.unsqueeze(0)
        elif donor.shape[0] == 1 and current.shape[0] != 1:
            donor = donor.expand_as(current)
        current_proj = self._project(current, mean, basis)
        donor_proj = self._project(donor, mean, basis)
        if self.donor_norm_match:
            donor_proj = match_norm_to_reference(donor_proj, current_proj)
        updated = current - (self.alpha * current_proj) + (self.alpha * donor_proj)
        x_mod = x.clone()
        x_mod[:, pos, :] = updated
        return x_mod

    def _pre_hook(self):
        def hook_fn(module, args):
            if not self.enabled or self.alpha == 0.0 or self.donor_tensor is None:
                return None
            return (self._apply_overwrite(args[0]), *args[1:])

        return hook_fn

    def _forward_hook(self):
        def hook_fn(module, args, output):
            if not self.enabled or self.alpha == 0.0 or self.donor_tensor is None:
                return output
            x_mod = self._apply_overwrite(extract_output_tensor(output))
            return replace_output_tensor(output, x_mod)

        return hook_fn

    def attach(self, model, layer):
        module, kind = resolve_site_module(model, layer, self.site)
        if kind == "pre":
            self.handle = module.register_forward_pre_hook(self._pre_hook())
            return
        self.handle = module.register_forward_hook(self._forward_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class BlockOutputCaptureHook:
    def __init__(self, layer):
        self.layer = int(layer)
        self.handle = None
        self.captured = None

    def _make_hook(self):
        def hook_fn(module, args, output):
            x = extract_output_tensor(output)
            self.captured = x[:, -1, :].detach().clone()
            return output

        return hook_fn

    def attach(self, model):
        self.handle = model.blocks[self.layer].register_forward_hook(self._make_hook())

    def clear(self):
        self.captured = None

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None