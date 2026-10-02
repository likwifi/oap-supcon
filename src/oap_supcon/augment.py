from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sequence_lengths(visibility: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """Real (unpadded) frame count per sequence.

    Sequences are zero-padded to the longest clip in the dataset, so every
    temporal operation must be scaled by the real length rather than by the
    padded buffer. Falling back to the padded length keeps the old behaviour for
    callers that genuinely have no padding (the smoke set, unit tests).
    """
    if lengths is None:
        return torch.full(
            (visibility.shape[0],), visibility.shape[1], dtype=torch.long, device=visibility.device
        )
    return lengths.to(device=visibility.device, dtype=torch.long).clamp(1, visibility.shape[1])


def frame_mask_from_lengths(visibility: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """[B,T] indicator of real (non-padding) frames."""
    real = sequence_lengths(visibility, lengths)
    positions = torch.arange(visibility.shape[1], device=visibility.device)
    return (positions[None, :] < real[:, None]).to(visibility.dtype)


def anatomical_parts(joints: int) -> list[list[int]]:
    """Return the paper's five-part partition: torso and four limbs."""
    if joints == 17:  # COCO-17
        return [[5, 6, 11, 12], [5, 7, 9], [6, 8, 10], [11, 13, 15], [12, 14, 16]]
    if joints == 18:  # OpenPose BODY-18
        return [[1, 2, 5, 8, 11], [5, 6, 7], [2, 3, 4], [11, 12, 13], [8, 9, 10]]
    if joints == 25:  # NTU/Kinect v2
        return [
            [0, 1, 2, 3, 20],
            [4, 5, 6, 7, 21, 22],
            [8, 9, 10, 11, 23, 24],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
        ]
    chunks = torch.tensor_split(torch.arange(joints), min(5, joints))
    return [chunk.tolist() for chunk in chunks if len(chunk)]


def _left_right_pairs(joints: int) -> list[tuple[int, int]]:
    if joints == 17:
        return [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    if joints == 18:
        return [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (14, 15), (16, 17)]
    if joints == 25:
        return (
            [(4 + offset, 8 + offset) for offset in range(4)]
            + [(12 + offset, 16 + offset) for offset in range(4)]
            + [(21, 23), (22, 24)]
        )
    return []


def _swap_left_right(x: torch.Tensor, visibility: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out, mask = x.clone(), visibility.clone()
    for left, right in _left_right_pairs(x.shape[1]):
        out[:, [left, right]] = out[:, [right, left]]
        mask[:, [left, right]] = mask[:, [right, left]]
    return out, mask


def generic_view(
    x: torch.Tensor,
    visibility: torch.Tensor,
    generator: torch.Generator,
    noise: float = 0.02,
    scale_range: tuple[float, float] = (0.95, 1.05),
):
    """Apply geometry appropriate to 2D pose or 3D skeleton coordinates."""
    out, mask = x.clone(), visibility.clone()
    batch, _, _, dimensions = out.shape
    scales = torch.empty((batch, 1, 1, 1), device=out.device, dtype=out.dtype).uniform_(
        scale_range[0], scale_range[1], generator=generator
    )
    out = out * scales
    if dimensions == 2:
        flips = torch.rand(batch, generator=generator, device=out.device) < 0.5
        for b in torch.nonzero(flips, as_tuple=False).flatten().tolist():
            flipped, flipped_mask = _swap_left_right(out[b], mask[b])
            flipped[..., 0] = -flipped[..., 0]
            out[b], mask[b] = flipped, flipped_mask
    elif dimensions == 3:
        # Y is vertical; yaw therefore rotates the X-Z plane.
        angles = (torch.rand(batch, generator=generator, device=out.device) - 0.5) * math.radians(20)
        for b, angle in enumerate(angles):
            c, s = torch.cos(angle), torch.sin(angle)
            xz = out[b, ..., [0, 2]].clone()
            out[b, ..., 0] = c * xz[..., 0] - s * xz[..., 1]
            out[b, ..., 2] = s * xz[..., 0] + c * xz[..., 1]
    eps = torch.randn(out.shape, generator=generator, device=out.device, dtype=out.dtype) * noise
    out = (out + eps) * (mask > 0).unsqueeze(-1)
    return out, mask


def _resample_crop(x: torch.Tensor, visibility: torch.Tensor, start: int, length: int, target: int):
    """Resample frames [start, start+length) onto `target` frames, re-padded to T."""
    crop_x = x[start : start + length]
    crop_v = visibility[start : start + length]
    flat = crop_x.permute(1, 2, 0).reshape(1, -1, length)
    sampled_x = F.interpolate(flat, size=target, mode="linear", align_corners=True)
    sampled_x = sampled_x.reshape(x.shape[1], x.shape[2], target).permute(2, 0, 1)
    sampled_v = F.interpolate(crop_v.T.unsqueeze(0), size=target, mode="nearest").squeeze(0).T
    out_x, out_v = torch.zeros_like(x), torch.zeros_like(visibility)
    out_x[:target] = sampled_x * (sampled_v > 0).unsqueeze(-1)
    out_v[:target] = sampled_v
    return out_x, out_v


def temporal_crop_view(
    x: torch.Tensor,
    visibility: torch.Tensor,
    generator: torch.Generator,
    min_ratio: float = 0.5,
    max_ratio: float = 0.8,
    lengths: torch.Tensor | None = None,
):
    """Sample one contiguous crop of the REAL frames and resize it back to that length.

    The crop is drawn from [0, L) and resampled onto L frames, so the view keeps the
    sequence's own padding structure instead of stretching a crop of mostly-padding
    across the whole buffer.
    """
    outputs, masks = [], []
    real = sequence_lengths(visibility, lengths)
    for b in range(x.shape[0]):
        span = int(real[b])
        if span < 2:  # nothing to crop from
            outputs.append(x[b])
            masks.append(visibility[b])
            continue
        ratio = float(torch.empty((), device=x.device).uniform_(min_ratio, max_ratio, generator=generator))
        length = min(span, max(2, round(ratio * span)))
        start = int(torch.randint(span - length + 1, (), generator=generator, device=x.device))
        cropped, cropped_mask = _resample_crop(x[b], visibility[b], start, length, span)
        outputs.append(cropped)
        masks.append(cropped_mask)
    return torch.stack(outputs), torch.stack(masks)


def speed_perturb(
    x: torch.Tensor,
    visibility: torch.Tensor,
    generator: torch.Generator,
    speed_range: tuple[float, float] = (0.8, 1.2),
    lengths: torch.Tensor | None = None,
):
    """Drop/duplicate frames through circular temporal resampling of the real frames."""
    outputs, masks = [], []
    real = sequence_lengths(visibility, lengths)
    for b in range(x.shape[0]):
        span = int(real[b])
        if span < 2:
            outputs.append(x[b])
            masks.append(visibility[b])
            continue
        speed = torch.empty((), device=x.device).uniform_(speed_range[0], speed_range[1], generator=generator)
        start = torch.rand((), generator=generator, device=x.device) * span
        positions = (start + torch.arange(span, device=x.device) * speed) % span
        lower = positions.floor().long()
        upper = (lower + 1) % span
        fraction = (positions - lower).view(span, 1, 1)
        sampled = x[b, lower] * (1 - fraction) + x[b, upper] * fraction
        nearest = positions.round().long() % span
        sampled_mask = visibility[b, nearest]
        out_x, out_v = torch.zeros_like(x[b]), torch.zeros_like(visibility[b])
        out_x[:span] = sampled * (sampled_mask > 0).unsqueeze(-1)
        out_v[:span] = sampled_mask
        outputs.append(out_x)
        masks.append(out_v)
    return torch.stack(outputs), torch.stack(masks)


def corrupt(
    x: torch.Tensor,
    visibility: torch.Tensor,
    family: str,
    severity: float,
    generator: torch.Generator,
    lengths: torch.Tensor | None = None,
):
    if severity <= 0:
        return x.clone(), visibility.clone()
    out, mask = x.clone(), visibility.clone()
    batch, frames, joints = mask.shape
    if family == "random_joint":
        # Independent joint-frame masking is the protocol defined in the paper.
        # Padding already has visibility 0, so the effective severity on the real
        # frames is exactly `severity`.
        keep = torch.rand(mask.shape, generator=generator, device=x.device) > severity
        mask = mask * keep
    elif family == "dynamic_joint":
        raw = torch.rand(mask.shape, generator=generator, device=x.device) > severity
        smooth = F.avg_pool1d(raw.float().transpose(1, 2), kernel_size=3, stride=1, padding=1)
        mask = mask * (smooth.transpose(1, 2) >= 0.5)
    elif family == "body_part":
        parts = anatomical_parts(joints)
        for b in range(batch):
            order = torch.randperm(len(parts), generator=generator, device=x.device).tolist()
            target = max(1, round(severity * joints))
            selected: list[int] = []
            for p in order:
                selected.extend(parts[p])
                if len(set(selected)) >= target:
                    break
            mask[b, :, sorted(set(selected))] = 0
    elif family == "temporal":
        # The interruption must span `severity` of the REAL clip, not of the
        # padded buffer, or the requested severity is not what is applied.
        real = sequence_lengths(visibility, lengths)
        for b in range(batch):
            span = int(real[b])
            length = min(span, max(1, round(severity * span)))
            start = int(torch.randint(span - length + 1, (), generator=generator, device=x.device))
            mask[b, start : start + length, :] = 0
    else:
        raise ValueError(f"unknown corruption family: {family}")
    return out * (mask > 0).unsqueeze(-1), mask.float()


def occlusion_view(
    x,
    visibility,
    severity: float,
    generator: torch.Generator,
    lengths: torch.Tensor | None = None,
    families: tuple[str, ...] = ("random_joint", "body_part", "temporal"),
):
    """Sample one corruption family per sequence (the anatomical occlusion sampler)."""
    outputs, masks = [], []
    choices = torch.randint(len(families), (x.shape[0],), generator=generator, device=x.device)
    for b, choice in enumerate(choices.tolist()):
        corrupted, corrupted_mask = corrupt(
            x[b : b + 1],
            visibility[b : b + 1],
            families[choice],
            severity,
            generator,
            None if lengths is None else lengths[b : b + 1],
        )
        outputs.append(corrupted[0])
        masks.append(corrupted_mask[0])
    return torch.stack(outputs), torch.stack(masks)


_per_sample_occlusion = occlusion_view  # backwards-compatible alias


def corruption_families(mode: str) -> tuple[str, ...]:
    """One policy used by BOTH the main views and temporal consistency branch."""
    if mode == "complete_to_partial_no_part_mask":
        return ("random_joint", "temporal")
    return ("random_joint", "body_part", "temporal")


def temporal_training_views(x, visibility, mode, severity, generator, lengths=None,
                            min_ratio=0.5, max_ratio=0.8):
    first = temporal_crop_view(x, visibility, generator, min_ratio, max_ratio, lengths)
    second = temporal_crop_view(x, visibility, generator, min_ratio, max_ratio, lengths)
    families = corruption_families(mode)
    second = occlusion_view(*second, severity, generator, lengths, families=families)
    if mode in {"two_partial", "dropout_to_partial", "mixed_partial"}:
        first = occlusion_view(*first, severity, generator, lengths, families=families)
    return first, second


def training_views(x, visibility, mode: str, severity: float, generator: torch.Generator, lengths=None):
    if mode == "none":
        return (x, visibility), (x, visibility)
    if mode == "joint_dropout":
        masked = corrupt(x, visibility, "random_joint", severity, generator, lengths)
        return masked, masked
    if mode == "generic":
        first = generic_view(*temporal_crop_view(x, visibility, generator, lengths=lengths), generator)
        second = generic_view(*temporal_crop_view(x, visibility, generator, lengths=lengths), generator)
        return first, second
    if mode == "mixed_partial":
        # Two independently occluded views, with an explicit clean-view mixture.
        # All branches use the same geometry; confidence never rescales it.
        views = []
        for _ in range(2):
            base, confidence = generic_view(x, visibility, generator, noise=0.01)
            partial, partial_confidence = occlusion_view(
                base, confidence, severity, generator, lengths,
                families=corruption_families(mode),
            )
            keep_clean = torch.rand((len(x), 1, 1), generator=generator, device=x.device) < 0.3
            views.append((torch.where(keep_clean.unsqueeze(-1), base, partial),
                          torch.where(keep_clean, confidence, partial_confidence)))
        return tuple(views)
    if mode == "dropout_to_partial":
        # masked_ce is the strongest robustness baseline and is orthogonal to the
        # contrastive objective: give view A random joint dropout so the CE and
        # global terms also train on partial evidence, and keep view B on the
        # full anatomical occlusion sampler.
        dropped = corrupt(x, visibility, "random_joint", severity, generator, lengths)
        strong_pair = occlusion_view(
            *speed_perturb(*generic_view(x, visibility, generator, noise=0.02), generator, lengths=lengths),
            severity, generator, lengths,
        )
        return generic_view(*dropped, generator, noise=0.01), strong_pair
    weak = generic_view(x, visibility, generator, noise=0.01)
    strong_base = speed_perturb(
        *generic_view(x, visibility, generator, noise=0.02), generator, lengths=lengths
    )
    if mode == "complete_to_partial_no_part_mask":
        strong_without_parts = occlusion_view(
            *strong_base, severity, generator, lengths, families=corruption_families(mode)
        )
        return weak, strong_without_parts
    strong = occlusion_view(*strong_base, severity, generator, lengths)
    if mode == "occlusion":
        return strong, strong
    if mode == "two_partial":
        return strong, occlusion_view(
            *generic_view(x, visibility, generator), severity, generator, lengths
        )
    if mode == "complete_to_partial":
        return weak, strong
    raise ValueError(f"unknown augmentation mode: {mode}")
