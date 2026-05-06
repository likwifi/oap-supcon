from __future__ import annotations

import math

import torch
import torch.nn.functional as F


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
    out = (out + eps) * mask.unsqueeze(-1)
    return out, mask


def _resample_crop(x: torch.Tensor, visibility: torch.Tensor, start: int, length: int):
    crop_x = x[start : start + length]
    crop_v = visibility[start : start + length]
    target_frames = x.shape[0]
    flat = crop_x.permute(1, 2, 0).reshape(1, -1, length)
    sampled_x = F.interpolate(flat, size=target_frames, mode="linear", align_corners=True)
    sampled_x = sampled_x.reshape(x.shape[1], x.shape[2], target_frames).permute(2, 0, 1)
    sampled_v = F.interpolate(crop_v.T.unsqueeze(0), size=target_frames, mode="nearest").squeeze(0).T
    return sampled_x * sampled_v.unsqueeze(-1), sampled_v


def temporal_crop_view(
    x: torch.Tensor,
    visibility: torch.Tensor,
    generator: torch.Generator,
    min_ratio: float = 0.5,
    max_ratio: float = 0.8,
):
    """Sample one independent contiguous crop per sequence and resize it to T."""
    outputs, masks = [], []
    frames = x.shape[1]
    for b in range(x.shape[0]):
        ratio = float(torch.empty((), device=x.device).uniform_(min_ratio, max_ratio, generator=generator))
        length = min(frames, max(2, round(ratio * frames)))
        start = int(torch.randint(frames - length + 1, (), generator=generator, device=x.device))
        cropped, cropped_mask = _resample_crop(x[b], visibility[b], start, length)
        outputs.append(cropped)
        masks.append(cropped_mask)
    return torch.stack(outputs), torch.stack(masks)


def speed_perturb(
    x: torch.Tensor,
    visibility: torch.Tensor,
    generator: torch.Generator,
    speed_range: tuple[float, float] = (0.8, 1.2),
):
    """Drop/duplicate frames through circular temporal resampling."""
    outputs, masks = [], []
    frames = x.shape[1]
    for b in range(x.shape[0]):
        speed = torch.empty((), device=x.device).uniform_(speed_range[0], speed_range[1], generator=generator)
        start = torch.rand((), generator=generator, device=x.device) * frames
        positions = (start + torch.arange(frames, device=x.device) * speed) % frames
        lower = positions.floor().long()
        upper = (lower + 1) % frames
        fraction = (positions - lower).view(frames, 1, 1)
        sampled = x[b, lower] * (1 - fraction) + x[b, upper] * fraction
        nearest = positions.round().long() % frames
        sampled_mask = visibility[b, nearest]
        outputs.append(sampled * sampled_mask.unsqueeze(-1))
        masks.append(sampled_mask)
    return torch.stack(outputs), torch.stack(masks)


def corrupt(
    x: torch.Tensor,
    visibility: torch.Tensor,
    family: str,
    severity: float,
    generator: torch.Generator,
):
    if severity <= 0:
        return x.clone(), visibility.clone()
    out, mask = x.clone(), visibility.clone()
    batch, frames, joints = mask.shape
    if family == "random_joint":
        # Independent joint-frame masking is the protocol defined in the paper.
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
        length = max(1, round(severity * frames))
        for b in range(batch):
            start = int(torch.randint(frames - length + 1, (), generator=generator, device=x.device))
            mask[b, start : start + length, :] = 0
    else:
        raise ValueError(f"unknown corruption family: {family}")
    return out * mask.unsqueeze(-1), mask.float()


def _per_sample_occlusion(
    x,
    visibility,
    severity: float,
    generator: torch.Generator,
    families: tuple[str, ...] = ("random_joint", "body_part", "temporal"),
):
    outputs, masks = [], []
    choices = torch.randint(len(families), (x.shape[0],), generator=generator, device=x.device)
    for b, choice in enumerate(choices.tolist()):
        corrupted, corrupted_mask = corrupt(
            x[b : b + 1], visibility[b : b + 1], families[choice], severity, generator
        )
        outputs.append(corrupted[0])
        masks.append(corrupted_mask[0])
    return torch.stack(outputs), torch.stack(masks)


def training_views(x, visibility, mode: str, severity: float, generator: torch.Generator):
    if mode == "none":
        return (x, visibility), (x, visibility)
    if mode == "joint_dropout":
        masked = corrupt(x, visibility, "random_joint", severity, generator)
        return masked, masked
    if mode == "generic":
        first = generic_view(*temporal_crop_view(x, visibility, generator), generator)
        second = generic_view(*temporal_crop_view(x, visibility, generator), generator)
        return first, second
    weak = generic_view(x, visibility, generator, noise=0.01)
    strong_base = speed_perturb(*generic_view(x, visibility, generator, noise=0.02), generator)
    if mode == "complete_to_partial_no_part_mask":
        strong_without_parts = _per_sample_occlusion(
            *strong_base, severity, generator, families=("random_joint", "temporal")
        )
        return weak, strong_without_parts
    strong = _per_sample_occlusion(*strong_base, severity, generator)
    if mode == "occlusion":
        return strong, strong
    if mode == "two_partial":
        return strong, _per_sample_occlusion(*generic_view(x, visibility, generator), severity, generator)
    if mode == "complete_to_partial":
        return weak, strong
    raise ValueError(f"unknown augmentation mode: {mode}")
