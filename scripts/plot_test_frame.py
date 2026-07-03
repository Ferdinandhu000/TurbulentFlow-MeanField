import os
import sys
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cfd.embedding import Voronoi
from common.functional import compute_velocity_field


def reduce_to_single_channel(frame: torch.Tensor) -> torch.Tensor:
    if frame.ndim != 3:
        raise ValueError(f"Expected frame with shape (C, H, W), got {tuple(frame.shape)}")
    return compute_velocity_field(frame, dim=0)


def ensure_resolution(frame: torch.Tensor, resolution: Tuple[int, int]) -> torch.Tensor:
    h, w = frame.shape[-2:]
    if (h, w) == resolution:
        return frame
    return F.interpolate(frame.unsqueeze(0), size=resolution, mode="bicubic", align_corners=False).squeeze(0)


def scale_sensor_positions(
    sensor_positions: torch.Tensor,
    source_resolution: Tuple[int, int],
    target_resolution: Tuple[int, int],
) -> torch.Tensor:
    if source_resolution == target_resolution:
        return sensor_positions
    src_h, src_w = source_resolution
    tgt_h, tgt_w = target_resolution
    scale_h = tgt_h / src_h
    scale_w = tgt_w / src_w
    scaled = sensor_positions.float().clone()
    scaled[:, 0] = scaled[:, 0] * scale_h
    scaled[:, 1] = scaled[:, 1] * scale_w
    scaled = scaled.long()
    scaled[:, 0].clamp_(min=0, max=tgt_h - 1)
    scaled[:, 1].clamp_(min=0, max=tgt_w - 1)
    return scaled


def plot_full_frame(
    frame: torch.Tensor,
    title: str,
    out_path: str,
    vmin: float,
    vmax: float,
    dpi: int,
    sensor_positions: torch.Tensor | None = None,
) -> None:
    h, w = frame.shape[-2:]
    figwidth = 18 / 25.4
    aspect_ratio = h / w

    fig, ax = plt.subplots(1, 1, figsize=(figwidth, figwidth * aspect_ratio))
    ax.imshow(
        frame.squeeze(0).cpu().numpy(),
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="bicubic",
        aspect="auto",
    )
    if sensor_positions is not None:
        x = sensor_positions[:, 1].cpu().numpy()
        y = sensor_positions[:, 0].cpu().numpy()
        ax.scatter(
            x,
            y,
            c="white",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            s=0.5,
            marker="o",
            edgecolors="black",
            linewidths=0.1,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def plot_sensor_points(
    sensor_positions: torch.Tensor,
    sensor_values: torch.Tensor,
    resolution: Tuple[int, int],
    title: str,
    out_path: str,
    vmin: float,
    vmax: float,
    dpi: int,
) -> None:
    h, w = resolution
    figwidth = 18 / 25.4
    aspect_ratio = h / w
    fig, ax = plt.subplots(1, 1, figsize=(figwidth, figwidth * aspect_ratio))

    x = sensor_positions[:, 1].cpu().numpy()
    y = sensor_positions[:, 0].cpu().numpy()
    colors = sensor_values.cpu().numpy()

    ax.scatter(
        x,
        y,
        c=colors,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        s=0.5,
        marker="o",
        edgecolors="black",
        linewidths=0.1,
    )

    ax.set_xlim(0, w - 1)
    ax.set_ylim(0, h - 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, transparent=False, facecolor="white", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    resolution = (72, 192)
    dpi = 900

    root = ROOT_DIR
    tensors_dir = os.path.join(root, "tensors", "test")
    fullstate_path = os.path.join(tensors_dir, "fullstate_values", "fv_test_0_000210.pt")
    sensor_pos_path = os.path.join(root, "sensor_position_pt", "pos_seed1.pt")
    out_dir = os.path.join(root, "plots_paper")

    os.makedirs(out_dir, exist_ok=True)

    fullstate = torch.load(fullstate_path, map_location="cpu", weights_only=True).float()
    fullstate = fullstate[0]
    source_resolution = (fullstate.shape[-2], fullstate.shape[-1])
    fullstate = ensure_resolution(fullstate, resolution)
    fullstate = reduce_to_single_channel(fullstate)

    sensor_positions = torch.load(sensor_pos_path, map_location="cpu", weights_only=True).int()
    sensor_positions = scale_sensor_positions(sensor_positions, source_resolution, resolution)

    h_indices = sensor_positions[:, 0]
    w_indices = sensor_positions[:, 1]
    sensor_values = fullstate[0, h_indices, w_indices]

    vmin = 0.0
    vmax = 5.0

    voronoi = Voronoi(resolution=resolution, sensor_positions=sensor_positions)
    voronoi_frame = voronoi(fullstate.unsqueeze(0).unsqueeze(0), seed=1)[0, 0]
    voronoi_frame = reduce_to_single_channel(voronoi_frame)

    # plot_sensor_points(
    #     sensor_positions=sensor_positions,
    #     sensor_values=sensor_values,
    #     resolution=resolution,
    #     title="Sensor Values (Seed=1)",
    #     out_path=os.path.join(out_dir, "s_test_200_sensor_points.png"),
    #     vmin=vmin,
    #     vmax=vmax,
    #     dpi=dpi,
    # )

    # plot_full_frame(
    #     frame=voronoi_frame,
    #     title="Voronoi Embedding (Seed=1)",
    #     out_path=os.path.join(out_dir, "s_test_200_voronoi.png"),
    #     vmin=vmin,
    #     vmax=vmax,
    #     dpi=dpi,
    #     sensor_positions=sensor_positions,
    # )

    plot_full_frame(
        frame=fullstate,
        title="Full State (First Frame)",
        out_path=os.path.join(out_dir, "small_test_210_fullstate.png"),
        vmin=vmin,
        vmax=vmax,
        dpi=dpi,
    )


if __name__ == "__main__":
    main()
