"""
A minimum viable example of inference taking a test scene of WildRGB-D dataset.
Usage: uv run --extra example examples/wildrgbd/mve.py
"""

from pathlib import Path
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.structures import Pointclouds
from pytorch3d.utils import cameras_from_opencv_projection
from torchcodec.encoders import VideoEncoder
from vggt_omega.models import VGGTOmega
from vggt_omega.utils import load_fn, pose_enc
from torch import Tensor
import torch


def main():
    # Load the model
    model = VGGTOmega()
    model.load_state_dict(
        torch.load(
            "/home/3dv_25/shared/hdd_vg/projects/facebook/VGGT-Omega/vggt_omega_1b_512.pt"
        )
    )
    model = model.to("cuda")

    # Load the image
    # (B, S, 3, H, W) in [0, 1]
    image = (
        load_fn.load_and_preprocess_images(
            sorted(
                Path(
                    "/home/3dv_25/shared/hdd_vg/datasets/wildrgb_d/TV/scenes/scene_000/rgb"
                ).glob("*.png")
            )
        )[::32]
        .unsqueeze(0)
        .to("cuda")
    )
    print(f"image: {tuple(image.shape)}")

    # Switch to evaluation mode
    model = model.eval().requires_grad_(False)
    with torch.inference_mode():
        # Run model inference
        with torch.autocast("cuda", dtype=torch.bfloat16):
            token, index = model.aggregator(image)
            # [(B, S, H, W, 1) in exp, (B, S, H, W, 1) in exp+1]
            y: tuple[Tensor, Tensor] = model.dense_head(token, *image.shape[-2:], index)
            depth, depth_conf = y
            # (B, S, 9)
            pose: Tensor = model.camera_head(token, index)

        assert depth.dtype == torch.float32
        assert pose.dtype == torch.float32

        # Build point cloud from depth and camera pose
        # [(B, S, 3, 4) in W2C, (B, S, 3, 3)]
        pose_extr, pose_intr = pose_enc.encoding_to_camera(pose, image.shape[-2:])
        # [(B, N, 3), (B, N, 3)]
        pcd = Pointclouds(
            points=unproject(depth, pose_extr, pose_intr).flatten(-4, -2),
            features=image.movedim(-3, -1).contiguous().flatten(-4, -2),
        )

        # Interpolate camera poses
        # (B, S_seq, 9)
        pose_seq = interpolate_pose(pose, step=16)

        # Render point cloud
        # (B, S_seq, 3, H, W) in [0, 255]
        image_seq = (
            rasterize_pcd(
                pcd=pcd,
                pose=pose_seq,
                frame_size=image.shape[-2:],
            )
            .clamp(0.0, 1.0)
            .mul(255.0)
            .to(torch.uint8)
        )

        # Save the rendered frames as a video
        VideoEncoder(image_seq[0].cpu(), frame_rate=30.0).to_file(
            Path(__file__).with_suffix(".mp4"),
            codec="h264",
        )


def interpolate_pose(pose: Tensor, step: int) -> Tensor:
    """
    Interpolate camera pose encoding across keyframes into output frames.

    Shape:
        [(B, S_i, 3 + 4 + 2), S_o / (S_i - 1)] -> (B, S_o, 3 + 4 + 2)
    """

    eps = torch.finfo(pose.dtype).eps
    S_i = pose.shape[-2]
    S_o = (S_i - 1) * step

    # Half-open positions
    # (S_o,) in [0, S_i - 1)
    pos = pose.new_tensor(range(S_o)).div(step)
    i0 = pos.long()
    i1 = i0 + 1
    w = pos.sub(i0)[..., None]

    # Keyframe pairs
    # [[(B, S_o, 3), (B, S_o, 4) in [x, y, z, w], (B, S_o, 2) in [h, w]]; 2]
    t0, q0, f0 = pose[:, i0].split([3, 4, 2], dim=-1)
    t1, q1, f1 = pose[:, i1].split([3, 4, 2], dim=-1)

    # Lerp for translation and FOV
    t_seq = t0.lerp(t1, w)
    f_seq = f0.lerp(f1, w)

    # slerp quaternion (shortest arc + small-angle lerp fallback)
    dot = q0.mul(q1).sum(dim=-1, keepdim=True)
    q1 = q1.where(dot.ge(0.0), -q1)
    dot = dot.abs().clamp(max=1.0)
    theta = dot.acos()
    sin_theta = theta.sin()
    small = sin_theta < eps
    sin_theta = sin_theta.clamp(min=eps)
    a = torch.where(small, 1.0 - w, (1.0 - w).mul(theta).sin() / sin_theta)
    b = torch.where(small, w, w.mul(theta).sin() / sin_theta)
    q_seq = a * q0 + b * q1

    # (B, S_o, 3 + 4 + 2)
    return torch.cat([t_seq, q_seq, f_seq], dim=-1)


def rasterize_pcd(
    pcd: Pointclouds,
    pose: Tensor,
    frame_size: tuple[int, int],
) -> Tensor:
    """
    Rasterize point cloud from interpolated camera poses into RGB frames.

    Shape:
        [[(B, N, 3), (B, N, 3)], (B, S_o, 9), [H, W]] -> (B, S_o, 3, H, W)
    """

    chunk = 16
    B, S_o = pose.shape[:2]
    N = B * S_o

    # Decode pose encoding to camera matrices in OpenCV convention
    # [(B, S_o, 3, 4) in W2C, (B, S_o, 3, 3)]
    extr, intr = pose_enc.encoding_to_camera(pose, frame_size)

    # Flatten batch and split camera pose
    # [(N, 3, 3), (N, 3), (N, 3, 3)]
    R, t = extr.flatten(0, 1).split([3, 1], dim=-1)
    t = t.squeeze(-1)
    K = intr.flatten(0, 1)
    image_size = pose.new_tensor([frame_size] * N)

    # Render in chunks of camera batch (pcd.extend duplicates the cloud)
    rgb_chunks = []
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        cameras = cameras_from_opencv_projection(
            R=R[i:j],
            tvec=t[i:j],
            camera_matrix=K[i:j],
            image_size=image_size[i:j],
        )
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=cameras,
                raster_settings=PointsRasterizationSettings(
                    image_size=frame_size,
                    points_per_pixel=8,
                    radius=4.0 / min(frame_size),
                    max_points_per_bin=524288,
                ),
            ),
            compositor=AlphaCompositor(),
        )
        # (j - i, H, W, 3) -> (j - i, 3, H, W)
        rgb: Tensor = renderer(pcd.extend(j - i))
        rgb_chunks.append(rgb.movedim(-1, -3).contiguous())

    # (N, 3, H, W) -> (B, S_o, 3, H, W)
    return torch.cat(rgb_chunks, dim=0).unflatten(0, (B, S_o))


def unproject(depth: Tensor, extr: Tensor, intr: Tensor) -> Tensor:
    """
    Unproject camera pose and depth to point in world coordinate.

    Formula:
        Y = R^t @ K^-1 @ [u, v, 1]^t
        C = -R^t @ T
        P = D * Y + C

    Shape:
        [(..., H, W, 1), (..., 3, 4) in W2C, (..., 3, 3)] -> (..., H, W, 3) in world
    """

    H, W = depth.shape[-3:-1]
    eps = torch.finfo(intr.dtype).eps

    # Create pixel grid
    # [(H, W); 2]
    u, v = torch.meshgrid(
        intr.new_tensor(range(W)),
        intr.new_tensor(range(H)),
        indexing="xy",
    )

    # (..., 3, 3) -> [(..., 1, 1); 4]
    fx = intr[..., 0:1, 0:1]
    fy = intr[..., 1:2, 1:2]
    cx = intr[..., 0:1, 2:3]
    cy = intr[..., 1:2, 2:3]

    # Ray direction in camera coordinate
    # [(..., H, W); 3] -> (..., H, W, 3) -> (..., H, W, 3, 1)
    ray_x = u.sub(cx).div(fx.clamp(min=eps))
    ray_y = v.sub(cy).div(fy.clamp(min=eps))
    ray_z = torch.ones_like(ray_x)
    y_cam = torch.stack([ray_x, ray_y, ray_z], dim=-1)[..., None]

    # C2W transformation
    # (..., 1, 1, 3, 4) -> [(..., 1, 1, 3, 3), (..., 1, 1, 3, 1)]
    r, t = extr[..., None, None, :, :].split([3, 1], dim=-1)
    r_inv = r.mT.contiguous()
    t_inv = r_inv.matmul(t).neg()

    # (..., 1, 1, 3, 3) @ (..., H, W, 3, 1) -> (..., H, W, 3, 1)
    # [(..., H, W, 3, 1), (..., 1, 1, 3, 1)] -> [(..., H, W, 3), (..., 1, 1, 3)]
    y_wld, c_wld = tuple(x.squeeze(-1) for x in (r_inv.matmul(y_cam), t_inv))

    # (..., H, W, 1) * (..., H, W, 3) + (..., 1, 1, 3) -> (..., H, W, 3)
    p_wld = depth.mul(y_wld).add(c_wld)
    return p_wld


if __name__ == "__main__":
    main()
