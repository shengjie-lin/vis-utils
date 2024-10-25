from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import torch
from imageio.v3 import imread
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial.transform import Rotation


def get_cur_timestamp():
    return datetime.now().strftime('%m-%d_%H-%M-%S')


def ch_cam_pose_spec(T, src, tgt, pose_type='c2w'):
    """ src/tgt spec:
            0: x->right, y->front, z->up
            1: x->right, y->down, z->front
            2: x->right, y->up, z->back
            3: x->front, y->left, z->up
        pose_type:
            'c2w': camera to world, where world can refer to any frame
            'w2c': world to camera """
    # x-to-0 transforms
    Ts = np.array((np.eye(4),
                   ((1, 0, 0, 0),
                    (0, 0, 1, 0),
                    (0, -1, 0, 0),
                    (0, 0, 0, 1)),
                   ((1, 0, 0, 0),
                    (0, 0, -1, 0),
                    (0, 1, 0, 0),
                    (0, 0, 0, 1)),
                   ((0, -1, 0, 0),
                    (1, 0, 0, 0),
                    (0, 0, 1, 0),
                    (0, 0, 0, 1))), dtype=T.dtype)
    s = T.shape[-2:]
    T = homogenize_transforms(T)
    return (T @ np.linalg.inv(Ts[src]) @ Ts[tgt] if pose_type == 'c2w' else np.linalg.inv(Ts[tgt]) @ Ts[src] @ T)[..., :s[0], :s[1]]


def gen_lookat_pose(c, t, u=None, pose_spec=2, pose_type='c2w'):
    """ generates a c2w pose
        c: camera center
        t: target to look at
        u: up vector
        pose_spec: cam frame spec
            0: x->right, y->front, z->up
            1: x->right, y->down, z->front
            2: x->right, y->up, z->back
            3: x->front, y->left, z->up
        pose_type: one of {'c2w', 'w2c'} """
    if u is None:
        u = np.array((0, 0, 1))
    y = t - c
    y = y / np.linalg.norm(y)
    x = np.cross(y, u)
    x = x / np.linalg.norm(x)
    z = np.cross(x, y)
    R = ch_cam_pose_spec(np.array((x, y, z)).T, 0, pose_spec)
    if pose_type == 'w2c':
        R = R.T
        c = -R @ c
    return np.concatenate((R, c[:, None]), axis=1, dtype=np.float32)


def gen_elliptical_poses(a, b, theta, h, azimuth_lo=None, azimuth_hi=None, target=None, up=None, n=10, pose_spec=2):
    """ generate n poses (c2w) distributed along an ellipse of (a, b, theta) at height h """
    if azimuth_lo is None:
        azimuth_lo = 0
    if azimuth_hi is None:
        azimuth_hi = np.pi * 2
    if target is None:
        target = np.zeros(3)
    poses = []
    for alpha in np.linspace(azimuth_lo, azimuth_hi, num=n, endpoint=(azimuth_hi - azimuth_lo) % (np.pi * 2)):
        x0 = a * np.cos(alpha)
        y0 = b * np.sin(alpha)
        x = x0 * np.cos(theta) - y0 * np.sin(theta)
        y = y0 * np.cos(theta) + x0 * np.sin(theta)
        z = h
        poses.append(gen_lookat_pose(np.array((x, y, z)), target, u=up, pose_spec=pose_spec))
    return poses


def gen_circular_poses(r, h, azimuth_lo=None, azimuth_hi=None, target=None, up=None, n=10, pose_spec=2):
    """ generate n poses (c2w) distributed along a circle of radius r at height h """
    return gen_elliptical_poses(r, r, 0, h, azimuth_lo=azimuth_lo, azimuth_hi=azimuth_hi, target=target, up=up, n=n, pose_spec=pose_spec)


def gen_spherical_poses(r, elevation_lo, elevation_hi, azimuth_lo=None, azimuth_hi=None, target=None, up=None, m=3, n=10, pose_spec=2):
    c2ws = []
    for g in np.linspace(elevation_lo, elevation_hi, num=m):
        c2ws.extend(gen_circular_poses(r * np.cos(g), r * np.sin(g), azimuth_lo=azimuth_lo, azimuth_hi=azimuth_hi, target=target, up=up, n=n, pose_spec=pose_spec))
    return c2ws


def gen_spherical_spiral_poses(r, elevation_lo, elevation_hi, start, end, target=None, up=None, n=10, pose_spec=2):
    assert n > 1
    if target is None:
        target = np.zeros(3)
    c2ws = []
    for i in range(n):
        elevation = elevation_lo + (elevation_hi - elevation_lo) * i / (n - 1)
        azimuth = (start + (end - start) * i / (n - 1)) * np.pi * 2
        c2ws.append(gen_lookat_pose(np.array((r * np.cos(elevation) * np.cos(azimuth), r * np.cos(elevation) * np.sin(azimuth), r * np.sin(elevation))), target, u=up, pose_spec=pose_spec))
    return c2ws


def gen_spherical_random_poses(r_lo, r_hi, elevation_lo, elevation_hi, azimuth_lo, azimuth_hi, roll_lo, roll_hi, G_box, s_box, n, u=None, pose_spec=2, pose_type='c2w'):
    """
    G_box: the pose of the target box in world coordinates. The origin of the box is at its center.
    s_box: the scale of the target box in world units.
    """
    if u is None:
        u = np.array((0, 0, 1))
    c2ws = []
    for _ in range(n):
        r = np.random.uniform(r_lo, r_hi)
        e = np.random.uniform(elevation_lo, elevation_hi)
        a = np.random.uniform(azimuth_lo, azimuth_hi)
        c = np.array((r * np.cos(e) * np.cos(a), r * np.cos(e) * np.sin(a), r * np.sin(e)))
        t = np.random.uniform(low=-0.5, high=0.5, size=3)
        t = (G_box @ homogenize_vecs(s_box * t))[:3, 0]
        roll = np.random.uniform(roll_lo, roll_hi)
        u_rolled = Rotation.from_rotvec(normalize_vecs(t - c)[0] * roll).apply(u)
        c2ws.append(gen_lookat_pose(c, t, u=u_rolled, pose_spec=pose_spec, pose_type=pose_type))
    return c2ws


def homogenize_vecs(vecs, val=1):
    """ homogenize vecs to be (..., 4, 1) """
    if vecs.shape[-1] != 1:
        vecs = vecs[..., None]
    s = vecs.shape
    if s[-2] == 4:
        return vecs
    if isinstance(vecs, np.ndarray):
        return np.concatenate((vecs, np.full((*s[:-2], 1, 1), val, dtype=vecs.dtype)), axis=-2)
    return torch.cat((vecs, torch.full((*s[:-2], 1, 1), val, dtype=vecs.dtype, device=vecs.device)), dim=-2)


def homogenize_transforms(T):
    """ homogenize transform matrices to be (..., 4, 4) """
    s = T.shape
    if s[-2:] == (4, 4):
        return T
    T_comp = np.zeros((*s[:-2], 4, 4), dtype=T.dtype) if isinstance(T, np.ndarray) else torch.zeros(*s[:-2], 4, 4, dtype=T.dtype, device=T.device)
    T_comp[..., :3, :s[-1]] = T
    T_comp[..., 3, 3] = 1
    return T_comp


def decompose_sim3(T):
    """ T: (..., 4, 4) """
    if isinstance(T, np.ndarray):
        P = T.copy()
        s = np.linalg.norm(P[..., :3], axis=-2)
        S = np.stack([np.diag((*s_, 1)) for s_ in s])
    else:
        P = T.clone()
        s = torch.linalg.norm(P[..., :3], dim=-2)
        S = torch.diag_embed(torch.cat((s, torch.ones((*s.shape[:-1], 1)).to(s)), dim=-1))
    P[..., :3, :3] /= s[..., None, :]
    return P, S


def compute_trans_diff(T1, T2):
    T = T2 @ np.linalg.inv(T1)
    G, s = decompose_sim3(T)
    R = Rotation.from_matrix(G[:3, :3])
    r = np.linalg.norm(R.as_rotvec()) / np.pi * 180
    t = np.linalg.norm(G[:3, 3])
    s = np.abs(np.log(s))
    return r, t, s


def avg_trans(Ts, s=None, avg_func=np.mean):
    T = avg_func(Ts, axis=0)
    u, s_, vh = np.linalg.svd(T[:3, :3])
    T[:3, :3] = u * (s if s else s_) @ vh
    return T


def img_cat(imgs, axis, interval=0, color=255):
    assert axis in {0, 1}, 'axis must be either 0 or 1'
    h, w, c = imgs[0].shape
    gap = np.broadcast_to(color, (h, interval, c) if axis else (interval, w, c)).astype(imgs[0].dtype)
    t = [gap] * (len(imgs) * 2 - 1)
    t[::2] = imgs
    return np.concatenate(t, axis=axis)


def draw_pt(img, pt, K, pose=None, pose_spec=2, pose_type='c2w', radius=10, color=(160, 160, 160), thickness=-1):
    """ Draw a point specified in world coordinates on a calibrated image. """
    if pose is None:
        pose = np.eye(4)
    elif pose_type == 'c2w':
        pose = np.linalg.inv(pose)
    pose = ch_cam_pose_spec(pose, pose_spec, 1, pose_type='w2c')
    pt_cam = pose @ homogenize_vecs(pt)
    pt_img = (K @ pt_cam[:3] / pt_cam[2])[:2, 0]
    cv2.circle(img, pt_img.astype(int), radius, color, thickness=thickness)


def dilute_image(img, *args, color=(255, 255, 255), alpha=128):
    if len(args) == 4:
        t, b, l, r = args
        if img.shape[2] == 3:
            img[t:b, l:r] = blend_images(color, alpha, img[t:b, l:r], 255)[0]
        else:
            img[t:b, l:r, :3], img[t:b, l:r, 3] = blend_images(color, alpha, img[t:b, l:r, :3], img[t:b, l:r, 3])
    elif len(args) == 1:
        if img.shape[2] == 3:
            img[args] = blend_images(color, alpha, img[args], 255)[0]
        else:
            img[args, :3], img[args, 3] = blend_images(color, alpha, img[args, :3], img[args, 3])


def blend_images(rgb_fg, a_fg, rgb_bg, a_bg):
    rgb_fg = np.asarray(rgb_fg, dtype=int)
    a_fg = np.asarray(a_fg)
    if a_fg.ndim == rgb_fg.ndim:
        a_fg = a_fg.squeeze(axis=-1)
    rgb_bg = np.asarray(rgb_bg, dtype=int)
    a_bg = np.asarray(a_bg)
    if a_bg.ndim == rgb_bg.ndim:
        a_bg = a_bg.squeeze(axis=-1)
    t = 1 - a_fg / 255
    a = a_fg + a_bg * t
    rgb = (rgb_fg * a_fg[..., None] + rgb_bg * a_bg[..., None] * t) / a[..., None]
    return rgb.astype(np.uint8), a.astype(np.uint8)


def to_np_img(img):
    if isinstance(img, torch.Tensor):
        if img.shape[0] <= 4:
            img = img.permute(1, 2, 0)
        img = img.cpu().numpy()
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return img


def to_pt_img(img, channel_first=False, device='cuda'):
    if isinstance(img, np.ndarray):
        if channel_first and img.shape[-1] <= 4:
            img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).to(device)
    if img.dtype == torch.uint8:
        img = img.float() / 255
    return img


def put_texts(img, txts, x0=10, y0=10, dy=10, offset=0, bg=None, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(80, 80, 80), thickness=2):
    if bg is not None:
        match np.size(bg):
            case 1:
                alpha = 128
                bg = np.full(3, bg)
            case 3:
                alpha = 128
            case 4:
                alpha = bg[3]
                bg = bg[:3]
    color = np.asarray(color)
    if np.size(color) == 1:
        color = np.broadcast_to(color, 3)
    if img.shape[2] == 4 and len(color) < 4:
        color = np.append(color, 255)
    color = color.tolist()
    if isinstance(txts, str):
        txts = (txts,)
    for txt in txts:
        (w, h), b = cv2.getTextSize(txt, font_face, font_scale, thickness)
        if bg is not None:
            dilute_image(img, y0, y0 + h + b + offset * 2, x0, x0 + w + offset * 2, color=bg, alpha=alpha)
        cv2.putText(img, txt, (x0 + offset, y0 + offset + h), font_face, font_scale, color, thickness=thickness)
        y0 += h + b + offset * 2 + dy
    return img


def pad_imgs(imgs, color=(0, 0, 0)):
    h_max, w_max = np.array([img.shape[:2] for img in imgs]).max(axis=0)
    return [cv2.copyMakeBorder(img, (h_max - img.shape[0]) // 2, -((img.shape[0] - h_max) // 2), (w_max - img.shape[1]) // 2, -((img.shape[1] - w_max) // 2), cv2.BORDER_CONSTANT, value=color) for img in imgs]


def colorize_depth(depth, colormap='viridis', **kwargs):
    normalized_depth = Normalize(**kwargs)(depth)
    colorized_depth = plt.get_cmap(colormap)(normalized_depth)
    return (colorized_depth[..., :3] * 255).astype(np.uint8)


def normalize_vecs(vecs, axis=-1):
    norms = np.linalg.norm(vecs, axis=axis, keepdims=True)
    return vecs / norms, norms[..., 0]


def pose_lerp(G0, G1, ts, pose_type='o2w'):
    """ Linearly interpolate/extrapolate from two poses. """
    if pose_type.startswith('w2'):
        G0, G1 = np.linalg.inv(G0), np.linalg.inv(G1)
    G10 = np.linalg.inv(G0) @ G1
    R10, t10 = G10[:3, :3], G10[:3, 3]
    R10 = Rotation.from_matrix(R10).as_rotvec()
    ts = np.asarray(ts)[..., None]
    Rt0 = Rotation.from_rotvec(R10 * ts).as_matrix()
    tt0 = t10 * ts
    Gt0 = homogenize_transforms(np.concatenate((Rt0, tt0[..., None]), axis=-1))
    Gt = G0 @ Gt0
    if pose_type.startswith('w2'):
        return np.linalg.inv(Gt)
    return Gt


def vid_cmp(vids, axis, annotations=None, color=(240, 0, 0)):
    if annotations is None:
        annotations = [''] * len(vids)
    return [img_cat([put_texts(img, annotation, color=color) for img, annotation in zip(imgs, annotations)], axis) for imgs in zip(*vids)]


def plt2img(axis_off=False):
    if axis_off:
        plt.axis('off')
    with BytesIO() as buffer:
        plt.savefig(buffer, bbox_inches='tight', pad_inches=0)
        plt.close()
        return imread(buffer)
