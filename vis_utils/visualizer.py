import logging
from collections import defaultdict

import numpy as np
import open3d as o3d

from vis_utils.line_mesh import LineMesh
from vis_utils.utils import ch_pose_spec


class Visualizer:
    def __init__(self, visible=True, cam_info=None, frame_scale=0, pt_size=1) -> None:
        self.o3d_vis = o3d.visualization.Visualizer()
        self.visible = visible
        self.cam_info = defaultdict(lambda: None)
        if cam_info:
            self.cam_info.update(cam_info)
        if self.cam_info['width'] is None:
            self.cam_info['width'] = 1920
        if self.cam_info['height'] is None:
            self.cam_info['height'] = 1080
        self.o3d_vis.create_window(**{cam_param: self.cam_info[cam_param] for cam_param in ('width', 'height')}, visible=visible)
        self.view_ctrl = self.o3d_vis.get_view_control()
        self.apply_cam_info()
        self.o3d_vis.get_render_option().point_size = pt_size
        if frame_scale:
            self.o3d_vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_scale))

    def add_trajectory(self, *poses, pose_spec=2, pose_type='c2w', K_invs=None, ws=None, hs=None, cam_size=1, line_width=None, color=(0.5, 0.5, 0.5), connect_cams=False):
        """ pose_spec:
                0: x->right, y->front, z->up
                1: x->right, y->down, z->front
                2: x->right, y->up, z->back """
        if len(poses) == 1:
            Rs = [pose[:3, :3] for pose in poses[0]]
            ts = [pose[:3, 3] for pose in poses[0]]
        else:
            Rs, ts = poses
        n_cams = len(Rs)
        if K_invs is None:
            K_invs = np.linalg.inv(np.array(((1000, 0, 400),
                                             (0, 1000, 300),
                                             (0, 0, 1))))
            ws = 800
            hs = 600
        if np.ndim(K_invs) == 2:
            K_invs = np.broadcast_to(K_invs, (n_cams, 3, 3))
            hs = np.broadcast_to(hs, n_cams)
            ws = np.broadcast_to(ws, n_cams)
        if np.ndim(color) == 1:
            color = np.broadcast_to(color, (n_cams, len(color)))
        pts = np.zeros((n_cams, 4, 3, 1))
        pts[:, 1, 0, 0] = ws
        pts[:, 2, 1, 0] = hs
        pts[:, 3, 0, 0] = ws
        pts[:, 3, 1, 0] = hs
        pts[..., 2, 0] = 1
        points = []
        lines = []
        colors = []
        for i, (R, t) in enumerate(zip(Rs, ts)):
            if pose_type == 'w2c':
                R = R.T
                t = -R @ t
            R = ch_pose_spec(R, pose_spec, 1)
            cam_pts = np.vstack((t, (R @ K_invs[i] @ pts[i])[..., 0] * cam_size + t))
            cam_ls = np.array(((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 4), (3, 4)))
            points.extend(cam_pts)
            if connect_cams and len(lines):
                cam_ls = np.vstack((cam_ls, (-5, 0)))
            lines.extend(i * 5 + cam_ls)
            colors.extend((color[i],) * len(cam_ls))
        if line_width is None:
            ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points), lines=o3d.utility.Vector2iVector(lines))
            ls.colors = o3d.utility.Vector3dVector(colors)
            self.o3d_vis.add_geometry(ls)
        else:
            lm = LineMesh(points, lines, colors, radius=line_width / 2)
            self.o3d_vis.add_geometry(lm.geom)

    def add_points(self, pts, color=(0.5, 0, 0.5)):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.colors = o3d.utility.Vector3dVector(np.broadcast_to(color, (len(pts), 3)))
        self.o3d_vis.add_geometry(pcd)

    def add_point_cloud(self, pcd, color=None):
        if not isinstance(pcd, o3d.geometry.PointCloud):
            pcd = o3d.io.read_point_cloud(pcd)
        if color is not None:
            pcd.colors = o3d.utility.Vector3dVector(np.broadcast_to(color, (len(pcd.colors), 3)))
        self.o3d_vis.add_geometry(pcd)

    def add_mesh(self, mesh):
        if not isinstance(mesh, o3d.geometry.TriangleMesh):
            mesh = o3d.io.read_triangle_mesh(mesh)
        self.o3d_vis.add_geometry(mesh)

    def apply_cam_info(self, cam_info=None):
        f = False
        if cam_info is None:
            if {'width', 'height', 'fx', 'fy', 'cx', 'cy'}.issubset(self.cam_info):
                f = True
            else:
                logging.warning(f'self.cam_info ({self.cam_info}) is incomplete.')
        elif {'width', 'height', 'fx', 'fy', 'cx', 'cy'}.issubset(cam_info) and cam_info['width'] == self.cam_info['width'] and cam_info['height'] == self.cam_info['height']:
            self.cam_info.update(cam_info)
            f = True
        else:
            logging.warning(f'cam_info ({cam_info}) is incomplete or inconsistent.')
        if f:
            cam_params = self.view_ctrl.convert_to_pinhole_camera_parameters()
            cam_params.intrinsic.set_intrinsics(self.cam_info['width'], self.cam_info['height'], self.cam_info['fx'], self.cam_info['fy'], self.cam_info['cx'] - 0.5, self.cam_info['cy'] - 0.5)
            self.view_ctrl.convert_from_pinhole_camera_parameters(cam_params)

    def transform(self, T, relative=True):
        cam_params = self.view_ctrl.convert_to_pinhole_camera_parameters()
        cam_params.extrinsic = T @ (cam_params.extrinsic if relative else np.identity(4))
        self.view_ctrl.convert_from_pinhole_camera_parameters(cam_params)

    def capture_screen(self, do_render=False):
        return (np.asarray(self.o3d_vis.capture_screen_float_buffer(do_render=do_render)) * 255).astype(np.uint8)

    def show(self):
        if not self.visible:
            logging.error('The visualizer is not set to be visible!')
            return
        self.o3d_vis.run()
        self.destroy()

    def destroy(self):
        self.o3d_vis.destroy_window()
        del self.view_ctrl
        del self.o3d_vis


if __name__ == '__main__':
    vis = Visualizer(frame_scale=1)
    vis.show()
