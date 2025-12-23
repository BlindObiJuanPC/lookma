import numpy as np
import trimesh
import pyrender


class MeshRenderer:
    def __init__(self, width=256, height=256, device="cuda"):
        self.width = width
        self.height = height
        # WGL (Standard Windows) is default when EGL is removed
        try:
            self.renderer = pyrender.OffscreenRenderer(
                viewport_width=width, viewport_height=height
            )
        except Exception:
            self.renderer = None

    def render_mesh(self, image_rgb, vertices, faces, K=None):
        if self.renderer is None:
            return image_rgb

        # 1. Prepare Geometry
        verts = vertices.copy()

        # --- THE STANDARD BRIDGE ---
        # OpenCV Space:  +X Right, +Y Down, +Z Forward
        # OpenGL Native: +X Right, +Y Up,   -Z Forward

        # We rotate the mesh 180 degrees around the X-axis to bridge them.
        # Mathematically: Y -> -Y and Z -> -Z
        verts[:, 1] *= -1
        verts[:, 2] *= -1

        # Since we rotated exactly two axes, the winding order is preserved.
        # THE MESH WILL NOT BE TWISTED OR MIRRORED.
        # ---------------------------

        mesh = trimesh.Trimesh(verts, faces, process=False)
        scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.4, 0.4, 0.4)
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, alphaMode="OPAQUE", baseColorFactor=(0.3, 0.3, 1.0, 1.0)
        )
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=material))

        # 2. Setup Camera
        # We use the Identity matrix (Camera at origin, looking down -Z).
        # Because we flipped the mesh vertices to -Z, the camera sees them perfectly.
        camera_pose = np.eye(4)

        if K is not None:
            # We use the focal lengths directly.
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
        else:
            fx = fy = 500.0
            cx, cy = self.width / 2, self.height / 2

        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        scene.add(camera, pose=camera_pose)

        # 3. Lighting (Fixed at camera position)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=camera_pose)

        # 4. Render
        color, _ = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        # --- IMPORTANT ---
        # We DO NOT use np.flipud() here.
        # By flipping the Y-vertices and the principal point math,
        # the buffer will already be oriented correctly for OpenCV.
        # -----------------

        # 5. Composite
        mask = color[:, :, 3] > 0
        output = image_rgb.copy()
        output[mask] = output[mask] * 0.3 + color[mask, :3] * 0.7

        return output

    def delete(self):
        if self.renderer:
            self.renderer.delete()
