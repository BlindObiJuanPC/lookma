import glob
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SynthBodyDataset(Dataset):
    # Every 5th vertex of the 6890 vertices in SMPL
    DENSE_LANDMARK_IDS = list(range(0, 6890, 5))

    # Manually selected landmarks used for finding a crop box around the body.
    ROI_LANDMARK_IDS = [
        476,
        1488,
        3439,
        3676,
        4098,
        4290,
        4291,
        4858,
        4933,
        5059,
        5287,
        5361,
        5530,
        5627,
        5645,
        5775,
        5934,
        6200,
        6437,
        6842,
        7032,
        7594,
        7669,
        7795,
        8023,
        8095,
        8247,
        8321,
        8339,
        8469,
        8635,
        8751,
        8847,
        8966,
        9003,
        9008,
    ]

    def __init__(
        self,
        root_dir,
        specific_image=None,
        target_size=256,
        is_train=True,
        return_debug_info=False,
    ):
        self.root_dir = root_dir
        self.target_size = target_size
        self.is_train = is_train  # Turns augmentation on/off
        self.return_debug_info = return_debug_info

        print("Loading metadata JSON files...")
        if specific_image:
            target_meta = specific_image.replace("img_", "metadata_").replace(
                ".jpg", ".json"
            )
            self.json_paths = [os.path.join(root_dir, target_meta)]
        else:
            self.json_paths = sorted(
                glob.glob(os.path.join(root_dir, "metadata_*.json"))
            )

    def __len__(self):
        return len(self.json_paths)

    def get_aug_params(self, base_size):
        # Default: Perfect Center, 1.0 Scale, 0 Rotation
        rot = 0
        scale = 1.0
        shift_x = 0.0
        shift_y = 0.0

        if self.is_train:
            # Rotation: +/- 30 degrees (60% chance) - Conservative for Body
            if np.random.rand() < 0.6:
                rot_factor = 30
                rot = np.clip(
                    np.random.randn() * rot_factor, -2 * rot_factor, 2 * rot_factor
                )
            # rot = 0
            # Scale: 0.8x (Zoom In) to 1.2x (Zoom Out)
            # Zooming in simulates "Waist Up" shots (legs get cut off)
            scale_factor = 0.4
            scale = 1.0 + (np.random.rand() - 0.5) * 2 * scale_factor

            # Shift: Move center +/- 10%
            shift_factor = 0.1
            shift_x = (np.random.rand() - 0.5) * 2 * shift_factor
            shift_y = (np.random.rand() - 0.5) * 2 * shift_factor

        return rot, scale, shift_x, shift_y

    def __getitem__(self, idx):
        try:
            json_path = self.json_paths[idx]
            with open(json_path, "r") as f:
                meta = json.load(f)

            base_name = (
                os.path.basename(json_path)
                .replace("metadata_", "img_")
                .replace(".json", ".jpg")
            )
            img_path = os.path.join(self.root_dir, base_name)

            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            # Raw Labels
            cam_intrinsics = np.array(
                meta["camera"]["camera_to_image"], dtype=np.float32
            )
            cam_extrinsics = torch.tensor(
                meta["camera"]["world_to_camera"], dtype=torch.float32
            )
            pose = torch.tensor(meta["pose"], dtype=torch.float32).flatten()
            betas = torch.tensor(meta["body_identity"], dtype=torch.float32)
            trans = torch.tensor(meta["translation"], dtype=torch.float32)
            landmarks_2d = np.array(meta["landmarks"]["2D"], dtype=np.float32)

            # --- CALCULATE CROP BOX ---
            min_x, max_x = np.min(landmarks_2d[:, 0]), np.max(landmarks_2d[:, 0])
            min_y, max_y = np.min(landmarks_2d[:, 1]), np.max(landmarks_2d[:, 1])
            center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

            width = max_x - min_x
            height = max_y - min_y
            base_size = max(width, height) * 1.2  # 1.2x padding standard

            # --- AUGMENTATION CALCS ---
            rot, scale_aug, shift_x, shift_y = self.get_aug_params(base_size)

            # Apply shifts to center
            # Note: We shift the center relative to the size of the person
            center_x += base_size * shift_x
            center_y += base_size * shift_y

            # Apply scale to size
            # Note: We invert scale here.
            # If scale_aug = 1.2 (Zoom Out), the crop box must get BIGGER.
            # If scale_aug = 0.8 (Zoom In), the crop box must get SMALLER.
            proc_size = base_size / scale_aug

            # --- AFFINE TRANSFORM ---
            # Define destination dimensions
            dst_size = self.target_size
            dst_center = dst_size / 2.0

            # Calculate the Scale Ratio (Output / Input)
            # This variable is needed for both Matrix M and updating Intrinsics later
            scale_ratio = dst_size / proc_size

            # Create Rotation/Scale matrix centered on the PERSON
            # getRotationMatrix2D handles rotation around a point AND scaling
            M = cv2.getRotationMatrix2D((center_x, center_y), rot, scale_ratio)

            # Adjust Translation to center the result in the OUTPUT image
            # We shift the matrix so the person moves from 'center_x' to 'dst_center'
            M[0, 2] += dst_center - center_x
            M[1, 2] += dst_center - center_y

            # Warp Image
            crop_image = cv2.warpAffine(
                image,
                M,
                (dst_size, dst_size),
                flags=cv2.INTER_LINEAR,
                borderValue=(0, 0, 0),
            )

            # Warp Landmarks (Apply same M)
            ones = np.ones((landmarks_2d.shape[0], 1))
            landmarks_homo = np.hstack([landmarks_2d, ones])
            landmarks_2d_new = np.dot(M, landmarks_homo.T).T

            # --- UPDATE INTRINSICS (K) ---
            # scale_ratio is now correctly defined above
            new_intrinsics = cam_intrinsics.copy()
            new_intrinsics[0, 0] *= scale_ratio  # fx
            new_intrinsics[1, 1] *= scale_ratio  # fy

            # Update principal point
            # Start with old center in homogenous coords
            old_center = np.array([cam_intrinsics[0, 2], cam_intrinsics[1, 2], 1.0])
            # Apply the affine transform to the center pixel
            new_center = np.dot(M, old_center)

            new_intrinsics[0, 2] = new_center[0]
            new_intrinsics[1, 2] = new_center[1]

            # --- UPDATE EXTRINSICS (R, T) ---
            # If we rotated the image, we must rotate the World->Camera transform.
            # Rotating the image by 'rot' (degrees, CCW) is equivalent to rolling the camera
            # by 'rot'. We apply this rotation to the Extrinsics.
            # Point_cam = Ext * Point_world
            # Point_cam_new = R_z(rot) * Point_cam
            #               = R_z(rot) * Ext * Point_world
            # So New_Ext = R_z(rot) * Old_Ext
            if rot != 0:
                rad = -np.deg2rad(
                    rot
                )  # Image rotates CCW -> Camera axis frame rotates CW?
                # Actually, if we rotate the PIXELS CCW (Standard Image Rotation),
                # A point at (1,0) moves to (cos, sin).
                # This corresponds to rotating the camera frame CCW around Z?
                # Image rotates CCW -> Camera axis frame must rotate CW (or vice versa depending on coord system).
                # User reported previous (positive) was inverted. Swapping to negative.
                rad = np.deg2rad(-rot)
                cos_a = np.cos(rad)
                sin_a = np.sin(rad)
                # Z-axis rotation matrix (3x3)
                R_z = torch.tensor(
                    [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]],
                    dtype=torch.float32,
                )
                # Extrinsics is 4x4 (from metadata? No, tensor is typically 4x4 for world_to_camera)
                # Checking: meta["camera"]["world_to_camera"] is list of lists, usually 4x4.
                # Lines 84-86: cam_extrinsics = torch.tensor(..., dtype=torch.float32)

                # Apply Rotation to the 4x4 matrix
                # New_Ext = | R_z  0 | * Old_Ext
                #           |  0   1 |
                # The Translation part IS affected because we are rotating the frame
                # around the origin (0,0,0) of the Camera Frame.
                # So yes, we simply matmul the top 3 rows.

                cam_extrinsics[:3, :] = torch.matmul(R_z, cam_extrinsics[:3, :])

            if self.return_debug_info:
                return {
                    "image": torch.from_numpy(crop_image).permute(2, 0, 1).float(),
                    "pose": pose,
                    "betas": betas,
                    "trans": trans,
                    "cam_intrinsics": torch.from_numpy(new_intrinsics),
                    "cam_extrinsics": cam_extrinsics,
                    "landmarks_2d": torch.from_numpy(landmarks_2d_new),
                    # Debug info
                    "original_image": image,  # The raw loaded image (no crop, no resize)
                    "aug_params": {
                        "rot": rot,
                        "scale": scale_aug,  # Use the selected scale factor
                        "shift_x": shift_x,
                        "shift_y": shift_y,
                    },
                }

            return {
                "image": torch.from_numpy(crop_image).permute(2, 0, 1).float(),
                "pose": pose,
                "betas": betas,
                "trans": trans,
                "cam_intrinsics": torch.from_numpy(new_intrinsics),
                "cam_extrinsics": cam_extrinsics,
                "landmarks_2d": torch.from_numpy(landmarks_2d_new),
            }

        except Exception as e:
            # Fallback for corrupt files
            print(f"Error loading {self.json_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))


class SynthHandDataset(SynthBodyDataset):
    # Manually selected hand vertices (based on the image in the paper)
    DENSE_LANDMARK_IDS = [
        1982,
        1984,
        1986,
        1988,
        1991,
        1994,
        1999,
        2000,
        2006,
        2016,
        2017,
        2019,
        2024,
        2030,
        2035,
        2040,
        2046,
        2050,
        2052,
        2055,
        2059,
        2067,
        2070,
        2080,
        2081,
        2085,
        2095,
        2096,
        2099,
        2101,
        2107,
        2120,
        2126,
        2132,
        2133,
        2134,
        2144,
        2150,
        2154,
        2160,
        2166,
        2171,
        2173,
        2175,
        2180,
        2186,
        2195,
        2204,
        2209,
        2211,
        2218,
        2223,
        2224,
        2234,
        2246,
        2247,
        2249,
        2250,
        2251,
        2255,
        2258,
        2262,
        2263,
        2264,
        2266,
        2268,
        2270,
        2274,
        2275,
        2289,
        2292,
        2295,
        2298,
        2300,
        2311,
        2319,
        2323,
        2337,
        2342,
        2344,
        2360,
        2364,
        2365,
        2370,
        2375,
        2387,
        2389,
        2393,
        2407,
        2410,
        2413,
        2423,
        2433,
        2445,
        2454,
        2456,
        2459,
        2472,
        2477,
        2484,
        2492,
        2496,
        2497,
        2499,
        2504,
        2518,
        2522,
        2523,
        2534,
        2547,
        2556,
        2560,
        2565,
        2567,
        2583,
        2587,
        2590,
        2593,
        2606,
        2611,
        2619,
        2623,
        2625,
        2628,
        2629,
        2634,
        2651,
        2661,
        2673,
        2677,
        2684,
        2708,
        2711,
        2713,
        2724,
        2734,
        2745,
        2750,
        2755,
        2757,
        5681,
    ]

    def __init__(
        self,
        root_dir,
        specific_image=None,
        target_size=128,
        is_train=True,
        return_debug_info=False,
    ):
        super().__init__(
            root_dir, specific_image, target_size, is_train, return_debug_info
        )

    def get_aug_params(self, base_size):
        # Override for Hand: Aggressive Rotation (+/- 180)
        rot = 0
        scale = 1.0
        shift_x = 0.0
        shift_y = 0.0

        if self.is_train:
            # Rotation: +/- 180 degrees (60% chance) - Aggressive for Hand
            if np.random.rand() < 0.6:
                rot_factor = 60
                rot = np.clip(np.random.randn() * rot_factor, -180, 180)

            # Scale: 0.8x to 1.2x (Same as Body)
            scale_factor = 0.4
            scale = 1.0 + (np.random.rand() - 0.5) * 2 * scale_factor

            # Shift: +/- 10% (Same as Body)
            shift_factor = 0.1
            shift_x = (np.random.rand() - 0.5) * 2 * shift_factor
            shift_y = (np.random.rand() - 0.5) * 2 * shift_factor

        return rot, scale, shift_x, shift_y
