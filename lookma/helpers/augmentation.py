import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class AddISONoise(nn.Module):
    def forward(self, img, return_debug_info=False):
        info = {"applied": False, "sigma_read": 0.0, "gain": 0.0}

        if torch.rand(1) > 0.5:
            if return_debug_info:
                return img, info
            return img

        sigma_read = torch.rand(1, device=img.device) * 0.03
        gain = torch.rand(1, device=img.device) * 0.03
        shot_std = torch.sqrt(torch.clamp(img, min=1e-5)) * gain

        noisy = torch.clamp(
            img + torch.randn_like(img) * sigma_read + torch.randn_like(img) * shot_std,
            0,
            1,
        )

        if return_debug_info:
            info["applied"] = True
            info["sigma_read"] = sigma_read.item()
            info["gain"] = gain.item()
            return noisy, info
        return noisy


class Pixelate(nn.Module):
    def forward(self, img, return_debug_info=False):
        info = {"applied": False, "downsample_factor": 1}

        if torch.rand(1) < 0.3:
            B, C, H, W = img.shape
            factor = np.random.randint(2, 4)
            small = torch.nn.functional.interpolate(
                img, scale_factor=1 / factor, mode="nearest"
            )
            pixelated = torch.nn.functional.interpolate(
                small, size=(H, W), mode="nearest"
            )

            if return_debug_info:
                info["applied"] = True
                info["downsample_factor"] = int(factor)
                return pixelated, info
            return pixelated

        if return_debug_info:
            return img, info
        return img


class MotionBlur(nn.Module):
    def forward(self, img, return_debug_info=False):
        info = {"applied": False, "angle": 0.0, "kernel_size": 0}

        if torch.rand(1) < 0.5:
            # Uniformly sample kernel sizes [1..10]
            # Even kernels allowed per user request.
            k = int(torch.randint(1, 11, (1,)).item())

            angle = torch.rand(1).item() * 180

            # Create a horizontal line kernel
            kernel = torch.zeros((k, k), device=img.device)
            center = k // 2
            kernel[center, :] = 1.0

            # Rotate kernel matrix
            # Expand to [1, 1, k, k] to use TF.rotate
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            kernel = TF.rotate(kernel, angle)
            kernel = kernel.squeeze()

            # Normalize
            kernel = kernel / kernel.sum()

            # Apply convolution
            # img is [B, C, H, W]
            # Kernel needs to be [C, 1/groups, k, k] -> [3, 1, k, k] for depthwise conv
            C = img.shape[1]
            weight = kernel.expand(C, 1, k, k)

            # Pad image to maintain size
            pad = k // 2
            blurred = torch.nn.functional.conv2d(img, weight, padding=pad, groups=C)

            # Fix dimensions for even kernels (padding=k//2 results in +1 size)
            if k % 2 == 0:
                blurred = blurred[..., : img.shape[2], : img.shape[3]]

            if return_debug_info:
                info["applied"] = True
                info["angle"] = angle
                info["kernel_size"] = k
                return blurred, info
            return blurred

        if return_debug_info:
            return img, info
        return img


class TrainingAugmentation(nn.Module):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
        super().__init__()
        # Store ranges for manual sampling
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        self.color_aug = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.iso_noise = AddISONoise()
        self.pixelate = Pixelate()
        self.motion_blur = MotionBlur()

    def forward(self, img, return_debug_info=False):
        if not return_debug_info:
            # Fast path
            img = self.color_aug(img)
            img = self.motion_blur(img, return_debug_info=False)
            img = self.iso_noise(img, return_debug_info=False)
            img = self.pixelate(img, return_debug_info=False)
            return img

        # Debug path with manual parameter tracking
        jitter_info = {
            "brightness": None,
            "contrast": None,
            "saturation": None,
            "hue": None,
        }

        # Sample parameters for ColorJitter
        fn_idx, b, c, s, h = T.ColorJitter.get_params(
            self.color_aug.brightness,
            self.color_aug.contrast,
            self.color_aug.saturation,
            self.color_aug.hue,
        )

        # Apply transforms in sampled order
        img_aug = img
        for fn_id in fn_idx:
            if fn_id == 0 and b is not None:
                img_aug = TF.adjust_brightness(img_aug, b)
                jitter_info["brightness"] = b
            elif fn_id == 1 and c is not None:
                img_aug = TF.adjust_contrast(img_aug, c)
                jitter_info["contrast"] = c
            elif fn_id == 2 and s is not None:
                img_aug = TF.adjust_saturation(img_aug, s)
                jitter_info["saturation"] = s
            elif fn_id == 3 and h is not None:
                img_aug = TF.adjust_hue(img_aug, h)
                jitter_info["hue"] = h

        # Apply Motion Blur (First, before Noise/Pixelate)
        img_aug, blur_info = self.motion_blur(img_aug, return_debug_info=True)

        # Apply ISO Noise
        img_aug, iso_info = self.iso_noise(img_aug, return_debug_info=True)

        # Apply Pixelate
        img_aug, pix_info = self.pixelate(img_aug, return_debug_info=True)

        return img_aug, {
            "color_jitter": jitter_info,
            "iso_noise": iso_info,
            "pixelate": pix_info,
            "motion_blur": blur_info,
        }
