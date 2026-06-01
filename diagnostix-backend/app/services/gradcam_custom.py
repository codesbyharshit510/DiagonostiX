# gradcam_custom.py
"""
Grad-CAM utility for ResNet-style models.

Function:
    generate_gradcam(model, input_tensor, target_layer, target_category=None) -> base64_png_str

Notes:
- input_tensor expected shape: (1, C, H, W), normalized with ImageNet mean/std (same preprocessing as used for inference).
- target_layer: a nn.Module (e.g. model.layer4[-1]) to attach hooks to.
- Returns: base64-encoded PNG string (RGB) of the overlayed heatmap.
"""

from __future__ import annotations
import io
import base64
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import matplotlib.cm as cm  # matplotlib is commonly available in ML envs


# ImageNet normalization used in your preprocessing
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """
    Convert a single 1xC x H x W tensor (normalized) back to a PIL Image (RGB).
    """
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img = img_tensor.detach().cpu().numpy()
    # img shape C,H,W -> H,W,C
    img = np.transpose(img, (1, 2, 0))
    # unnormalize
    img = (img * _IMAGENET_STD) + _IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    img = (img * 255.0).astype(np.uint8)
    return Image.fromarray(img)


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    """Normalize CAM to [0,1]"""
    cam = np.maximum(cam, 0)
    maxv = cam.max()
    if maxv == 0:
        return cam
    return cam / maxv


def _apply_colormap_on_image(orig_img: Image.Image, cam: np.ndarray, colormap_name: str = "jet", alpha: float = 0.45) -> Image.Image:
    """
    Apply heatmap (cam) onto original PIL image.
    cam: HxW numpy array with values in [0,1]
    """
    # Ensure cam is HxW float32 in [0,1]
    cam_uint8 = np.uint8(255 * cam)
    # Use matplotlib colormap
    cmap = cm.get_cmap(colormap_name)
    colored_cam = cmap(cam_uint8 / 255.0)[:, :, :3]  # HxWx3 (float in [0,1])
    colored_cam = np.uint8(255 * colored_cam)
    heatmap = Image.fromarray(colored_cam).convert("RGBA")

    # Resize heatmap to original image size if needed
    if heatmap.size != orig_img.size:
        heatmap = heatmap.resize(orig_img.size, resample=Image.BILINEAR)

    # Make sure original image is RGBA
    base = orig_img.convert("RGBA")
    # Blend
    blended = Image.blend(base, heatmap, alpha=alpha)
    # Optional: make nicer by applying slight edge smoothing
    return blended.convert("RGB")


def generate_gradcam(model: torch.nn.Module, input_tensor: torch.Tensor, target_layer: torch.nn.Module, target_category: Optional[int] = None) -> str:
    """
    Compute Grad-CAM for the given model, input, and target layer.
    Returns a base64-encoded PNG string of the heatmap overlay.
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Storage for activations and gradients
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        # grad_out is a tuple
        gradients["value"] = grad_out[0].detach()

    # Register hooks
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    out = model(input_tensor)  # shape: (1, num_classes) or (1,) for single logit

    # Determine target category
    if target_category is None:
        if out.dim() == 2 and out.shape[1] > 1:
            target_category = int(out.argmax(dim=1).item())
        else:
            # if single output, use index 0
            target_category = 0

    # Backward on the target logit
    model.zero_grad()
    # If multi-class, pick the logit; if single logit/regression, take out[0]
    if out.dim() == 2 and out.shape[1] > 1:
        score = out[0, target_category]
    else:
        # For single-output models (rare here), take scalar
        score = out.flatten()[0]
    score.backward(retain_graph=True)

    # Pull stored values
    if "value" not in activations or "value" not in gradients:
        # cleanup hooks
        fh.remove()
        bh.remove()
        raise RuntimeError("Grad-CAM hooks failed to capture activations/gradients.")

    act = activations["value"]          # shape: (1, C, H, W)
    grad = gradients["value"]           # shape: (1, C, H, W)

    # Remove hooks
    fh.remove()
    bh.remove()

    # Global average pooling of gradients -> channel weights
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)   # shape: (1, C, 1, 1)

    # Weighted sum of activations
    cam = torch.sum(weights * act, dim=1, keepdim=True)    # shape: (1, 1, H, W)
    cam = cam.squeeze().cpu().numpy()                      # H x W

    # Normalize
    cam = _normalize_cam(cam)

    # Upsample CAM to input image size
    input_h = input_tensor.shape[2]
    input_w = input_tensor.shape[3]
    cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).to(device)  # 1x1xHcamxWcam
    cam_upsampled = F.interpolate(cam_tensor, size=(input_h, input_w), mode="bilinear", align_corners=False)
    cam_upsampled = cam_upsampled.squeeze().cpu().numpy()
    cam_upsampled = _normalize_cam(cam_upsampled)

    # Convert input_tensor back to PIL image for overlay
    orig_img = _tensor_to_pil(input_tensor.cpu())

    # Apply colormap and overlay
    try:
        overlay = _apply_colormap_on_image(orig_img, cam_upsampled, colormap_name="jet", alpha=0.45)
    except Exception:
        # fallback: simple red overlay
        heat = np.uint8(255 * cam_upsampled)
        heat = np.stack([heat, np.zeros_like(heat), np.zeros_like(heat)], axis=2)
        heat_img = Image.fromarray(heat).resize(orig_img.size)
        overlay = Image.blend(orig_img.convert("RGBA"), heat_img.convert("RGBA"), alpha=0.45).convert("RGB")

    # Optionally, add a small gaussian blur to make the heatmap softer
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Encode to base64 PNG
    buf = io.BytesIO()
    overlay.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64
