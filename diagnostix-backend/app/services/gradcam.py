import numpy as np
import base64
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from io import BytesIO
from PIL import Image
import torch

def generate_gradcam(model, input_tensor, target_layer, target_category=None):
    use_cuda = torch.cuda.is_available()
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=use_cuda)
    targets = None
    if target_category is not None:
        # lazy import to avoid hard dependency if not used
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(int(target_category))]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    # convert input tensor to image for overlay
    img_np = input_tensor.squeeze().permute(1,2,0).cpu().numpy()
    # normalize to 0-1
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    pil_img = Image.fromarray(visualization)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
