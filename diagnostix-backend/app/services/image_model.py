import torch
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18
from PIL import Image
from torchvision import transforms
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same preprocessing as training
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def build_model(num_classes: int):
    """Create ResNet18 architecture."""
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_image_model(model_path: str, num_classes: int):
    """Load model from state_dict safely (no pickle)."""
    model = build_model(num_classes)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_bytes: bytes):
    """Convert uploaded file to a tensor."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0).to(device)
