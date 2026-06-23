import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

WEIGHTS_PATH = "models/mobilenet_seatbelt_best.pt" # dummy path for now, maybe user has it elsewhere

def build_model():
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clf_model = build_model().to(device)
import os
if os.path.exists(WEIGHTS_PATH):
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    clf_model.load_state_dict(checkpoint["model"])
else:
    print("Warning: Seatbelt CNN weights not found. Using untrained weights.")

clf_model.eval()

clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.inference_mode()
def classify_seatbelt(pil_crop):
    tensor = clf_transform(pil_crop).unsqueeze(0).to(device)
    prob_worn = torch.softmax(clf_model(tensor), dim=1)[0, 1].item()
    return prob_worn >= 0.5, prob_worn
