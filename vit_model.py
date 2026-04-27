import torch
import timm
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model ─────────────────────────────────────────────────────────────────────
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=4)
checkpoint = torch.load("best_vit_finetuned.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# ── Classes — ImageFolder sorts alphabetically ────────────────────────────────
# Your train folder has: glioma / meningioma / notumor / pituitary
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ── Validation transform (MUST match val_tfms in your notebook exactly) ───────
val_transform = transforms.Compose([
    transforms.Resize(int(224 * 1.14)),          # 255
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image):
    """Returns 'ClassName (confidence%)' string."""
    img = image.convert("RGB")
    img = val_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs  = torch.softmax(output, dim=1)[0]
        pred   = probs.argmax().item()
        conf   = probs[pred].item() * 100

    return CLASS_NAMES[pred], conf