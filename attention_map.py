import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Same val transform as training
val_transform = transforms.Compose([
    transforms.Resize(int(224 * 1.14)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def generate_attention_map(model, image, device):
    model.eval()

    img = image.convert("RGB")
    img_t = val_transform(img).unsqueeze(0).to(device)
    img_t.requires_grad_(True)

    output = model(img_t)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, pred_class].backward()

    gradients = img_t.grad.detach().cpu().numpy()[0]   # (3,224,224)
    attn_map  = np.mean(np.abs(gradients), axis=0)     # (224,224)

    attn_map -= attn_map.min()
    attn_map /= (attn_map.max() + 1e-8)

    return attn_map