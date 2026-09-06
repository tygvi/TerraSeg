import gradio as gr
import torch
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image
import torchvision.transforms as T

# Class info
CLASS_NAMES = ['urban', 'agriculture', 'rangeland', 'forest', 'water', 'barren']
CLASS_COLORS = np.array([
    [0, 255, 255], [255, 255, 0], [255, 0, 255],
    [0, 255, 0], [0, 0, 255], [255, 255, 255]
])

# Load model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=6)
model.load_state_dict(torch.load('best_unet_256.pth', map_location=device))
model.to(device)
model.eval()

# Preprocessing
normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def mask_to_rgb(mask):
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(CLASS_COLORS):
        rgb[mask == i] = color
    return rgb

def predict(image):
    # Preprocess
    img = Image.fromarray(image).resize((256, 256))
    img_tensor = normalize(T.ToTensor()(img)).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred = model(img_tensor).argmax(dim=1).squeeze().cpu().numpy()

    # Convert to RGB
    pred_rgb = mask_to_rgb(pred)
    pred_rgb = Image.fromarray(pred_rgb).resize((image.shape[1], image.shape[0]))

    return pred_rgb

# Launch
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Satellite Image"),
    outputs=gr.Image(label="Segmentation Mask"),
    title="DeepGlobe Land Cover Segmentation",
    description="U-Net with pretrained ResNet34 encoder for satellite imagery segmentation. Classes: urban, agriculture, rangeland, forest, water, barren.",
    examples=None
)

if __name__ == "__main__":
    demo.launch()
