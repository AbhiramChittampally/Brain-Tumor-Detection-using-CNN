import os
from flask import Flask, request, render_template, jsonify
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import io
import numpy as np

app = Flask(__name__)

# --- Model Definition (must match your training script) ---
class ResNetFineTuned(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetFineTuned, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze layer3, layer4, and fc
        for name, child in self.model.named_children():
            if name in ['layer3', 'layer4', 'fc']:
                for param in child.parameters():
                    param.requires_grad = True
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# --- Load the Model ---
MODEL_PATH = 'resnet18_brain_tumor_finetuned.pth' # Make sure this path is correct
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetFineTuned(num_classes=4).to(device)
try:
    # Ensure the model is loaded correctly. If it was saved with DataParallel, need to adjust.
    state_dict = torch.load(MODEL_PATH, map_location=device)
    # If the model was saved with DataParallel, keys might have 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"Model loaded successfully from {MODEL_PATH} on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    # Handle the error, perhaps exit or set a flag to prevent predictions

# --- Image Transformations (must match your training transform) ---
preprocess = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Class Names (based on your dataset) ---
# Make sure these match the order your ImageFolder assigned during training
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(device) # Add a batch dimension

            with torch.no_grad():
                output = model(input_batch)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            predicted_class = CLASS_NAMES[predicted_idx.item()]
            confidence_score = confidence.item() * 100 # Convert to percentage

            return jsonify({
                'prediction': predicted_class,
                'confidence': f'{confidence_score:.2f}%'
            })
        except Exception as e:
            return jsonify({'error': f'Prediction error: {e}'})

if __name__ == '__main__':
    app.run(debug=True) # Set debug=False for production