import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define classes (must match training order)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Model architecture (must match training)
class ResNetFineTuned(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetFineTuned, self).__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.model = models.resnet18(weights=weights)
        
        # Freeze layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze deeper layers
        for name, child in self.model.named_children():
            if name in ['layer3', 'layer4', 'fc']:
                for param in child.parameters():
                    param.requires_grad = True
                    
        # Modify the classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load the trained model
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetFineTuned(num_classes=4).to(device)
    model_path = 'brain_tumor_model/resnet18_brain_tumor.pth'
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        # Handle potential architecture mismatch
        model = torch.load(model_path, map_location=device)
    
    model.eval()
    return model, device

# Preprocessing transform (must match validation transform)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Prediction function
def predict_image(image_path):
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Load model
    model, device = load_model()
    img_tensor = img_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, preds = torch.max(probabilities, 1)
    
    # Convert to human-readable results
    class_name = CLASS_NAMES[preds.item()]
    confidence = conf.item() * 100
    
    # Get class-wise probabilities
    class_probs = {CLASS_NAMES[i]: f"{probabilities[0][i].item()*100:.2f}%" 
                   for i in range(len(CLASS_NAMES))}
    
    return class_name, confidence, class_probs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/scan')
def scan():
    return render_template('scan.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save the file temporarily
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Make prediction
        class_name, confidence, class_probs = predict_image(filename)
        
        # Remove temporary file
        os.remove(filename)
        
        # Create result dictionary
        result = {
            'class': class_name,
            'confidence': f"{confidence:.2f}%",
            'probabilities': class_probs
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
