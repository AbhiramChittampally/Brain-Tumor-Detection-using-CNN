# 🧠 Brain Tumor Detection System using ResNet18 (Flask + PyTorch)

## 📂 Project Structure

```
project/
│
├── app.py                         # Flask app for prediction
├── train_enhanced.py              #Training the model

│
├── templates/
│   └── index.html                 # Frontend interface
│   └── about.html                 # Frontend interface
│   └── scan.html                 # Frontend interface
│
├── brain_tumor_model/
│   ├── resnet18_brain_tumor.pth              # trained model
│   ├── training_metrics.png         # Loss & accuracy plots
│   ├── confusion_matrix.png         # Confusion matrix
│   └── per_class_metrics.png                 # Bar chart of class metrics
```
```
To train the model download Kaggles brain tumor dataset of MRI scans and create folders in the same directory as :
brain_tumor_dataset/

```
---

## 🚀 Features

* Upload MRI scans via a web UI
* Automatically predicts tumor type: `glioma`, `meningioma`, `pituitary`, or `no_tumor`
* Backend powered by fine-tuned **ResNet18**
* Visual metrics: confusion matrix, accuracy curves, and class-wise scores
* Two training options:

  
  * `train_enhanced.py` — clean training on original dataset

---

## ⚙️ How to Run

### 1. Setup Environment

```bash
pip install torch torchvision flask scikit-learn matplotlib seaborn
```

### 2. Model Training (Optional)



```bash

python train_enhanced.py        # With extra data merging
```

### 3. Run the Flask App

Make sure the `resnet18_brain_tumor` file is present in `brain_tumor_model/`.

```bash
python app.py
```



## 🧪 Inference: `/predict` Endpoint

* **Method**: `POST`
* **Payload**: Form with image file (`file`)
* **Response**:

```json
{
  "prediction": "glioma",
  "confidence": "98.32%"
}
```

---

## 🖼️ Frontend Highlights

* Drag-and-drop MRI scan upload
* Real-time preview
* Styled using CSS and Google Fonts
* Displays tumor type and confidence
* Shows tailored info for each tumor class
* Responsive layout for mobile support

> *Note: replace the `REPLACE_WITH_YOUR_BRAIN_IMAGE_URL.jpg` in the CSS background for a cooler UI.*

---

## 📊 Evaluation Results

The model achieved **overall accuracy ranging between 97% to 99%** during validation across multiple runs.

### ✅ Confusion Matrix

<img src="brain_tumor_model/confusion_matrix_finetune.png" width="500">

### 📈 Training Metrics

<img width="1200" height="600" alt="training_metrics_finetune" src="https://github.com/user-attachments/assets/134bef0a-d1a8-4065-890d-99e4f4694e98" />


### 🔍 Per-Class Metrics

<img src="brain_tumor_model/per_class_metrics.png" width="600">

---

## 🧠 Model Architecture

* **Base**: ResNet18 pretrained on ImageNet
* **Fine-tuning**:

  * Layers `layer3`, `layer4`, and `fc` unfrozen
  * Added:

    * `Linear → ReLU → Dropout(0.3) → Linear(num_classes=4)`

---

## 🛠️ To Improve Further

* Add Grad-CAM visualization for heatmaps
* Add file size & format validations
* Include background image for enhanced UI
* Use `Flask-CORS` for cross-origin frontend hosting
* Dockerize the app for easier deployment

---

## 📬 Credits

Developed with ❤️ using:

* PyTorch
* Flask
* ResNet18
* HTML/CSS + JS
