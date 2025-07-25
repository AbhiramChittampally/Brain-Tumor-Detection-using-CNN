# 🧠 Brain Tumor Detection System using ResNet18 (Flask + PyTorch)

## 📂 Project Structure

```
project/
│
├── app.py                         # Flask app for prediction
├── train.py                       # Training with extra web data
├── train0.py                      # Base training script
│
├── templates/
│   └── index.html                 # Frontend interface
│
├── brain_tumor_model/
│   ├── resnet18_brain_tumor.pth              # Pretrained model
│   ├── resnet18_brain_tumor_finetuned.pth    # Fine-tuned model
│   ├── training_metrics_finetune.png         # Loss & accuracy plots
│   ├── confusion_matrix_finetune.png         # Confusion matrix
│   └── per_class_metrics.png                 # Bar chart of class metrics
```
```
To train the model donwload Kaggles brain tumor dataset of MRI scans and create folders in the same directory as brain_tumor_dataset and for fine tuning create the directories:
brain_tumor_extra_data/
├── glioma/
├── meningioma/
├── pituitary/
├── no tumor/
and
├── combined_dataset/
│   ├── Training/
│   └── Testing/
```
---

## 🚀 Features

* Upload MRI scans via a web UI
* Automatically predicts tumor type: `glioma`, `meningioma`, `pituitary`, or `no_tumor`
* Backend powered by fine-tuned **ResNet18**
* Visual metrics: confusion matrix, accuracy curves, and class-wise scores
* Two training options:

  * `train.py` — includes extra web data
  * `train0.py` — clean training on original dataset

---

## ⚙️ How to Run

### 1. Setup Environment

```bash
pip install torch torchvision flask scikit-learn matplotlib seaborn
```

### 2. Model Training (Optional)

You can retrain or fine-tune:

```bash
python train0.py       # For clean training
python train.py        # With extra data merging
```

### 3. Run the Flask App

Make sure the `resnet18_brain_tumor_finetuned.pth` file is present in `brain_tumor_model/`.

```bash
python app.py
```

App will be available at: `http://localhost:5000`

---

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

<img src="brain_tumor_model/training_metrics_finetune.png" width="700">

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
