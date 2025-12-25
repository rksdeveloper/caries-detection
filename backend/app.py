from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from collections import OrderedDict

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- X-ray Model ----------------
xray_model = models.resnet18(pretrained=False)
num_ftrs = xray_model.fc.in_features
xray_model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: caries, healthy
xray_model.load_state_dict(torch.load("model/xray_model.pth", map_location=device))
xray_model = xray_model.to(device)
xray_model.eval()

# ---------------- Clinical Model (Option 1) ----------------
clinical_model = nn.Sequential(
    nn.Linear(3, 16),   # input features = 3
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 4)    # output classes = 4 (saved model)
)

checkpoint = torch.load("model/clinical_model.pth", map_location=device)
state_dict = checkpoint.get("model", checkpoint)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("model.", "")
    new_state_dict[new_key] = v
clinical_model.load_state_dict(new_state_dict)
clinical_model = clinical_model.to(device)
clinical_model.eval()

# ---------------- Transforms ----------------
image_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

xray_classes = ["caries", "healthy"]
clinical_classes = ["class0","class1","class2","class3"]

# ---------------- X-ray Prediction ----------------
@app.route("/predict/xray", methods=["POST"])
def predict_xray():
    if "xray" not in request.files:
        return jsonify({"error":"No file uploaded"}),400
    file = request.files["xray"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = xray_model(image)
        _, predicted = torch.max(outputs,1)
        prediction = xray_classes[predicted.item()]

    return jsonify({"prediction": prediction})

# ---------------- Clinical Prediction ----------------
@app.route("/predict/clinical", methods=["POST"])
def predict_clinical():
    data = request.json
    if not data or "features" not in data:
        return jsonify({"error":"No clinical data provided"}),400

    features = torch.tensor([data["features"]],dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = clinical_model(features)
        _, predicted = torch.max(outputs,1)
        prediction = clinical_classes[predicted.item()]

    return jsonify({"prediction": prediction})

if __name__=="__main__":
    app.run(debug=True)
