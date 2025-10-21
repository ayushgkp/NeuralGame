from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import os
from werkzeug.utils import secure_filename
import zipfile
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

app = Flask(__name__)
CORS(app)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'zip', 'rar', '7z', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Single Player "Database" ---
player_data = {
    "playerLevel": 1,
    "xp": 0,
    "insightPoints": 100,
    "computeCredits": 10000,
    "currentChallenge": 1,
    "equippedModel": None,
    "equippedProject": None,
    "uploadedDatasets": [],
    "purchasedItems": [] 
}

# --- Challenge Answer Database ---
challenge_answers = {
    1: "nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3)", 2: "nn.ReLU()",
    3: "nn.MaxPool2d(kernel_size=2)", 4: "nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3)",
    5: "nn.MaxPool2d(kernel_size=2)", 6: "nn.Flatten()", 7: "nn.Linear(2048,512)",
    8: "nn.Dropout(0.5)", 9: "nn.Linear(512,10)", 10: "self.pool1(self.relu1(self.conv1(x)))"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- API Routes ---

@app.route('/get-progress', methods=['GET'])
def get_progress():
    print("Progress requested. Sending:", player_data)
    return jsonify(player_data)

@app.route('/save-progress', methods=['POST'])
def save_progress():
    global player_data
    player_data.update(request.json)
    print("Progress saved. New data:", player_data)
    return jsonify({"status": "success", "data": player_data})

@app.route('/validate-answer', methods=['POST'])
def validate_answer():
    data = request.json
    challenge_id = data.get('challengeId')
    user_input = data.get('userInput', '')
    equipped_model = player_data.get('equippedModel')

    def clean_string(s):
        return (s or "").strip().replace(" ", "")

    correct_answer = challenge_answers.get(challenge_id)
    if not correct_answer or clean_string(user_input) != clean_string(correct_answer):
        return jsonify({"correct": False})

    base_xp_reward = 100
    bonus_multiplier = {
        'efficientnet': 1.8, 'inception': 1.6,
        'resnet50': 1.5, 'vgg16': 1.2
    }.get(equipped_model, 1.0)
    
    final_xp_reward = int(base_xp_reward * bonus_multiplier)
    print(f" -> Correct! Awarding {final_xp_reward} XP.")
    return jsonify({"correct": True, "xpReward": final_xp_reward})

@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    if filename not in player_data['uploadedDatasets']:
        player_data['uploadedDatasets'].append(filename)
    
    return jsonify({
        "status": "success", 
        "message": f"File '{filename}' uploaded!",
        "datasets": player_data['uploadedDatasets']
    })

@app.route('/train-model', methods=['POST'])
def train_model():
    data = request.json
    model_name = data.get('modelName')
    dataset_name = data.get('datasetName')

    if not model_name or not dataset_name:
        return jsonify({"status": "error", "message": "Model and dataset must be provided."}), 400

    print(f"\n--- Starting Training Job ---")
    print(f"Model: {model_name}, Dataset: {dataset_name}")

    dataloaders = {}
    num_classes = 0

    if dataset_name == 'CIFAR-10':
        print(" -> Using built-in CIFAR-10 dataset.")
        try:
            transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_dataset = datasets.CIFAR10(root=DATA_FOLDER, train=True, download=True, transform=transform)
            val_dataset = datasets.CIFAR10(root=DATA_FOLDER, train=False, download=True, transform=transform)
            # --- THE FIX: Changed num_workers back to 0 to prevent freezing ---
            dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
            dataloaders['val'] = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
            num_classes = 10
            print(f" -> CIFAR-10 loaded successfully.")
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed to load CIFAR-10: {e}"}), 500
    
    else:
        print(f" -> Using uploaded dataset: {dataset_name}")
        zip_path = os.path.join(UPLOAD_FOLDER, dataset_name)
        extract_path = os.path.join(UPLOAD_FOLDER, os.path.splitext(dataset_name)[0])
        if not os.path.exists(zip_path): return jsonify({"status": "error", "message": "Dataset zip file not found."}), 404
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed to unzip file: {e}"}), 500

        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        train_dir = os.path.join(extract_path, 'train')
        val_dir = os.path.join(extract_path, 'val')
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            if os.path.exists(os.path.join(extract_path, 'seg_train', 'seg_train')):
                train_dir = os.path.join(extract_path, 'seg_train', 'seg_train')
                val_dir = os.path.join(extract_path, 'seg_test', 'seg_test')
            else:
                 return jsonify({"status":"error", "message":"Dataset must contain 'train' and 'val' folders."}), 400

        image_datasets = {'train': datasets.ImageFolder(train_dir, transform), 'val': datasets.ImageFolder(val_dir, transform)}
        # --- THE FIX: Changed num_workers back to 0 ---
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0) for x in ['train', 'val']}
        num_classes = len(image_datasets['train'].classes)

    # --- Model Selection & Training Logic ---
    model = None
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'inception':
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        return jsonify({"status": "error", "message": f"Model '{model_name}' not implemented."}), 501

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f" -> Using device: {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print(" -> Starting training for 1 epoch...")
    model.train()
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(" -> Training epoch complete.")

    print(" -> Evaluating model...")
    model.eval()
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
    
    accuracy = (running_corrects.double() / total) * 100
    
    print(f"--- Training Job Complete --- Final Accuracy: {accuracy:.2f}%")
    return jsonify({"status": "success", "accuracy": f"{accuracy:.2f}%"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

