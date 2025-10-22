from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import os
from werkzeug.utils import secure_filename
import zipfile
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time 
import torch.multiprocessing # Import multiprocessing

app = Flask(__name__)
CORS(app)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'zip', 'rar', '7z', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Single Player "Database" ---
player_data = {
    "playerLevel": 1, "xp": 0, "insightPoints": 100, "computeCredits": 10000,
    "currentChallenge": 1, "equippedModel": None, "equippedProject": None,
    "uploadedDatasets": [], "purchasedItems": []
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
    bonus_multiplier = {'efficientnet': 1.8, 'inception': 1.6, 'resnet50': 1.5, 'vgg16': 1.2}.get(equipped_model, 1.0)
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
        # We should ideally save progress here, but deferring for simplicity
    
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
    active_potions = data.get('activePotions', []) 

    if not model_name or not dataset_name:
        return jsonify({"status": "error", "message": "Model and dataset must be provided."}), 400

    start_time_total = time.time() 
    print(f"\n--- Starting Training Job ---")
    print(f"Model: {model_name}, Dataset: {dataset_name}")
    print(f"Active Potions: {active_potions}") 

    dataloaders = {}
    num_classes = 0

    # Define transforms
    base_transform_list = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    cifar_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    inception_base_transform_list = [transforms.Resize(299), transforms.CenterCrop(299), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    # Build training transforms dynamically
    train_transform_list = []
    current_base = base_transform_list
    current_norm = base_transform_list[-1]
    
    if model_name == 'inception':
        current_base = inception_base_transform_list
        current_norm = inception_base_transform_list[-1]
        train_transform_list.extend(current_base[:-2]) # Resize, Crop
    elif dataset_name == 'CIFAR-10':
         current_base = [transforms.Resize(224), transforms.ToTensor(), cifar_norm]
         current_norm = cifar_norm
         train_transform_list.extend(current_base[:-2]) # Resize
    else:
        train_transform_list.extend(current_base[:-2]) # Resize, Crop

    if 'random_flip' in active_potions: train_transform_list.append(transforms.RandomHorizontalFlip())
    if 'random_rotation' in active_potions: train_transform_list.append(transforms.RandomRotation(10))
    if 'color_jitter' in active_potions: train_transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

    train_transform_list.append(transforms.ToTensor())
    train_transform_list.append(current_norm)
    train_transform = transforms.Compose(train_transform_list)
    val_transform = transforms.Compose(current_base) 

    # Load Dataset
    start_time_data = time.time() 
    if dataset_name == 'CIFAR-10':
        print(" -> Using built-in CIFAR-10 dataset.")
        try:
            train_dataset = datasets.CIFAR10(root=DATA_FOLDER, train=True, download=True, transform=train_transform)
            val_dataset = datasets.CIFAR10(root=DATA_FOLDER, train=False, download=True, transform=val_transform)
            # --- THE CHANGE: Enable parallel workers ---
            dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
            dataloaders['val'] = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
            num_classes = 10
        except Exception as e: return jsonify({"status": "error", "message": f"Failed to load CIFAR-10: {e}"}), 500
    
    else: # Handle uploaded datasets
        print(f" -> Using uploaded dataset: {dataset_name}")
        zip_path = os.path.join(UPLOAD_FOLDER, dataset_name)
        extract_path = os.path.join(UPLOAD_FOLDER, os.path.splitext(dataset_name)[0])
        if not os.path.exists(zip_path): return jsonify({"status": "error", "message": "Dataset zip file not found."}), 404
        try: # Unzip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(extract_path)
        except Exception as e: return jsonify({"status": "error", "message": f"Failed to unzip file: {e}"}), 500
        train_dir, val_dir = os.path.join(extract_path, 'train'), os.path.join(extract_path, 'val')
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            if os.path.exists(os.path.join(extract_path, 'seg_train', 'seg_train')): 
                train_dir = os.path.join(extract_path, 'seg_train', 'seg_train')
                val_dir = os.path.join(extract_path, 'seg_test', 'seg_test')
            else: return jsonify({"status":"error", "message":"Dataset needs 'train'/'val' folders."}), 400
        try:
            image_datasets = {'train': datasets.ImageFolder(train_dir, train_transform), 'val': datasets.ImageFolder(val_dir, val_transform)}
            # --- THE CHANGE: Enable parallel workers ---
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'val']}
            num_classes = len(image_datasets['train'].classes)
        except Exception as e: return jsonify({"status": "error", "message": f"Error loading image folders: {e}"}), 500
    
    end_time_data = time.time() 
    print(f" -> Dataset loading took {end_time_data - start_time_data:.2f} seconds.")

    # Model Selection & Training Logic
    model = None
    try:
        print(f" -> Loading model: {model_name}")
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
            model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True) 
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.aux_logits = False 
        else: return jsonify({"status": "error", "message": f"Model '{model_name}' not implemented."}), 501
    except Exception as e: return jsonify({"status": "error", "message": f"Error loading model weights: {e}"}), 500

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f" -> Using device: {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print(" -> Starting training for 1 epoch...")
    start_time_train = time.time() 
    model.train()
    batch_num = 0
    try:
        for inputs, labels in dataloaders['train']:
            batch_num += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
            if batch_num % 50 == 0: print(f"    Batch {batch_num}/{len(dataloaders['train'])}...")
    except Exception as e: return jsonify({"status": "error", "message": f"Error during training loop: {e}"}), 500
    end_time_train = time.time() 
    print(f" -> Training epoch complete. Took {end_time_train - start_time_train:.2f} seconds.")

    print(" -> Evaluating model...")
    start_time_eval = time.time() 
    model.eval()
    running_corrects = 0
    total = 0
    try:
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
        if total == 0: return jsonify({"status": "error", "message": "Evaluation set empty!"}), 500
        accuracy = (running_corrects.double() / total) * 100
    except Exception as e: return jsonify({"status": "error", "message": f"Error during evaluation: {e}"}), 500
    
    end_time_eval = time.time() 
    end_time_total = time.time() 
    print(f" -> Evaluation complete. Took {end_time_eval - start_time_eval:.2f} seconds.")
    print(f"--- Training Job Complete --- Final Accuracy: {accuracy:.2f}%")
    print(f"--- Total time: {end_time_total - start_time_total:.2f} seconds ---")

    return jsonify({"status": "success", "accuracy": f"{accuracy:.2f}%"})

if __name__ == '__main__':
    # --- ADDED LINE: Necessary for multiprocessing stability on Windows ---
    torch.multiprocessing.freeze_support() 
    app.run(debug=True, port=5000)

