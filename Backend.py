from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import os
from werkzeug.utils import secure_filename
import zipfile
import time
import random
import copy # Needed for NAS crossover
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split # Needed for NAS evaluation
from torch.utils.checkpoint import checkpoint as activation_checkpoint # Needed for Memory Crystal
from torchvision import datasets, models, transforms
import torch.multiprocessing
import base64
from PIL import Image, ImageOps # Added ImageOps
import io
# --- Added for simpler image processing ---
import numpy as np
import cv2 # OpenCV for image processing

app = Flask(__name__)
CORS(app)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'zip', 'rar', '7z', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- MODIFIED SimpleCNN to handle different inputs ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        # Use a flag to know if this is an MNIST model
        is_mnist = in_channels == 1

        # Add padding if it's MNIST (28x28) to maintain dimensions
        padding = 1 if is_mnist else 0

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=padding)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=padding)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Calculate flatten_size based on input type
        if is_mnist: # 28x28 -> pool -> 14x14 -> pool -> 7x7
            self.flatten_size = 32 * 7 * 7
            fc1_out = 128
        else: # Assumes 32x32 from challenges for original logic
              # Need to recalculate based on non-padded convs if used for CIFAR
              # For original 32x32 input:
              # Conv1 (no pad): 32 -> 30, Pool1: 30 -> 15
              # Conv2 (no pad): 15 -> 13, Pool2: 13 -> 6 (integer division)
            self.flatten_size = 32 * 6 * 6 # Recalculated for 32x32 no padding
            fc1_out = 512

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flatten_size, fc1_out)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(fc1_out, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        # Check if flatten size matches fc1 input dynamically ONLY IF NOT MNIST
        # MNIST model has fixed size, other models might vary based on SimpleCNN usage
        if x.shape[1] != self.fc1.in_features and self.conv1.in_channels != 1:
             print(f"Warning: Adjusting FC layer input size. Expected {self.fc1.in_features}, got {x.shape[1]}")
             # Adjust fc1 on the fly if needed for non-MNIST models
             # This requires ensuring fc1_out matches original design
             original_out_features = self.fc1.out_features
             self.fc1 = nn.Linear(x.shape[1], original_out_features).to(x.device)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- LOAD PRE-TRAINED MNIST MODEL AT STARTUP ---
mnist_model = SimpleCNN(in_channels=1)
try:
    # Try to load onto GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mnist_model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
    mnist_model.to(device) # Move model to the selected device
    mnist_model.eval() # Set model to evaluation mode
    print(f"--- Pre-trained MNIST model loaded successfully onto {device}. ---")
except FileNotFoundError:
    mnist_model = None
    print("--- WARNING: 'mnist_model.pth' not found. MNIST minigame will be disabled. ---")
    print("--- Please run 'train_mnist.py' once to create the model file. ---")
except Exception as e:
    mnist_model = None
    print(f"--- ERROR loading MNIST model: {e} ---")


# --- Single Player "Database" ---
player_data = {
    "playerLevel": 1, "xp": 0, "insightPoints": 100, "computeCredits": 10000,
    "currentChallenge": 1, "equippedModel": None, "equippedProject": None,
    "uploadedDatasets": [], "purchasedItems": [],
    "memoryCrystalActive": False
}

# --- Challenge Answer Database ---
challenge_answers = {
    1: "nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3)", 2: "nn.ReLU()",
    3: "nn.MaxPool2d(kernel_size=2)", 4: "nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3)",
    5: "nn.MaxPool2d(kernel_size=2)", 6: "nn.Flatten()", 7: "nn.Linear(1152,512)", # Adjusted based on recalculated 32*6*6 flatten size
    8: "nn.Dropout(0.5)", 9: "nn.Linear(512,10)", 10: "self.pool1(self.relu1(self.conv1(x)))",
    11: "transforms.RandomHorizontalFlip()",
    12: "DataLoader(train_data,batch_size=64,shuffle=True)",
    13: "nn.CrossEntropyLoss()",
    14: "optim.Adam(model.parameters(),lr=0.001)",
    15: "loss.backward()",
    16: "optimizer.step()",
    17: "models.resnet18(weights='IMAGENET1K_V1')",
    18: "optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Helper to save progress internally ---
def save_progress_internal(data):
    global player_data
    player_data.update(data)
    # print("Internal save. New data:", player_data) # Less verbose logging


# --- CORE API Routes ---

@app.route('/get-progress', methods=['GET'])
def get_progress():
    # print("Progress requested. Sending:", player_data)
    return jsonify(player_data)

@app.route('/save-progress', methods=['POST'])
def save_progress():
    global player_data
    data_received = request.json
    # Filter out any keys that are not expected in player_data to prevent adding extra state
    valid_keys = player_data.keys()
    filtered_data = {k: v for k, v in data_received.items() if k in valid_keys}
    player_data.update(filtered_data)
    print("Progress saved. Relevant data:", filtered_data)
    return jsonify({"status": "success", "data": player_data})


@app.route('/validate-answer', methods=['POST'])
def validate_answer():
    data = request.json
    challenge_id_str = data.get('challengeId')
    user_input = data.get('userInput', '')
    equipped_model = player_data.get('equippedModel') # Note: this uses player's equipped model

    try:
        challenge_id = int(challenge_id_str)
    except (ValueError, TypeError):
        return jsonify({"correct": False, "message": "Invalid challengeId format"}), 400

    def clean_string(s):
        # Remove parentheses, spaces, and convert to lowercase for robust comparison
        return (s or "").strip().replace(" ", "").replace("(", "").replace(")", "").lower()

    correct_answer = challenge_answers.get(challenge_id)

    # Specific check for challenge 7's answer based on updated SimpleCNN flatten size
    if challenge_id == 7:
        correct_answer = "nn.Linear(1152,512)" # Update correct answer based on 32*6*6

    cleaned_user_input = clean_string(user_input)
    cleaned_correct_answer = clean_string(correct_answer)

    # Debugging print statements
    # print(f"Challenge ID: {challenge_id}")
    # print(f"User Input Raw: '{user_input}'")
    # print(f"User Input Cleaned: '{cleaned_user_input}'")
    # print(f"Correct Answer Raw: '{correct_answer}'")
    # print(f"Correct Answer Cleaned: '{cleaned_correct_answer}'")

    if not correct_answer or cleaned_user_input != cleaned_correct_answer:
        print(f" -> Incorrect. User:'{cleaned_user_input}', Expected:'{cleaned_correct_answer}'")
        return jsonify({"correct": False})

    base_xp_reward = 100
    bonus_multiplier = {'efficientnet': 1.8, 'inception': 1.6, 'resnet50': 1.5, 'vgg16': 1.2}.get(equipped_model, 1.0)
    final_xp_reward = int(base_xp_reward * bonus_multiplier)

    print(f" -> Correct! Awarding {final_xp_reward} XP.")
    return jsonify({"correct": True, "xpReward": final_xp_reward})

# --- UPLOAD ROUTE ---
@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    try:
        file.save(save_path)
        print(f" -> File saved to {save_path}")
        message = f"File '{filename}' uploaded!"
        saved_filename_for_list = filename

        if filename.lower().endswith('.zip'):
            extract_folder_name = os.path.splitext(filename)[0]
            target_dir = os.path.join(app.config['UPLOAD_FOLDER'], extract_folder_name)
            os.makedirs(target_dir, exist_ok=True)
            print(f" -> Extracting zip file to {target_dir}...")
            try:
                with zipfile.ZipFile(save_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                message = f"File '{filename}' uploaded and extracted!"
            except Exception as e_zip:
                 print(f" -> Error during zip extraction: {e_zip}")
                 return jsonify({"status": "error", "message": f"Error extracting zip: {e_zip}"}), 500
        else:
             print(f" -> File '{filename}' is not a zip, saved directly.")

        if saved_filename_for_list not in player_data['uploadedDatasets']:
            player_data['uploadedDatasets'].append(saved_filename_for_list)
            save_progress_internal(player_data)

        return jsonify({"status": "success", "message": message, "datasets": player_data['uploadedDatasets']})
    except Exception as e:
        print(f" -> Error during file upload/saving: {e}")
        return jsonify({"status": "error", "message": f"Server error processing file: {e}"}), 500

# --- Transforms Helper ---
def build_transforms(potions, is_training=True, model_name=None, dataset_name=None):
    potions = potions or []
    is_mnist = dataset_name == 'MNIST'

    if is_mnist:
        img_size = 28
    elif model_name == 'inception':
        img_size = 299
    else:
        img_size = 224 # Default for ImageNet models

    transform_list = []

    if is_mnist:
        # MNIST transform is simple
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,))) # MNIST specific mean/std
    else: # ImageNet style transforms
        if is_training:
            # Add augmentations if training
            if 'random_flip' in potions: transform_list.append(transforms.RandomHorizontalFlip())
            if 'random_rotation' in potions: transform_list.append(transforms.RandomRotation(10))
            if 'color_jitter' in potions: transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
            # Standard resize/crop for training
            transform_list.append(transforms.RandomResizedCrop(img_size))
        else:
            # Standard resize/crop for validation
             transform_list.extend([transforms.Resize(256), transforms.CenterCrop(img_size)])


        transform_list.append(transforms.ToTensor())

        # Use standard ImageNet mean/std unless it's SimpleCNN on CIFAR-10
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if dataset_name == 'CIFAR-10' and model_name == 'SimpleCNN':
             # CIFAR-10 specific mean/std, often used with simple models
             normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform_list.append(normalize)

    return transforms.Compose(transform_list)

# --- Dataset Loading Helper ---
def load_data(dataset_name, active_potions, model_name):
    print(f" -> Attempting to load dataset: {dataset_name}")
    dataloaders = {}
    num_classes = 0

    train_transform = build_transforms(active_potions, is_training=True, model_name=model_name, dataset_name=dataset_name)
    val_transform = build_transforms(None, is_training=False, model_name=model_name, dataset_name=dataset_name)

    # --- Standard Datasets ---
    if dataset_name == 'CIFAR-10':
        print(" -> Using built-in CIFAR-10 dataset.")
        try:
            train_dataset = datasets.CIFAR10(root=DATA_FOLDER, train=True, download=True, transform=train_transform)
            val_dataset = datasets.CIFAR10(root=DATA_FOLDER, train=False, download=True, transform=val_transform)
            dataloaders['train'] = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
            dataloaders['val'] = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
            num_classes = 10
        except Exception as e: raise ValueError(f"Failed to load CIFAR-10: {e}")

    elif dataset_name == 'MNIST':
        print(" -> Using built-in MNIST dataset.")
        # MNIST requires grayscale input (1 channel)
        if model_name not in [None, 'SimpleCNN']:
             raise ValueError(f"Model '{model_name}' expects 3 color channels, but MNIST is grayscale. Use SimpleCNN.")
        try:
            train_dataset = datasets.MNIST(root=DATA_FOLDER, train=True, download=True, transform=train_transform)
            val_dataset = datasets.MNIST(root=DATA_FOLDER, train=False, download=True, transform=val_transform)
            dataloaders['train'] = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
            dataloaders['val'] = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
            num_classes = 10 # 10 digits (0-9)
        except Exception as e: raise ValueError(f"Failed to load MNIST: {e}")

    # --- Uploaded Dataset Handling ---
    else:
        print(f" -> Using uploaded dataset: {dataset_name}")
        # Determine the base path (could be zip extraction folder or just the filename if it's an image folder itself)
        extract_folder_name = os.path.splitext(dataset_name)[0]
        base_path = os.path.join(UPLOAD_FOLDER, extract_folder_name)
        if not os.path.isdir(base_path):
             base_path_alt = os.path.join(UPLOAD_FOLDER, dataset_name) # Maybe it wasn't zipped
             if os.path.isdir(base_path_alt):
                 base_path = base_path_alt
             else:
                 raise FileNotFoundError(f"Dataset folder not found at '{base_path}' or '{base_path_alt}'.")

        # Standard structure: base_path/train and base_path/val
        train_dir = os.path.join(base_path, 'train')
        val_dir = os.path.join(base_path, 'val')

        # Fallback structures if standard isn't found
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            # Check for common nested structures (e.g., dataset_name/dataset_name/train)
            nested_train = os.path.join(base_path, extract_folder_name, 'train')
            nested_val = os.path.join(base_path, extract_folder_name, 'val')
            if os.path.exists(nested_train) and os.path.exists(nested_val):
                train_dir, val_dir = nested_train, nested_val
            else:
                # Check Intel Image Classification structure
                intel_train = os.path.join(base_path, 'seg_train', 'seg_train')
                intel_val = os.path.join(base_path, 'seg_test', 'seg_test')
                if os.path.exists(intel_train) and os.path.exists(intel_val):
                    train_dir, val_dir = intel_train, intel_val
                # Add more fallback checks here if needed for other common structures

        # Final check if train/val directories were found
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
             raise FileNotFoundError(f"Dataset structure invalid in '{base_path}'. Need 'train' and 'val' folders (checked standard and common fallbacks).")

        print(f"   -> Found train data: {train_dir}")
        print(f"   -> Found validation data: {val_dir}")

        # Load using ImageFolder
        try:
            image_datasets = {
                'train': datasets.ImageFolder(train_dir, train_transform),
                'val': datasets.ImageFolder(val_dir, val_transform)
            }
            # Create DataLoaders
            dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=(x=='train'), num_workers=0) for x in ['train', 'val']}
            num_classes = len(image_datasets['train'].classes)
            if num_classes == 0:
                 raise ValueError(f"No classes found in '{train_dir}'. Check dataset structure.")
        except Exception as e:
            raise ValueError(f"Error loading images from folders ('{train_dir}', '{val_dir}'): {e}")

    print(f" -> Dataset loaded. Found {num_classes} classes.")
    return dataloaders, num_classes

# --- Model Loading Helper ---
def get_model(model_name, num_classes, dataset_name=None):
    print(f" -> Loading model: {model_name}")
    model = None
    in_channels = 1 if dataset_name == 'MNIST' else 3 # Determine input channels

    if model_name == 'resnet50':
        if in_channels != 3: raise ValueError("ResNet50 requires 3 input channels.")
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes) # Replace the final layer
    elif model_name == 'vgg16':
        if in_channels != 3: raise ValueError("VGG16 requires 3 input channels.")
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes) # Replace the final layer
    elif model_name == 'efficientnet':
        if in_channels != 3: raise ValueError("EfficientNet requires 3 input channels.")
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes) # Replace the final layer
    elif model_name == 'inception':
        if in_channels != 3: raise ValueError("InceptionV3 requires 3 input channels.")
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
        # Handle main classifier
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        # Handle auxiliary classifier if present (needed during training sometimes)
        if model.AuxLogits is not None:
             num_ftrs_aux = model.AuxLogits.fc.in_features
             model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)
        # IMPORTANT: Set aux_logits to False for evaluation/inference phase usually
        model.aux_logits = False
    elif model_name == 'SimpleCNN' or model_name is None:
        print(f" -> Model '{model_name or 'SimpleCNN'}' selected. Using SimpleCNN with {in_channels} input channels.")
        model = SimpleCNN(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model name '{model_name}' requested.")

    return model

# --- TRAINING/EVALUATION FUNCTIONS ---
def train_one_epoch(model, loader, device, optimizer, criterion, use_checkpoint):
    model.train() # Set model to training mode
    running_loss = 0.0; total = 0; correct = 0; batch_num = 0
    start_epoch_time = time.time()
    for inputs, labels in loader:
        batch_num += 1
        batch_start_time = time.time()
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True) # Use non_blocking for potential speedup

        optimizer.zero_grad(set_to_none=True) # More efficient zeroing

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()): # Automatic Mixed Precision if CUDA available
            if use_checkpoint and not isinstance(model, models.Inception3): # Inception aux requires special handling with checkpoint
                 # Use reentrant=False if possible (check PyTorch version compatibility)
                 outputs = activation_checkpoint(lambda inp: model(inp), inputs, use_reentrant=False, preserve_rng_state=False) # Checkpoint needs function
            else:
                 outputs = model(inputs)
                 # Handle Inception V3's potential tuple output during training if aux_logits=True
                 if isinstance(outputs, models.inception.InceptionOutputs):
                     outputs = outputs.logits # Use the main output

            loss = criterion(outputs, labels)

        # loss.backward() # Standard backward pass
        # optimizer.step()

        # Mixed Precision Scaler (if using AMP)
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        batch_time = time.time() - batch_start_time
        # Log less frequently for large datasets
        log_freq = max(1, len(loader) // 10) # Log 10 times per epoch
        if batch_num % log_freq == 0:
            print(f"    Batch {batch_num}/{len(loader)} - Loss: {loss.item():.4f}, Time: {batch_time:.3f}s")
        # --- DEBUG: Break early for faster testing ---
        # if batch_num >= 50:
        #     print("--- DEBUG: Breaking epoch early ---")
        #     break
        # --- END DEBUG ---


    epoch_loss = running_loss / max(1, total)
    epoch_acc = 100.0 * correct / max(1, total)
    epoch_duration = time.time() - start_epoch_time
    print(f" -> Train Epoch Complete: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%. Duration: {epoch_duration:.2f}s")
    return epoch_loss, epoch_acc

@torch.no_grad() # Disable gradient calculations for evaluation
def evaluate(model, loader, device):
    model.eval() # Set model to evaluation mode
    running_corrects = 0; total = 0; batch_num = 0
    start_eval_time = time.time()
    for inputs, labels in loader:
        batch_num += 1
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()): # Use AMP for evaluation too
            outputs = model(inputs)
            # Handle Inception V3 tuple output during eval
            if isinstance(outputs, models.inception.InceptionOutputs):
                 outputs = outputs.logits # Use the main output
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)
        total += labels.size(0)
        # --- DEBUG: Break early for faster testing ---
        # if batch_num >= 25:
        #     print("--- DEBUG: Breaking evaluation early ---")
        #     break
        # --- END DEBUG ---

    eval_duration = time.time() - start_eval_time
    print(f" -> Evaluation complete. Duration: {eval_duration:.2f}s")
    if total == 0: return 0.0
    accuracy = (running_corrects.double() / total) * 100
    return accuracy

# --- MAIN TRAINING ROUTE ---
@app.route('/train-model', methods=['POST'])
def train_model():
    data = request.json
    model_name = data.get('modelName')
    dataset_name = data.get('datasetName')
    active_potions = data.get('activePotions', [])
    # Check if memory crystal is in purchased items to enable checkpointing
    use_checkpoint = 'memory_crystal' in player_data.get('purchasedItems', [])

    # Default to SimpleCNN if nothing equipped
    if model_name is None:
        model_name = 'SimpleCNN'

    # Check purchase if not SimpleCNN
    if model_name != 'SimpleCNN' and model_name not in player_data.get('purchasedItems', []):
        return jsonify({"status": "error", "message": f"Model '{model_name}' is not purchased."}), 403

    if not dataset_name:
        return jsonify({"status": "error", "message": "Dataset must be provided."}), 400

    start_time_total = time.time()
    print(f"\n--- Starting Training Job ---")
    print(f"Model: {model_name}, Dataset: {dataset_name}, Potions: {active_potions}, Checkpoint: {use_checkpoint}")

    try:
        dataloaders, num_classes = load_data(dataset_name, active_potions, model_name)
        model = get_model(model_name, num_classes, dataset_name=dataset_name)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f" -> Using device: {device}")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        # Use Adam optimizer - generally good default
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # Consider adding a learning rate scheduler for longer training runs
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # Example

        # --- Training Loop (Simplified to 1 epoch for the game) ---
        num_epochs = 1
        print(f" -> Starting training for {num_epochs} epoch...")
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], device, optimizer, criterion, use_checkpoint)
        # In a real scenario, you'd loop for num_epochs and maybe use scheduler.step()

        # --- Evaluation ---
        print(" -> Evaluating model...")
        accuracy = evaluate(model, dataloaders['val'], device)
        print(f" -> Evaluation accuracy: {accuracy:.2f}%")

    except (FileNotFoundError, ValueError, RuntimeError, Exception) as e:
        import traceback
        print(f" !!! Training/Evaluation Error: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500
    finally:
        # Clean up GPU memory if possible
        if 'model' in locals() and hasattr(model, 'cpu'): model.cpu()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


    total_time = time.time() - start_time_total
    print(f"--- Training Job Complete --- Final Accuracy: {accuracy:.2f}%")
    print(f"--- Total time: {total_time:.2f} seconds ---")
    return jsonify({"status": "success", "accuracy": f"{accuracy:.2f}%", "time_taken": total_time})


# --- NAS Code ---
# (Keep the NAS classes: ArchitectureGene, DynamicCNN)
class ArchitectureGene:
    """Represents a single architecture configuration"""
    def __init__(self):
        self.conv_layers = [] # List of dicts: {'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding'}
        self.pool_layers = [] # List of dicts: {'kernel_size', 'stride'}
        self.fc_layers = []   # List of dicts: {'in_features', 'out_features'}
        self.dropout_rate = 0.5
        self.activation = 'relu' # Could potentially mutate this later

    def add_conv_layer(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        self.conv_layers.append({
            'in_channels': in_channels, 'out_channels': out_channels,
            'kernel_size': kernel_size, 'stride': stride, 'padding': padding
        })
    def add_pool_layer(self, kernel_size, stride=None):
        if stride is None: stride = kernel_size
        self.pool_layers.append({'kernel_size': kernel_size, 'stride': stride})
    def add_fc_layer(self, in_features, out_features):
        # Ensure features are positive integers
        in_features = max(1, int(in_features))
        out_features = max(1, int(out_features))
        self.fc_layers.append({'in_features': in_features, 'out_features': out_features})

    def mutate(self):
        """Apply random mutations to the architecture."""
        mutations = [
            self._mutate_add_conv, self._mutate_remove_conv, self._mutate_change_channels,
            self._mutate_add_pool, self._mutate_remove_pool, self._mutate_change_kernel,
            self._mutate_add_fc, self._mutate_remove_fc, self._mutate_change_fc_units,
            self._mutate_change_dropout
        ]
        num_mutations = random.randint(1, 3) # Allow more mutations
        for _ in range(num_mutations):
            try:
                mutation_func = random.choice(mutations)
                mutation_func()
            except Exception as e:
                print(f"  Warning: Mutation failed ({mutation_func.__name__}): {e}") # Log mutation errors

        # Ensure channel counts are consistent after mutations
        self._fix_channel_consistency()
        # Ensure FC layer connections are consistent
        self._fix_fc_consistency()


    def _mutate_add_conv(self):
        if len(self.conv_layers) < 6: # Limit depth
            idx = random.randint(0, len(self.conv_layers))
            prev_channels = self.conv_layers[idx-1]['out_channels'] if idx > 0 else 3
            # Stricter channel limits for game constraints
            new_channels = min(prev_channels * random.choice([1.5, 2]), 64); new_channels = max(int(new_channels), 8)
            new_layer = {
                'in_channels': prev_channels, 'out_channels': new_channels,
                'kernel_size': 3, 'stride': 1, 'padding': 1
            }
            self.conv_layers.insert(idx, new_layer)

    def _mutate_remove_conv(self):
        if len(self.conv_layers) > 1: # Keep at least one conv layer
            self.conv_layers.pop(random.randrange(len(self.conv_layers)))

    def _mutate_change_channels(self):
        if self.conv_layers:
            layer = random.choice(self.conv_layers)
            # Stricter channel limits
            layer['out_channels'] = min(layer['out_channels'] * random.choice([0.5, 1.5, 2]), 64); layer['out_channels'] = max(int(layer['out_channels']), 8)

    def _mutate_add_pool(self):
        # Add pool maybe after a conv layer, limit total pools
        if len(self.pool_layers) < 3 and len(self.conv_layers) > len(self.pool_layers):
             idx = random.randint(0, len(self.pool_layers))
             self.pool_layers.insert(idx, {'kernel_size': 2, 'stride': 2})

    def _mutate_remove_pool(self):
        if self.pool_layers:
            self.pool_layers.pop(random.randrange(len(self.pool_layers)))

    def _mutate_change_kernel(self):
         # Change kernel size (e.g., conv or pool) - less common
         if random.random() < 0.1: # Low probability
            if self.conv_layers:
                layer = random.choice(self.conv_layers)
                layer['kernel_size'] = random.choice([1, 3, 5])
                # Adjust padding if kernel changes? (Keep simple: assume padding=1 for k=3, 2 for k=5, 0 for k=1)
                if layer['kernel_size'] == 1: layer['padding'] = 0
                elif layer['kernel_size'] == 3: layer['padding'] = 1
                elif layer['kernel_size'] == 5: layer['padding'] = 2


    def _mutate_add_fc(self):
         if len(self.fc_layers) < 3: # Limit FC layers
             idx = random.randint(0, len(self.fc_layers))
             prev_units = self.fc_layers[idx-1]['out_features'] if idx > 0 else 512 # Assume some input if first
             # Stricter unit limits
             new_units = max(min(int(prev_units * random.choice([0.5, 0.75])), 256), 32)
             new_layer = {'in_features': prev_units, 'out_features': new_units}
             self.fc_layers.insert(idx, new_layer)

    def _mutate_remove_fc(self):
         if len(self.fc_layers) > 1: # Keep at least one FC layer
             self.fc_layers.pop(random.randrange(len(self.fc_layers)))

    def _mutate_change_fc_units(self):
         if self.fc_layers:
             layer = random.choice(self.fc_layers)
             # Stricter unit limits
             layer['out_features'] = max(min(int(layer['out_features'] * random.choice([0.5, 1.5])), 256), 32)


    def _mutate_change_dropout(self):
        self.dropout_rate = round(random.uniform(0.1, 0.7), 2) # Wider range

    def _fix_channel_consistency(self):
        """Ensures in_channels matches previous out_channels."""
        current_channels = 3
        for layer in self.conv_layers:
            layer['in_channels'] = current_channels
            current_channels = layer['out_channels']

    def _fix_fc_consistency(self):
        """Ensures FC in_features matches previous out_features."""
        # Note: Input to the first FC layer is dynamic (flatten_size), handled in DynamicCNN init
        if len(self.fc_layers) > 1:
            for i in range(1, len(self.fc_layers)):
                 self.fc_layers[i]['in_features'] = self.fc_layers[i-1]['out_features']


class DynamicCNN(nn.Module):
    """Dynamically generated CNN based on ArchitectureGene"""
    # Assuming input_size will be passed correctly (224 for NAS by default)
    def __init__(self, gene: ArchitectureGene, num_classes: int = 10, input_size=224):
        super().__init__()
        self.gene = gene
        self.features = nn.Sequential() # Use Sequential for features
        self.classifier = nn.Sequential() # Use Sequential for classifier

        current_channels = 3 # NAS still assumes 3 color channels for complexity
        current_size = input_size
        last_conv_idx = -1

        # Build feature extractor (Conv + ReLU + Pool blocks)
        pool_to_add = list(gene.pool_layers) # Copy pool layer configs

        for i, conv_config in enumerate(gene.conv_layers):
            # Ensure consistency just before creating layer
            conv_config['in_channels'] = current_channels
            # Validate kernel/padding
            kernel = conv_config.get('kernel_size', 3)
            padding = conv_config.get('padding', 1)
            stride = conv_config.get('stride', 1)

            conv = nn.Conv2d(conv_config['in_channels'], conv_config['out_channels'],
                             kernel, stride=stride, padding=padding)
            self.features.add_module(f"conv_{i}", conv)
            self.features.add_module(f"relu_{i}", nn.ReLU(inplace=True))
            current_channels = conv_config['out_channels']
            last_conv_idx = i

            # Calculate output size after conv
            current_size = (current_size - kernel + 2 * padding) // stride + 1

            # Add pooling layer if available and sensible (e.g., after conv blocks)
            # Try to distribute pools somewhat evenly
            should_pool = len(pool_to_add) > 0 and (i == len(gene.conv_layers) - 1 or (i+1) % max(1, len(gene.conv_layers)//(len(pool_to_add)+1)) == 0)
            if should_pool:
                 pool_config = pool_to_add.pop(0)
                 pool_kernel = pool_config.get('kernel_size', 2)
                 pool_stride = pool_config.get('stride', pool_kernel)
                 pool = nn.MaxPool2d(pool_kernel, stride=pool_stride)
                 self.features.add_module(f"pool_{i}", pool)
                 # Calculate output size after pool
                 current_size = current_size // pool_kernel # Assume kernel=stride if stride not specified


        current_size = max(1, current_size) # Ensure size is at least 1x1
        self.flatten_size = current_channels * current_size * current_size

        if self.flatten_size <= 0:
            print(f"Warning: Invalid NAS architecture resulted in flatten_size {self.flatten_size}. Defaulting to safe values.")
            # Fallback: Create a minimal valid structure
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2)
            )
            final_size = input_size // 4
            self.flatten_size = 32 * final_size * final_size if final_size > 0 else 32 # Ensure positive

        # Build classifier (Flatten + FC layers)
        self.classifier.add_module("flatten", nn.Flatten())
        current_features = self.flatten_size

        # Handle empty or inconsistent FC layers defined in gene
        if not gene.fc_layers or gene.fc_layers[0]['in_features'] == 0:
             # Add a default FC layer if none/invalid generated (use stricter limit)
             fc1_out = max(min(current_features // 8, 256), 32) # Calculate based on actual flatten size
             gene.fc_layers = [{'in_features': current_features, 'out_features': fc1_out}] # Overwrite gene's fc

        # Ensure first FC layer's input matches actual flatten_size
        gene.fc_layers[0]['in_features'] = current_features

        # Fix consistency for subsequent layers (needed again after potential default overwrite)
        if len(gene.fc_layers) > 1:
             for i in range(1, len(gene.fc_layers)):
                 gene.fc_layers[i]['in_features'] = gene.fc_layers[i-1]['out_features']

        # Add generated FC layers
        for i, fc_config in enumerate(gene.fc_layers):
            # Ensure valid features (should be fixed, but double-check)
            in_f = max(1, fc_config['in_features'])
            out_f = max(1, fc_config['out_features'])

            fc = nn.Linear(in_f, out_f)
            self.classifier.add_module(f"fc_{i}", fc)
            current_features = out_f # Update for next potential layer

            # Add ReLU and Dropout between FC layers (but not after the last one)
            if i < len(gene.fc_layers) - 1:
                self.classifier.add_module(f"fc_relu_{i}", nn.ReLU(inplace=True))
                self.classifier.add_module(f"fc_dropout_{i}", nn.Dropout(gene.dropout_rate))

        # Add final output layer ONLY if the last defined FC layer doesn't match num_classes
        if current_features != num_classes:
            final_in = max(1, current_features) # Ensure positive input
            self.classifier.add_module("output_fc", nn.Linear(final_in, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def create_random_architecture() -> ArchitectureGene:
    """Create a random initial architecture with stricter limits for low VRAM"""
    gene = ArchitectureGene()
    num_conv = random.randint(2, 4); current_channels = 3 # Slightly more conv layers allowed
    for i in range(num_conv):
        # Stricter channel limits
        out_channels = min(current_channels * random.choice([1.5, 2]), 64); out_channels = max(int(out_channels), 8)
        # Assume padding=1 for conv layers in NAS for simplicity
        gene.add_conv_layer(current_channels, out_channels, kernel_size=3, padding=1)
        current_channels = out_channels
        # Increased chance of pooling, but max 3 pools
        if random.random() < 0.6 and len(gene.pool_layers) < 3 :
             gene.add_pool_layer(2)

    # Note: Flatten size is calculated dynamically in DynamicCNN, no need to estimate here

    # Add at least one FC layer definition (input size is placeholder, fixed later)
    # Stricter unit limits
    fc1_out = max(min(int(current_channels * 8 * 8), 256), 32) # Rough estimate, use smaller max units
    gene.add_fc_layer(512, fc1_out) # Placeholder input size 512

    gene.dropout_rate = round(random.uniform(0.2, 0.6), 2)
    return gene

def crossover(parent1: ArchitectureGene, parent2: ArchitectureGene) -> ArchitectureGene:
    """Performs crossover between two parent architectures."""
    child = ArchitectureGene()
    child.dropout_rate = random.choice([parent1.dropout_rate, parent2.dropout_rate])

    # Crossover Conv layers (single point crossover)
    len1, len2 = len(parent1.conv_layers), len(parent2.conv_layers)
    if len1 > 0 and len2 > 0:
        split1 = random.randint(0, len1)
        split2 = random.randint(0, len2)
        if random.random() < 0.5:
            child.conv_layers = copy.deepcopy(parent1.conv_layers[:split1] + parent2.conv_layers[split2:])
        else:
            child.conv_layers = copy.deepcopy(parent2.conv_layers[:split2] + parent1.conv_layers[split1:])
    elif len1 > 0:
        child.conv_layers = copy.deepcopy(parent1.conv_layers)
    else:
        child.conv_layers = copy.deepcopy(parent2.conv_layers)

    # Ensure at least one conv layer exists
    if not child.conv_layers:
        child.conv_layers = copy.deepcopy(random.choice([parent1.conv_layers, parent2.conv_layers]))
        if not child.conv_layers: # If both parents were somehow empty
             child.add_conv_layer(3, 16, 3) # Add a default

    # Crossover Pool layers (take from one parent, limit by conv layers)
    chosen_pool_parent = random.choice([parent1.pool_layers, parent2.pool_layers])
    child.pool_layers = copy.deepcopy(chosen_pool_parent)
    # Limit pool layers (e.g., max 3)
    child.pool_layers = child.pool_layers[:min(len(child.pool_layers), 3)]

    # Crossover FC layers (single point)
    len1_fc, len2_fc = len(parent1.fc_layers), len(parent2.fc_layers)
    if len1_fc > 0 and len2_fc > 0:
        split1_fc = random.randint(0, len1_fc)
        split2_fc = random.randint(0, len2_fc)
        if random.random() < 0.5:
             child.fc_layers = copy.deepcopy(parent1.fc_layers[:split1_fc] + parent2.fc_layers[split2_fc:])
        else:
             child.fc_layers = copy.deepcopy(parent2.fc_layers[:split2_fc] + parent1.fc_layers[split1_fc:])
    elif len1_fc > 0:
        child.fc_layers = copy.deepcopy(parent1.fc_layers)
    else:
        child.fc_layers = copy.deepcopy(parent2.fc_layers)

     # Ensure at least one fc layer exists
    if not child.fc_layers:
        child.fc_layers = copy.deepcopy(random.choice([parent1.fc_layers, parent2.fc_layers]))
        if not child.fc_layers: # If both parents were somehow empty
             child.add_fc_layer(512, 64) # Add a default


    # Fix consistency after crossover
    child._fix_channel_consistency()
    child._fix_fc_consistency() # Input to first layer is handled in DynamicCNN init

    return child


def evaluate_architecture_real(gene: ArchitectureGene, train_loader, val_loader, device, num_classes, epochs=1):
    """Evaluates a single architecture by training briefly."""
    start_eval_time = time.time()
    model = None # Initialize model variable
    try:
        # Pass input_size=224 explicitly as NAS assumes ImageNet-like input size
        model = DynamicCNN(gene, num_classes=num_classes, input_size=224).to(device)

        # --- Model Size/Complexity Check ---
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024) # Approx MB (float32)
        # Stricter limits for game constraints
        max_params = 10_000_000 # Limit to 10M parameters
        max_size_mb = 100 # Limit to 100MB
        if total_params > max_params or model_size_mb > max_size_mb:
             raise ValueError(f"Model too complex: {total_params:,} params / {model_size_mb:.2f}MB. Limits: {max_params:,} / {max_size_mb}MB.")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001) # Simple optimizer for eval
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        # --- Brief Training ---
        model.train()
        train_batches_limit = 20 # Train on fewer batches for speed
        batch_count = 0
        for data, target in train_loader:
            if batch_count >= train_batches_limit: break
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_count += 1

        # --- Brief Evaluation ---
        model.eval()
        correct, total = 0, 0
        inference_times = []
        eval_batches_limit = 10 # Evaluate on fewer batches
        batch_count = 0
        with torch.no_grad():
            for data, target in val_loader:
                if batch_count >= eval_batches_limit: break
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    eval_start = time.time()
                    output = model(data)
                    inference_times.append(time.time() - eval_start)
                    _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                batch_count += 1

        accuracy = 100. * correct / total if total > 0 else 0
        # Use median inference time for robustness against outliers
        avg_inference_time = np.median(inference_times) * 1000 if inference_times else 999
        eval_duration = time.time() - start_eval_time
        print(f"    Evaluated arch: Acc={accuracy:.2f}%, Time={avg_inference_time:.1f}ms, Size={model_size_mb:.2f}MB ({eval_duration:.1f}s)")

        return {
            'accuracy': accuracy, 'inference_time': avg_inference_time,
            'model_size': model_size_mb, 'total_params': total_params, 'success': True
        }

    except Exception as e:
        eval_duration = time.time() - start_eval_time
        # Log specific error type
        error_type = type(e).__name__
        print(f"    Arch evaluation failed ({error_type}): {e} ({eval_duration:.1f}s)")
        return {'accuracy': 0.0, 'inference_time': 9999.0, 'model_size': 999.0, 'total_params': 0, 'success': False}
    finally:
        # Ensure model is removed from GPU memory
        if model is not None: del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()


# --- NAS ROUTE ---
@app.route('/nas', methods=['POST'])
def neural_architecture_search():
    data = request.json or {}
    preferences = data.get('preferences', {})
    potions = data.get('potions', []) # Potions for data augmentation during eval

    print("--- Starting Neural Architecture Search ---")
    print(f"Preferences: {preferences}")
    print(f"Using Potions: {potions}")

    start_nas_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Prepare Dataset (using CIFAR-10 subset for NAS eval) ---
    try:
        # NAS typically uses ImageNet size transforms (224x224) even on CIFAR-10
        nas_train_transform = build_transforms(potions, is_training=True, model_name=None, dataset_name='CIFAR-10')
        nas_val_transform = build_transforms(None, is_training=False, model_name=None, dataset_name='CIFAR-10')

        full_train_dataset = datasets.CIFAR10(root=DATA_FOLDER, train=True, download=True, transform=nas_train_transform)
        full_val_dataset = datasets.CIFAR10(root=DATA_FOLDER, train=False, download=True, transform=nas_val_transform)

        # Use smaller subsets for faster NAS evaluation
        train_subset_size = 5000
        val_subset_size = 1000
        # Ensure sizes don't exceed dataset length
        train_subset_size = min(train_subset_size, len(full_train_dataset))
        val_subset_size = min(val_subset_size, len(full_val_dataset))

        # Use generators for reproducibility if needed, otherwise random split is fine
        train_subset, _ = random_split(full_train_dataset, [train_subset_size, len(full_train_dataset) - train_subset_size])
        val_subset, _ = random_split(full_val_dataset, [val_subset_size, len(full_val_dataset) - val_subset_size])

        # Use smaller batch size for NAS evaluation to fit potentially larger models
        nas_batch_size = 16
        train_loader = DataLoader(train_subset, batch_size=nas_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=nas_batch_size, shuffle=False, num_workers=0, pin_memory=True)
        num_classes = 10 # CIFAR-10 has 10 classes
    except Exception as e:
         print(f" !!! NAS Error: Failed to load dataset: {e}")
         return jsonify({"status": "error", "message": f"Failed to load dataset for NAS: {e}"}), 500

    # --- Evolutionary Algorithm Parameters ---
    population_size = 10 # Keep small for speed
    generations = 5   # Keep small for speed
    elite_size = 2      # Keep top individuals
    mutation_rate = 0.8 # Higher mutation rate for exploration

    # --- Initialize Population ---
    population = [create_random_architecture() for _ in range(population_size)]
    all_results = [] # Store results from all generations

    # --- Evolution Loop ---
    for generation in range(generations):
        gen_start_time = time.time()
        print(f"--- NAS Generation {generation + 1}/{generations} ---")
        generation_results = [] # Results for this generation
        valid_evaluated_in_gen = 0

        # --- Evaluate Population ---
        for i, gene in enumerate(population):
            print(f"  Evaluating architecture {i+1}/{len(population)}...")
            evaluation = evaluate_architecture_real(gene, train_loader, val_loader, device, num_classes, epochs=1)

            if evaluation['success']:
                valid_evaluated_in_gen += 1
                # --- Calculate Fitness ---
                acc_pref = preferences.get('accuracy', 50) / 100.0
                speed_pref = preferences.get('speed', 30) / 100.0 # Lower inference time is better
                eff_pref = preferences.get('efficiency', 20) / 100.0 # Lower size/params is better

                norm_acc = evaluation['accuracy'] / 100.0
                # Normalize speed (e.g., against a baseline like 200ms) - lower is better
                norm_speed = max(0, 1 - (evaluation['inference_time'] / 200.0))
                # Normalize efficiency (e.g., against 50MB) - lower is better
                norm_eff = max(0, 1 - (evaluation['model_size'] / 50.0))

                # Weighted sum fitness
                fitness = (norm_acc * acc_pref + norm_speed * speed_pref + norm_eff * eff_pref) * 100

                evaluation['fitness'] = round(fitness, 2)
                evaluation['generation'] = generation + 1
                evaluation['architecture_id'] = f"G{generation+1}A{i+1}"
                # More detailed summary
                evaluation['gene_summary'] = (f"{len(gene.conv_layers)} Conv ({gene.conv_layers[-1]['out_channels']}ch), "
                                              f"{len(gene.pool_layers)} Pool, {len(gene.fc_layers)} FC ({gene.fc_layers[-1]['out_features']}u), "
                                              f"Drp={gene.dropout_rate}")
                generation_results.append((gene, evaluation))
                all_results.append(evaluation) # Keep all valid results

        # Sort current generation by fitness (descending)
        generation_results.sort(key=lambda x: x[1]['fitness'], reverse=True)

        # Log top result of the generation
        if generation_results:
             top_gene, top_eval = generation_results[0]
             print(f"  Top of Gen {generation+1}: Fit={top_eval['fitness']:.2f}, Acc={top_eval['accuracy']:.2f}%, Time={top_eval['inference_time']:.1f}ms, Size={top_eval['model_size']:.2f}MB")
        else:
             print(f"  No valid architectures evaluated in Generation {generation+1}.")

        # --- Breed Next Generation (if not the last one) ---
        if generation < generations - 1:
            new_population = []
            # Keep elites
            elites = [gene for gene, _ in generation_results[:elite_size]]
            new_population.extend(elites)

            # Generate offspring via tournament selection and crossover/mutation
            num_offspring = population_size - len(new_population)
            parents_pool = [gene for gene, _ in generation_results] # Use successful individuals as potential parents

            if not parents_pool: # If no parents survived, repopulate randomly
                print("  Warning: No surviving parents. Repopulating randomly.")
                new_population.extend([create_random_architecture() for _ in range(num_offspring)])
            else:
                for _ in range(num_offspring):
                    # Tournament selection (select 2 random parents, pick the best) - simple selection
                    parent1 = random.choice(parents_pool)
                    parent2 = random.choice(parents_pool)
                    # Note: For proper tournament, select k individuals and pick best fitness, but random choice is simpler here.

                    child = crossover(parent1, parent2)
                    if random.random() < mutation_rate:
                        child.mutate()
                    new_population.append(child)

            population = new_population # Update population for next generation

        gen_duration = time.time() - gen_start_time
        print(f"--- Generation {generation+1} complete ({gen_duration:.1f}s), {valid_evaluated_in_gen} valid architectures evaluated ---")
        # --- End Evolution Loop ---

    # --- Final Selection ---
    # Sort ALL valid results found across all generations
    valid_results = [res for res in all_results if res['success']]
    valid_results.sort(key=lambda x: x['fitness'], reverse=True)

    # Get top unique results (based on parameter count to avoid very similar models)
    unique_top_results = []
    seen_params = set()
    for res in valid_results:
        # Check params again just in case, ensure it's not zero
        param_key = int(res.get('total_params', 0))
        if param_key > 0 and param_key not in seen_params:
             unique_top_results.append(res)
             seen_params.add(param_key)
             if len(unique_top_results) >= 5: # Limit to top 5 unique results
                 break

    total_nas_time = time.time() - start_nas_time
    print(f"--- NAS Complete! ---")
    print(f"Total time: {total_nas_time:.1f}s")
    print(f"Evaluated {len(valid_results)} valid architectures in total.")
    if unique_top_results:
        print("Top 5 Unique Architectures Found:")
        for i, res in enumerate(unique_top_results):
             print(f"  #{i+1}: ID={res['architecture_id']}, Fit={res['fitness']:.2f}, Acc={res['accuracy']:.2f}%, Time={res['inference_time']:.1f}ms, Size={res['model_size']:.2f}MB")
             print(f"      Summary: {res['gene_summary']}")
    else:
        print("  No valid architectures found during the search.")


    return jsonify({
        'message': 'Neural Architecture Search complete!',
        'results': unique_top_results, # Send top 5 unique results
        'total_evaluated': len(valid_results),
        'best_fitness': unique_top_results[0]['fitness'] if unique_top_results else 0
    })


# --- Function to process uploaded photos (Kept for reference, but not used by canvas) ---
def process_image_for_mnist(image_pil):
    """Processes a PIL image (from photo) to resemble an MNIST digit."""
    try:
        # 1. Convert to Grayscale numpy array
        img_gray = image_pil.convert('L')
        img_np = np.array(img_gray)

        # 2. Thresholding (Otsu's method finds a good threshold automatically)
        # We need black background, white digit, so use THRESH_BINARY_INV
        threshold_value, img_thresh = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        print(f"    Otsu threshold value: {threshold_value}") # Log the threshold

        # 3. Find contours to locate the digit
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No contours found in image after thresholding.")

        # Find the bounding box of the largest contour (likely the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # 4. Crop the image to the bounding box
        img_cropped = img_thresh[y:y+h, x:x+w]

        # 5. Add Padding to make it square and add border
        # Make the largest dimension (w or h) the size, add padding
        size = max(w, h)
        pad_w = (size - w) // 2
        pad_h = (size - h) // 2
        # Add extra padding (e.g., 20% of the size) to prevent digit touching edges after resize
        border = int(size * 0.2)
        img_padded = cv2.copyMakeBorder(img_cropped, pad_h + border, pad_h + border, pad_w + border, pad_w + border,
                                        cv2.BORDER_CONSTANT, value=0) # Pad with black

        # 6. Resize to 28x28 (MNIST size) - Use AREA interpolation for shrinking
        img_resized = cv2.resize(img_padded, (28, 28), interpolation=cv2.INTER_AREA)

        # Convert back to PIL Image
        img_final_pil = Image.fromarray(img_resized)
        return img_final_pil

    except Exception as e:
        print(f"Error during image processing: {e}")
        # Return a blank image on error? Or raise? Let's return None.
        return None

# --- API ROUTE FOR MNIST PREDICTION (Simplified for Canvas) ---
@app.route('/predict-digit', methods=['POST'])
def predict_digit():
    global mnist_model
    if mnist_model is None:
        return jsonify({"error": "MNIST model is not loaded."}), 500

    data = request.json
    image_data = data.get('imageData') # Expecting base64 string from canvas

    try:
        if not image_data or not image_data.startswith('data:image/png;base64,'):
             raise ValueError("Invalid image data format received.")
        # Decode base64 string
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('L') # Convert directly to grayscale

        # --- Simple Processing for Canvas ---
        # 1. Resize to 28x28
        # Use ANTIALIAS for better quality resizing down
        image_pil_resized = image_pil.resize((28, 28), Image.Resampling.LANCZOS)

        # 2. Define MNIST Transform (Normalization)
        mnist_transform = transforms.Compose([
            transforms.ToTensor(), # Converts PIL [0,255] to Tensor [0,1]
            transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific mean/std
        ])

        # 3. Apply transform and add batch dimension
        image_tensor = mnist_transform(image_pil_resized).unsqueeze(0)

        # --- End Simple Processing ---

        # Move tensor to the model's device
        device = next(mnist_model.parameters()).device
        image_tensor = image_tensor.to(device)

        # Make prediction
        with torch.no_grad():
            output = mnist_model(image_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_class = probabilities.topk(1, dim=0)

        predicted_digit = top_class.item()
        confidence = top_prob.item() * 100

        print(f"Prediction: Digit={predicted_digit} with Confidence={confidence:.2f}%")

    except Exception as e:
         import traceback
         print(f"Error during prediction: {e}")
         traceback.print_exc()
         return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify({
        "predicted_digit": predicted_digit,
        "confidence": f"{confidence:.2f}"
    })


# Entry point
if __name__ == '__main__':
    # Add freeze_support() if using num_workers > 0 on Windows/macOS with multiprocessing
    torch.multiprocessing.freeze_support()
    # Run on 0.0.0.0 to make it accessible on the network if needed, use debug=False for production
    app.run(host='0.0.0.0', port=5000, debug=True)
