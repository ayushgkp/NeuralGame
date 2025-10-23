from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import os
from werkzeug.utils import secure_filename
import zipfile
import io
import time
import random
import copy # Needed for NAS crossover
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split # Needed for NAS evaluation
from torch.utils.checkpoint import checkpoint as activation_checkpoint # Needed for Memory Crystal
from torchvision import datasets, models, transforms
import torch.multiprocessing

app = Flask(__name__)
CORS(app)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'zip', 'rar', '7z', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- (FIX 2) ADDED SimpleCNN DEFINITION ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # These layers match the World 1 challenges
        # Note: Challenge 7 implies a different structure than 1-6, leading to 2048 features.
        # We will follow the challenge answers.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Challenge 7 specifies 2048 input features for fc1
        self.flatten_size = 2048

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Forward pass matching challenge 10
        x = self.pool1(self.relu1(self.conv1(x)))
        # Complete the forward pass
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)

        # Need to handle potential size mismatch if 2048 is incorrect
        if x.shape[1] != self.flatten_size:
             # This is a fallback in case architecture is wrong, but we trust challenge 7
             # A more robust way would be to dynamically calculate self.flatten_size
             # For this game, we'll assume the 2048 is the intended path
             print(f"Warning: flatten size mismatch. Expected {self.flatten_size}, got {x.shape[1]}")
             # As a simple fix, let's redefine fc1 on the fly if it's the first time
             if not hasattr(self, 'fc1_adjusted'):
                 self.fc1 = nn.Linear(x.shape[1], 512).to(x.device)
                 self.fc1_adjusted = True

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
# --- END SimpleCNN DEFINITION ---


# --- Single Player "Database" ---
player_data = {
    "playerLevel": 1, "xp": 0, "insightPoints": 100, "computeCredits": 10000,
    "currentChallenge": 1, "equippedModel": None, "equippedProject": None,
    "uploadedDatasets": [], "purchasedItems": [],
    "memoryCrystalActive": False # NEW: Add state for memory crystal
}

# --- Challenge Answer Database ---
challenge_answers = {
    1: "nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3)", 2: "nn.ReLU()",
    3: "nn.MaxPool2d(kernel_size=2)", 4: "nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3)",
    5: "nn.MaxPool2d(kernel_size=2)", 6: "nn.Flatten()", 7: "nn.Linear(2048,512)",
    8: "nn.Dropout(0.5)", 9: "nn.Linear(512,10)", 10: "self.pool1(self.relu1(self.conv1(x)))",
    # --- (FIX 5) ADDED MISSING ANSWERS ---
    11: "transforms.RandomHorizontalFlip()",
    12: "DataLoader(train_data,batch_size=64,shuffle=True)",
    13: "nn.CrossEntropyLoss()",
    14: "optim.Adam(model.parameters(),lr=0.001)",
    15: "loss.backward()",
    16: "optimizer.step()",
    17: "models.resnet18(weights='IMAGENET1K_V1')",
    18: "optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)"
    # --- END FIX 5 ---
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Helper to save progress internally ---
def save_progress_internal(data):
    global player_data
    player_data.update(data)
    print("Internal save. New data:", player_data)


# --- YOUR CORE API Routes (Kept from your working code) ---

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
    challenge_id_str = data.get('challengeId')
    user_input = data.get('userInput', '')
    equipped_model = player_data.get('equippedModel')

    # Handle both string and int challenge IDs
    try:
        challenge_id = int(challenge_id_str)
    except (ValueError, TypeError):
        return jsonify({"correct": False, "message": "Invalid challengeId format"}), 400

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

# --- YOUR UPLOAD ROUTE (Merged with his Zip Extraction) ---
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

        # --- NEW ZIP EXTRACTION LOGIC (from friend's code) ---
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
        # --- END ZIP EXTRACTION LOGIC ---

        if saved_filename_for_list not in player_data['uploadedDatasets']:
            player_data['uploadedDatasets'].append(saved_filename_for_list)
            save_progress_internal(player_data)

        return jsonify({"status": "success", "message": message, "datasets": player_data['uploadedDatasets']})
    except Exception as e:
        print(f" -> Error during file upload/saving: {e}")
        return jsonify({"status": "error", "message": f"Server error processing file: {e}"}), 500


# --- Potions / Transforms Helper ---
# --- (FIX 3) MODIFIED FUNCTION DEFINITION ---
def build_transforms(potions, is_training=True, model_name=None, dataset_name=None):
    potions = potions or []
    # Use 299 for Inception, 224 for others
    img_size = 299 if model_name == 'inception' else 224

    transform_list = []
    if is_training:
        if 'random_flip' in potions: transform_list.append(transforms.RandomHorizontalFlip())
        if 'random_rotation' in potions: transform_list.append(transforms.RandomRotation(10))
        if 'color_jitter' in potions: transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

    if model_name == 'inception':
         transform_list.extend([transforms.Resize(img_size), transforms.CenterCrop(img_size)])
    else:
        transform_list.extend([transforms.Resize(256), transforms.CenterCrop(img_size)])

    transform_list.append(transforms.ToTensor())

    # Use ImageNet normalization by default for transfer learning
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # --- (FIX 3) Use dataset_name variable ---
    if dataset_name == 'CIFAR-10' and model_name == 'SimpleCNN': # SimpleCNN was trained on CIFAR-10's norm
         normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform_list.append(normalize)

    return transforms.Compose(transform_list)


# --- Dataset Loading Helper ---
def load_data(dataset_name, active_potions, model_name):
    print(f" -> Attempting to load dataset: {dataset_name}")
    dataloaders = {}
    num_classes = 0

    # --- (FIX 3) UPDATED FUNCTION CALLS ---
    train_transform = build_transforms(active_potions, is_training=True, model_name=model_name, dataset_name=dataset_name)
    val_transform = build_transforms(None, is_training=False, model_name=model_name, dataset_name=dataset_name)

    if dataset_name == 'CIFAR-10':
        print(" -> Using built-in CIFAR-10 dataset.")
        try:
            train_dataset = datasets.CIFAR10(root=DATA_FOLDER, train=True, download=True, transform=train_transform)
            val_dataset = datasets.CIFAR10(root=DATA_FOLDER, train=False, download=True, transform=val_transform)
            dataloaders['train'] = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
            dataloaders['val'] = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
            num_classes = 10
        except Exception as e: raise ValueError(f"Failed to load CIFAR-10: {e}")

    else: # Handle uploaded datasets
        print(f" -> Using uploaded dataset: {dataset_name}")
        extract_folder_name = os.path.splitext(dataset_name)[0]
        extract_path = os.path.join(UPLOAD_FOLDER, extract_folder_name)

        if not os.path.isdir(extract_path):
             extract_path_alt = os.path.join(UPLOAD_FOLDER, dataset_name)
             if os.path.isdir(extract_path_alt): extract_path = extract_path_alt
             else: raise FileNotFoundError(f"Dataset folder not found at '{extract_path}'.")

        train_dir, val_dir = os.path.join(extract_path, 'train'), os.path.join(extract_path, 'val')
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            if os.path.exists(os.path.join(extract_path, 'seg_train', 'seg_train')):
                 train_dir = os.path.join(extract_path, 'seg_train', 'seg_train')
                 val_dir = os.path.join(extract_path, 'seg_test', 'seg_test')
            elif os.path.isdir(extract_path) and len(os.listdir(extract_path)) > 0:
                 first_subdir = os.path.join(extract_path, os.listdir(extract_path)[0])
                 if os.path.isdir(first_subdir):
                     potential_train, potential_val = os.path.join(first_subdir, 'train'), os.path.join(first_subdir, 'val')
                     if os.path.exists(potential_train) and os.path.exists(potential_val):
                         train_dir, val_dir = potential_train, potential_val
            if not os.path.exists(train_dir) or not os.path.exists(val_dir):
                 raise FileNotFoundError("Dataset structure invalid. Need 'train' and 'val' folders.")

        try:
            image_datasets = {'train': datasets.ImageFolder(train_dir, train_transform), 'val': datasets.ImageFolder(val_dir, val_transform)}
            dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=(x=='train'), num_workers=0) for x in ['train', 'val']}
            num_classes = len(image_datasets['train'].classes)
        except Exception as e: raise ValueError(f"Error loading images from folders: {e}")

    print(f" -> Dataset loaded. Found {num_classes} classes.")
    return dataloaders, num_classes


# --- Model Loading Helper ---
def get_model(model_name, num_classes):
    print(f" -> Loading model: {model_name}")
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
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.aux_logits = False
    else: # Fallback to a simple CNN if name is unknown
        print(f" -> Model '{model_name}' selected. Using SimpleCNN.")
        model = SimpleCNN(num_classes)
    return model


# --- TRAINING AND EVALUATION FUNCTIONS (with Memory Crystal logic) ---
def train_one_epoch(model, loader, device, optimizer, criterion, use_checkpoint):
    model.train()
    running_loss = 0.0; total = 0; correct = 0; batch_num = 0
    for inputs, labels in loader:
        batch_num += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        # --- MEMORY CRYSTAL (Activation Checkpointing) ---
        if use_checkpoint:
             # Checkpointing requires a function that returns model output
             # Use a lambda to make it compatible
             outputs = activation_checkpoint(lambda inp: model(inp), inputs)
        else:
             outputs = model(inputs)
        # --- END MEMORY CRYSTAL ---

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        if batch_num % 50 == 0: print(f"    Batch {batch_num}/{len(loader)}...")

        # *** THIS LINE IS NOW COMMENTED OUT FOR FULL TRAINING ***
        # if batch_num >= 100: break # Limit batches per epoch for faster demo

    epoch_loss = running_loss / max(1, total)
    epoch_acc = 100.0 * correct / max(1, total)
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    running_corrects = 0; total = 0; batch_num = 0
    for inputs, labels in loader:
        batch_num += 1
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total += labels.size(0)

        # *** THIS LINE IS NOW COMMENTED OUT FOR FULL EVALUATION ***
        # if batch_num >= 50: break # Limit batches for faster demo

    if total == 0: return 0.0
    return (running_corrects.double() / total) * 100


# --- MAIN TRAINING ROUTE (Updated) ---
@app.route('/train-model', methods=['POST'])
def train_model():
    data = request.json
    model_name = data.get('modelName')

    # --- (FIX 9) ADDED VALIDATION BLOCK ---
    # Allow 'SimpleCNN' as a free default, otherwise check purchase
    # Also check if model_name is None
    if model_name != 'SimpleCNN' and model_name not in player_data.get('purchasedItems', []):
        print(f"Auth Error: User tried to train '{model_name}' without purchasing.")
        return jsonify({"status": "error", "message": f"Model '{model_name}' is not in your purchased items. Equip a model first."}), 403
    # --- END FIX 9 ---

    dataset_name = data.get('datasetName')
    active_potions = data.get('activePotions', [])
    # Check if Memory Crystal is active (using purchasedItems)
    use_checkpoint = 'memory_crystal' in player_data.get('purchasedItems', [])

    if not model_name or not dataset_name:
        return jsonify({"status": "error", "message": "Model and dataset must be provided."}), 400

    start_time_total = time.time()
    print(f"\n--- Starting Training Job ---")
    print(f"Model: {model_name}, Dataset: {dataset_name}, Potions: {active_potions}, Checkpoint: {use_checkpoint}")

    try:
        start_time_data = time.time()
        dataloaders, num_classes = load_data(dataset_name, active_potions, model_name)
        end_time_data = time.time()
        print(f" -> Dataset loading took {end_time_data - start_time_data:.2f} seconds.")

        model = get_model(model_name, num_classes)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f" -> Using device: {device}")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        print(" -> Starting training for 1 epoch...")
        start_time_train = time.time()
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], device, optimizer, criterion, use_checkpoint)
        end_time_train = time.time()
        print(f" -> Train Epoch Complete: Loss={train_loss:.4f}, Acc={train_acc:.2f}%. Took {end_time_train - start_time_train:.2f}s.")

        print(" -> Evaluating model...")
        start_time_eval = time.time()
        accuracy = evaluate(model, dataloaders['val'], device)
        end_time_eval = time.time()
        print(f" -> Evaluation complete. Took {end_time_eval - start_time_eval:.2f} seconds.")

    except (FileNotFoundError, ValueError, RuntimeError, Exception) as e:
        print(f" !!! Training/Evaluation Error: {e}")
        return jsonify({"status": "error", "message": f"Error during training: {e}"}), 500

    end_time_total = time.time()
    total_time = end_time_total - start_time_total # Calculate total time

    print(f"--- Training Job Complete --- Final Accuracy: {accuracy:.2f}%")
    print(f"--- Total time: {total_time:.2f} seconds ---")

    return jsonify({"status": "success", "accuracy": f"{accuracy:.2f}%", "time_taken": total_time})


# --- NEURAL ARCHITECTURE SEARCH (NAS) CODE (Copied from friend's backend) ---
# This is a large, complex feature.
# It defines its own models and training loops.

class ArchitectureGene:
    """Represents a single architecture configuration"""
    def __init__(self):
        self.conv_layers = []
        self.pool_layers = []
        self.fc_layers = []
        self.dropout_rate = 0.5
        self.activation = 'relu'

    def add_conv_layer(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        self.conv_layers.append({
            'in_channels': in_channels, 'out_channels': out_channels,
            'kernel_size': kernel_size, 'stride': stride, 'padding': padding
        })
    def add_pool_layer(self, kernel_size, stride=None):
        if stride is None: stride = kernel_size
        self.pool_layers.append({'kernel_size': kernel_size, 'stride': stride})
    def add_fc_layer(self, in_features, out_features):
        self.fc_layers.append({'in_features': in_features, 'out_features': out_features})

    def mutate(self):
        mutations = [ self._mutate_add_conv, self._mutate_remove_conv, self._mutate_change_channels,
                      self._mutate_add_pool, self._mutate_change_dropout ]
        num_mutations = random.randint(1, 2)
        for _ in range(num_mutations): random.choice(mutations)()

    def _mutate_add_conv(self):
        if len(self.conv_layers) < 6:
            prev_channels = self.conv_layers[-1]['out_channels'] if self.conv_layers else 3
            new_channels = min(prev_channels * random.choice([1.5, 2]), 256)
            new_channels = max(int(new_channels), 8)
            self.add_conv_layer(prev_channels, new_channels, 3)
    def _mutate_remove_conv(self):
        if len(self.conv_layers) > 1: self.conv_layers.pop(random.randrange(len(self.conv_layers)))
    def _mutate_change_channels(self):
        if self.conv_layers:
            layer = random.choice(self.conv_layers)
            layer['out_channels'] = min(layer['out_channels'] * random.choice([0.5, 1.5, 2]), 256)
            layer['out_channels'] = max(int(layer['out_channels']), 8)
    def _mutate_add_pool(self):
        if len(self.pool_layers) < len(self.conv_layers) // 2 and len(self.pool_layers) < 3: self.add_pool_layer(2)
    def _mutate_change_dropout(self): self.dropout_rate = round(random.uniform(0.1, 0.7), 2)

    def _calculate_flatten_size(self, input_size=32):
        size = input_size
        channels = 3
        for i, conv_config in enumerate(self.conv_layers):
            conv_config['in_channels'] = channels
            size = (size - conv_config['kernel_size'] + 2 * conv_config['padding']) // conv_config['stride'] + 1
            channels = conv_config['out_channels']
            if i < len(self.pool_layers):
                size = size // self.pool_layers[i]['kernel_size']
        size = max(1, size)
        return channels * size * size

class DynamicCNN(nn.Module):
    """Dynamically generated CNN based on ArchitectureGene"""
    def __init__(self, gene: ArchitectureGene, num_classes: int = 10, input_size=32):
        super().__init__()
        self.gene = gene
        self.features = nn.ModuleList()
        self.classifier = nn.ModuleList()

        current_channels = 3
        current_size = input_size
        pool_idx = 0

        for i, conv_config in enumerate(gene.conv_layers):
            conv_config['in_channels'] = current_channels
            conv = nn.Conv2d(conv_config['in_channels'], conv_config['out_channels'],
                             conv_config['kernel_size'], stride=conv_config['stride'], padding=conv_config['padding'])
            self.features.append(conv)
            self.features.append(nn.ReLU(inplace=True))
            current_channels = conv_config['out_channels']
            current_size = (current_size - conv_config['kernel_size'] + 2 * conv_config['padding']) // conv_config['stride'] + 1

            if pool_idx < len(gene.pool_layers):
                 pool_config = gene.pool_layers[pool_idx]
                 pool = nn.MaxPool2d(pool_config['kernel_size'], stride=pool_config['stride'])
                 self.features.append(pool)
                 current_size = current_size // pool_config['kernel_size']
                 pool_idx += 1

        current_size = max(1, current_size)
        self.flatten_size = current_channels * current_size * current_size

        # --- (FIX 7) ADDED FALLBACK CHECK ---
        if self.flatten_size <= 0:
            print(f"Warning: Invalid NAS architecture. flatten_size is {self.flatten_size}. Defaulting to 512.")
            self.flatten_size = 512 # Fallback to prevent crash
        # --- END FIX 7 ---

        self.classifier.append(nn.Flatten())
        current_features = self.flatten_size

        if not gene.fc_layers:
             gene.add_fc_layer(current_features, max(current_features // 4, 64))
             current_features = gene.fc_layers[0]['out_features']

        if gene.fc_layers:
             gene.fc_layers[0]['in_features'] = self.flatten_size

        for i, fc_config in enumerate(gene.fc_layers):
            if i > 0: fc_config['in_features'] = gene.fc_layers[i-1]['out_features']
            else: fc_config['in_features'] = self.flatten_size

            fc = nn.Linear(fc_config['in_features'], fc_config['out_features'])
            self.classifier.append(fc)
            current_features = fc_config['out_features']
            if i < len(gene.fc_layers) - 1:
                self.classifier.append(nn.ReLU(inplace=True))
                self.classifier.append(nn.Dropout(gene.dropout_rate))

        if current_features != num_classes:
            self.classifier.append(nn.Linear(current_features, num_classes))

    # --- (FIX 4) REPLACED forward METHOD ---
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        for layer in self.classifier:
            x = layer(x)
        return x
    # --- END FIX 4 ---

# *** NAS Stricter Constraints Applied Here ***
def create_random_architecture() -> ArchitectureGene:
    """Create a random initial architecture with stricter limits for low VRAM"""
    gene = ArchitectureGene()
    # Max 3 conv layers
    num_conv = random.randint(2, 3); current_channels = 3
    for i in range(num_conv):
        # Limit max out_channels to 32
        out_channels = min(current_channels * random.choice([1.5, 2]), 32); out_channels = max(int(out_channels), 8)
        gene.add_conv_layer(current_channels, out_channels, 3)
        current_channels = out_channels
        # Increase chance of pooling to 80%
        if random.random() < 0.8 and len(gene.pool_layers) < i // 2 + 1 and len(gene.pool_layers) < 3 : gene.add_pool_layer(2)

    # Calculate flatten size based on 224 input
    flatten_approx = current_channels * (224 // (2**len(gene.pool_layers)))**2

    # Keep FC layer limit at 256
    fc1_out = max(min(flatten_approx // 8, 256), 32)
    if fc1_out <= 0: fc1_out = 32 # Handle edge case
    if flatten_approx <= 0: flatten_approx = 512 # Handle edge case
    gene.add_fc_layer(flatten_approx, fc1_out)
    gene.dropout_rate = round(random.uniform(0.2, 0.6), 2)
    return gene


def crossover(parent1: ArchitectureGene, parent2: ArchitectureGene) -> ArchitectureGene:
    child = ArchitectureGene()
    len1, len2 = len(parent1.conv_layers), len(parent2.conv_layers)
    child.conv_layers = copy.deepcopy(random.choice([parent1.conv_layers[:len1//2] + parent2.conv_layers[len2//2:],
                                                     parent2.conv_layers[:len2//2] + parent1.conv_layers[len1//2:]]))
    child.pool_layers = copy.deepcopy(random.choice([parent1.pool_layers, parent2.pool_layers]))
    child.fc_layers = copy.deepcopy(random.choice([parent1.fc_layers, parent2.fc_layers]))
    child.dropout_rate = random.choice([parent1.dropout_rate, parent2.dropout_rate])
    # Fix in_channels after crossover
    channels = 3
    for layer in child.conv_layers:
        layer['in_channels'] = channels
        channels = layer['out_channels']
    return child

def evaluate_architecture_real(gene: ArchitectureGene, train_loader, val_loader, device, num_classes, epochs=2):
    start_eval_time = time.time()
    try:
        # *** NAS FIX 2: Explicitly pass input_size=224 to the model builder ***
        model = DynamicCNN(gene, num_classes=num_classes, input_size=224)

        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            model.train()
            batch_count = 0
            for data, target in train_loader:
                if batch_count >= 30: break
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                batch_count += 1

        model.eval()
        correct, total = 0, 0
        inference_times = []
        batch_count = 0
        with torch.no_grad():
            for data, target in val_loader:
                if batch_count >= 15: break
                data, target = data.to(device), target.to(device)
                eval_start = time.time()
                output = model(data)
                inference_times.append(time.time() - eval_start)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                batch_count += 1

        accuracy = 100. * correct / total if total > 0 else 0
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 999
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / (1024 * 1024)
        eval_duration = time.time() - start_eval_time
        print(f"    Evaluated arch: Acc={accuracy:.2f}%, Time={avg_inference_time*1000:.1f}ms, Size={model_size_mb:.2f}MB ({eval_duration:.1f}s)")

        return {'accuracy': accuracy, 'inference_time': avg_inference_time * 1000,
                'model_size': model_size_mb, 'total_params': total_params, 'success': True}
    except Exception as e:
        eval_duration = time.time() - start_eval_time
        print(f"    Architecture evaluation failed: {e} ({eval_duration:.1f}s)")
        return {'accuracy': 0.0, 'inference_time': 9999.0, 'model_size': 999.0, 'total_params': 0, 'success': False}


@app.route('/nas', methods=['POST'])
def neural_architecture_search():
    """Real Neural Architecture Search using evolutionary algorithm"""
    data = request.json or {}
    preferences = data.get('preferences', {})
    potions = data.get('potions', [])

    print("--- Real NAS Search Requested ---")
    print(f"Preferences: {preferences}")
    print(f"Potions: {potions}")

    start_nas_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        nas_train_transform = build_transforms(potions, is_training=True, model_name=None) # Use default 224x224
        nas_val_transform = build_transforms(None, is_training=False, model_name=None)
        train_dataset = datasets.CIFAR10(root=DATA_FOLDER, train=True, download=True, transform=nas_train_transform)
        val_dataset = datasets.CIFAR10(root=DATA_FOLDER, train=False, download=True, transform=nas_val_transform)
        train_subset, _ = random_split(train_dataset, [5000, len(train_dataset) - 5000])
        val_subset, _ = random_split(val_dataset, [1000, len(val_dataset) - 1000])
        # Use smaller batch size to reduce VRAM usage during evaluation
        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=0)
        num_classes = 10
    except Exception as e:
         return jsonify({"status": "error", "message": f"Failed to load dataset for NAS: {e}"}), 500

    population_size = 10
    generations = 5
    elite_size = 2

    population = [create_random_architecture() for _ in range(population_size)]
    all_results = []

    for generation in range(generations):
        gen_start_time = time.time()
        print(f"--- NAS Generation {generation + 1}/{generations} ---")
        generation_results = []
        for i, gene in enumerate(population):
            print(f"  Evaluating architecture {i+1}/{len(population)}...")
            evaluation = evaluate_architecture_real(gene, train_loader, val_loader, device, num_classes, epochs=1)

            if evaluation['success']:
                acc_pref = preferences.get('accuracy', 50) / 100.0
                speed_pref = preferences.get('speed', 30) / 100.0
                eff_pref = preferences.get('efficiency', 20) / 100.0
                norm_acc = evaluation['accuracy'] / 100.0
                norm_speed = max(0, 1 - (evaluation['inference_time'] / 200.0))
                norm_eff = max(0, 1 - (evaluation['model_size'] / 50.0))
                fitness = (norm_acc * acc_pref + norm_speed * speed_pref + norm_eff * eff_pref) * 100

                evaluation['fitness'] = round(fitness, 2)
                evaluation['generation'] = generation + 1
                evaluation['architecture_id'] = f"G{generation+1}A{i+1}"
                evaluation['gene_summary'] = f"{len(gene.conv_layers)} Conv, {len(gene.pool_layers)} Pool, {len(gene.fc_layers)} FC"
                generation_results.append((gene, evaluation))
                all_results.append(evaluation)

        generation_results.sort(key=lambda x: x[1]['fitness'], reverse=True)

        if generation_results:
             top_gene, top_eval = generation_results[0]
             print(f"  Top of Gen {generation+1}: Fit={top_eval['fitness']:.2f}, Acc={top_eval['accuracy']:.2f}%, Time={top_eval['inference_time']:.1f}ms, Size={top_eval['model_size']:.2f}MB")

        if generation < generations - 1:
            elite_genes = [gene for gene, _ in generation_results[:elite_size]]
            new_population = elite_genes[:]

            while len(new_population) < population_size:
                if len(elite_genes) >= 2:
                    parent1, parent2 = random.sample(elite_genes, 2)
                    child = crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(random.choice(elite_genes if elite_genes else population)) # Failsafe

                child.mutate()
                new_population.append(child)
            population = new_population

        gen_duration = time.time() - gen_start_time
        print(f"--- Generation {generation+1} complete ({gen_duration:.1f}s) ---")

    all_results.sort(key=lambda x: x['fitness'], reverse=True)
    unique_top_results = []
    seen_params = set()
    for res in all_results:
        # Check if the model has a valid (non-zero) number of parameters
        if res['total_params'] > 0 and res['total_params'] not in seen_params:
             unique_top_results.append(res)
             seen_params.add(res['total_params'])
             if len(unique_top_results) >= 5:
                 break

    total_nas_time = time.time() - start_nas_time
    print(f"--- NAS Complete! ---")
    print(f"Total time: {total_nas_time:.1f}s")
    # Make sure we actually evaluated something before printing
    if all_results:
        print(f"Evaluated {len([r for r in all_results if r['success']])} valid architectures.")
    else:
        print("Evaluated 0 valid architectures.")
        
    for i, res in enumerate(unique_top_results):
         print(f"  #{i+1}: ID={res['architecture_id']}, Fit={res['fitness']:.2f}, Acc={res['accuracy']:.2f}%, Time={res['inference_time']:.1f}ms, Size={res['model_size']:.2f}MB")

    return jsonify({
        'message': 'Neural Architecture Search complete!',
        'results': unique_top_results,
        'total_evaluated': len([r for r in all_results if r['success']]), # Count only successful evaluations
        'best_fitness': unique_top_results[0]['fitness'] if unique_top_results else 0
    })


# Entry point
if __name__ == '__main__':
    # Add freeze_support() if using num_workers > 0 on Windows
    torch.multiprocessing.freeze_support()
    app.run(debug=True, port=5000)