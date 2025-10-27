document.addEventListener('DOMContentLoaded', () => {
    const game = {
        backendUrl: 'http://localhost:5000', // Use localhost for better compatibility

        // --- PLAYER STATE ---
        playerLevel: 1, xp: 0, xpToNextLevel: 100, insightPoints: 100,
        computeCredits: 10000, // Start with more for testing
        hintUsedThisLevel: false, equippedModel: null,
        equippedProject: null, uploadedDatasets: [], purchasedItems: [],
        currentChallenge: 1,

        // --- Potion State ---
        activePotions: [],

        // --- NAS Lab State (from friend's code) ---
        nasUnlocked: false,
        nasPreferences: { accuracy: 50, speed: 30, efficiency: 20 },

        // --- Debug Mode State (from friend's code) ---
        debugMode: false,

        // --- REVERTED: MNIST Lab State (Back to Drawing Canvas) ---
        mnistCanvas: null,
        mnistCtx: null,
        isDrawing: false,

        // NEW: Model Memory Costs (as a percentage)
        // SimpleCNN = 10%
        // VGG = 120% (requires crystal)
        // ResNet = 110% (requires crystal)
        // Inception = 100% (just barely fits)
        // EfficientNet = 40%
        // Memory Crystal = 1000% (virtually unlimited)
        memoryCosts: {
            'SimpleCNN': 10,
            'vgg16': 120,
            'resnet50': 110,
            'efficientnet': 40,
            'inception': 100,
            'default': 10
        },

        worlds: [
            { id: 1, name: "CNN Fundamentals", unlocked: true },
            { id: 2, name: "The Training Pipeline", unlocked: false, unlocksAt: 4 },
            { id: 3, name: "Advanced Techniques", unlocked: false, unlocksAt: 6 },
        ],
        challenges: [
            // Note: Answers are removed from the frontend. The backend is the source of truth.
            { world: 1, title: "The First Layer", instructions: `<p>Welcome, Architect! Let's build a CNN.</p><p>Every CNN starts with a convolutional layer, <code class='code-class'>Conv2d</code>. It needs <code class='code-variable'>in_channels</code> (3 for RGB) and produces <code class='code-variable'>out_channels</code>.</p><p class="goal"><strong>Goal:</strong> Use <code class='code-function'>nn.Conv2d</code> with <code class='code-number'>16</code> output channels and a kernel size of <code class='code-number'>3</code>.</p>`, code: `class SimpleCNN(nn.Module):<br>    def __init__(self):<br>        super().__init__()<br>        self.conv1 = [_]`, hint: "Use all three keyword arguments: in_channels, out_channels, and kernel_size.", hintCost: 50, insightReward: 50, vis: { type: 'conv', label: 'Conv2D (16)', size: 90 } },
            { world: 1, title: "Activation", instructions: `<p>After a convolution, we need a non-linear activation function. The most common is <code class='code-class'>ReLU</code>.</p><p class="goal"><strong>Goal:</strong> Add a <code class='code-function'>nn.ReLU()</code> layer.</p>`, code: `class SimpleCNN(nn.Module):<br>    def __init__(self):<br>        # ...<br>        self.conv1 = nn.Conv2d(...)<br>        self.relu1 = [_]`, hint: "This function is called with empty parentheses.", hintCost: 50, insightReward: 50, vis: { type: 'relu', label: 'ReLU' } },
            { world: 1, title: "Pooling", instructions: `<p>Now, let's add a <code class='code-class'>MaxPool2d</code> layer to downsample the feature map, making the network faster and more robust.</p><p class="goal"><strong>Goal:</strong> Add a <code class='code-function'>nn.MaxPool2d</code> layer with a kernel size of <code class='code-number'>2</code>.</p>`, code: `class SimpleCNN(nn.Module):<br>    def __init__(self):<br>        # ...<br>        self.relu1 = nn.ReLU()<br>        self.pool1 = [_]`, hint: "The argument for the window size is kernel_size.", hintCost: 50, insightReward: 50, vis: { type: 'pool', label: 'MaxPool2D', size: 45 } },
            { world: 1, title: "Stacking Layers", instructions: `<p>Real power comes from stacking blocks. Let's add a second convolutional layer.</p><p>The <code class='code-variable'>in_channels</code> must match the previous <code class='code-variable'>out_channels</code> (<code class='code-number'>16</code>). We'll increase features to <code class='code-number'>32</code>.</p><p class="goal"><strong>Goal:</strong> Add a second <code class='code-function'>nn.Conv2d</code> layer.</p>`, code: `class SimpleCNN(nn.Module):<br>    def __init__(self):<br>        # ... (First Block)<br>        self.pool1 = nn.MaxPool2d(...)<br><br>        # Second Block<br>        self.conv2 = [_]`, hint: "Input channels here must match the output channels of self.conv1.", hintCost: 50, insightReward: 75, vis: { type: 'conv', label: 'Conv2D (32)', size: 40 } },
            { world: 1, title: "Completing the Block", instructions: `<p>Let's complete our second convolutional block with its own ReLU and MaxPool layers.</p><p class="goal"><strong>Goal:</strong> Add a <code class='code-function'>nn.ReLU()</code> and a <code class='code-function'>nn.MaxPool2d(kernel_size=2)</code>.</p>`, code: `        # ...<br>        self.conv2 = nn.Conv2d(...)<br>        self.relu2 = nn.ReLU()<br>        self.pool2 = [_]`, hint: "It's the same pooling layer as the first block.", hintCost: 50, insightReward: 75, vis: [{ type: 'relu', label: 'ReLU' },{ type: 'pool', label: 'MaxPool2D', size: 20 }] },
            { world: 1, title: "Flattening the Output", instructions: `<p>After the convolutional blocks, we must flatten the 2D feature map into a 1D vector to prepare it for the final classification layers.</p><p class="goal"><strong>Goal:</strong> Add a <code class='code-function'>nn.Flatten()</code> layer.</p>`, code: `class SimpleCNN(nn.Module):<br>    def __init__(self):<br>        # ...<br>        self.pool2 = nn.MaxPool2d(...)<br><br>        self.flatten = [_]`, hint: "The function to unroll a tensor is simply called Flatten.", hintCost: 50, insightReward: 75, vis: { type: 'flat', label: 'Flatten', size: 100 } },
            { world: 1, title: "The Classifier Head", instructions: `<p>A <code class='code-class'>Linear</code> layer performs classification on the flattened vector. The input size depends on the output of the conv layers. After two pooling layers, a 32x32 image becomes 8x8. So, the size is 32 channels * 8 * 8 = 2048.</p><p class="goal"><strong>Goal:</strong> Add a <code class='code-function'>nn.Linear</code> layer from <code class='code-number'>2048</code> features to <code class='code-number'>512</code>.</p>`, code: `        # ...<br>        self.flatten = nn.Flatten()<br>        self.fc1 = [_]`, hint: "The function takes in_features and out_features.", hintCost: 75, insightReward: 100, vis: { type: 'linear', label: 'Linear (512)', size: 80 } },
            { world: 1, title: "Adding Dropout", instructions: `<p>To prevent overfitting, we can add a <code class='code-class'>Dropout</code> layer. It randomly sets some inputs to zero during training, forcing the network to be more robust.</p><p class="goal"><strong>Goal:</strong> Add a <code class='code-function'>nn.Dropout(0.5)</code> layer with a 50% probability.</p>`, code: `        # ...<br>        self.fc1 = nn.Linear(2048, 512)<br>        self.dropout = [_]`, hint: "The argument is the probability p of an element to be zeroed.", hintCost: 75, insightReward: 100, vis: { type: 'dropout', label: 'Dropout' } },
            { world: 1, title: "The Final Output", instructions: `<p>The final layer must have an output size equal to the number of classes. For CIFAR-10, there are 10 classes.</p><p class="goal"><strong>Goal:</strong> Add the final <code class='code-function'>nn.Linear</code> layer with 10 outputs.</p>`, code: `        # ...<br>        self.dropout = nn.Dropout(0.5)<br>        self.fc2 = [_]`, hint: "The in_features here must match the out_features of self.fc1.", hintCost: 50, insightReward: 100, vis: { type: 'linear', label: 'Output (10)', size: 60 } },
            { world: 1, title: "The Forward Pass", instructions: `<p>You've defined the architecture! Now, in the <code class='code-function'>forward</code> method, we define how data flows through it.</p><p class="goal"><strong>Goal:</strong> Pass the input <code class='code-variable'>x</code> through the first block: conv1 -> relu1 -> pool1.</p>`, code: `    def forward(self, x):<br>        # Pass data through the first block<br>        x = [_]`, hint: "Nest the functions, with the input x in the innermost one.", hintCost: 75, insightReward: 125 },
            { world: 2, title: "Image Augmentation", instructions: `<p>Let's prepare our data. To make the model more robust, we randomly alter training images. This is **augmentation**.</p><p class="goal"><strong>Goal:</strong> Add <code class='code-function'>transforms.RandomHorizontalFlip()</code> to our list of transforms.</p>`, code: `transform = transforms.Compose([<br>    [_],<br>    transforms.ToTensor(),<br>    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),<br>])`, hint: "The function name describes exactly what it does.", hintCost: 75, insightReward: 125, vis: { type: 'data', label: 'Augmentation' } },
            { world: 2, title: "The DataLoader", instructions: `<p>The <code class='code-class'>DataLoader</code> is a utility that feeds data to our model in batches during training.</p><p class="goal"><strong>Goal:</strong> Create a <code class='code-function'>DataLoader</code> with a <code class='code-variable'>batch_size</code> of 64 and shuffle the data.</p>`, code: `train_data = YourDataset(...)<br>train_loader = [_]`, hint: "The three main arguments are the dataset, batch_size, and shuffle.", hintCost: 75, insightReward: 125, vis: { type: 'data', label: 'DataLoader' } },
            { world: 2, title: "Loss Function", instructions: `<p>The loss function measures how wrong the model's prediction is compared to the actual label. For multi-class classification, <code class='code-class'>CrossEntropyLoss</code> is the standard.</p><p class="goal"><strong>Goal:</strong> Define the criterion as <code class='code-function'>nn.CrossEntropyLoss()</code>.</p>`, code: `model = SimpleCNN()<br>optimizer = ...<br>criterion = [_]`, hint: "This function takes no arguments for basic use.", hintCost: 75, insightReward: 125, vis: { type: 'optim', label: 'Loss Fn' } },
            { world: 2, title: "The Optimizer", instructions: `<p>The optimizer is the algorithm that updates the model's weights to reduce the loss. A popular choice is <code class='code-class'>Adam</code>.</p><p class="goal"><strong>Goal:</strong> Create an <code class='code-function'>optim.Adam</code> optimizer with a learning rate of <code class='code-number'>0.001</code>.</p>`, code: `model = SimpleCNN()<br>criterion = nn.CrossEntropyLoss()<br>optimizer = [_]`, hint: "The first argument is model.parameters(), and the second is lr.", hintCost: 75, insightReward: 125, vis: { type: 'optim', label: 'Optimizer' } },
            { world: 2, title: "Backpropagation", instructions: `<p>This is the core of learning! <code class='code-function'>loss.backward()</code> calculates how much each model weight contributed to the error.</p><p class="goal"><strong>Goal:</strong> Call the backward pass on the loss.</p>`, code: `# Inside the training loop...<br>loss = criterion(outputs, labels)<br>[_]`, hint: "Just call the .backward() method on the loss variable.", hintCost: 100, insightReward: 150 },
            { world: 2, title: "Updating Weights", instructions: `<p>After calculating the gradients with <code>backward()</code>, we tell the optimizer to update the model's weights based on those gradients.</p><p class="goal"><strong>Goal:</strong> Call the optimizer's step function to update the model.</p>`, code: `# Inside the training loop...<br>loss.backward()<br>[_]`, hint: "The method is called .step().", hintCost: 100, insightReward: 150 },
            { world: 3, title: "Transfer Learning", instructions: `<p>Instead of building from scratch, we can use a powerful, pre-trained model like <code class='code-class'>ResNet</code>. This is **Transfer Learning**.</p><p class="goal"><strong>Goal:</strong> Load a pre-trained <code class='code-function'>models.resnet18</code> from the store.</p>`, code: `# Load a powerful pre-trained model<br>model = [_]`, hint: "The argument to get pre-trained weights is weights='IMAGENET1K_V1'.", hintCost: 100, insightReward: 250, vis: { type: 'model', label: 'ResNet-18' } },
            { world: 3, title: "Learning Rate Scheduler", instructions: `<p>A scheduler adjusts the learning rate during training. A common strategy is to decrease it over time to fine-tune the model.</p><p class="goal"><strong>Goal:</strong> Create a <code class='code-function'>StepLR</code> scheduler that reduces the LR by a factor of 0.1 every 5 epochs.</p>`, code: `optimizer = optim.Adam(...)<br>scheduler = [_]`, hint: "The scheduler needs the optimizer, a step_size, and a gamma factor.", hintCost: 100, insightReward: 250, vis: { type: 'optim', label: 'LR Scheduler' } },
        ],
        datasets: [ { id: 'cifar10', name: 'CIFAR-10', icon: '🖼️', description: 'Default starter dataset.' } ],
        shopItems: [
            { id: 'vgg16', name: 'VGG-16', icon: '🏛️', cost: 1500, purchased: false, stats: { acc: '92%', speed: 'Medium' } },
            { id: 'resnet50', name: 'ResNet-50', icon: '🚀', cost: 2500, purchased: false, stats: { acc: '94%', speed: 'Fast' } },
            { id: 'efficientnet', name: 'EfficientNet', icon: '⚡', cost: 4000, purchased: false, stats: { acc: '95%', speed: 'Very Fast' } },
            { id: 'inception', name: 'InceptionV3', icon: '🌀', cost: 3500, purchased: false, stats: { acc: '94%', speed: 'Medium' } },
            // --- NEW: Memory Crystal Item ---
            { id: 'memory_crystal', name: 'Memory Crystal', icon: '💎', cost: 10000, purchased: false, stats: { acc: 'N/A', speed: 'Slower' }, description: 'Enables training massive models with low memory via checkpointing.' }
        ],
        projectItems: [
            { id: 'img_classification', name: 'Image Classifier', icon: '📸', cost: 5000, purchased: false, description: 'A complete blueprint for a state-of-the-art image classification project.' },
            { id: 'object_detection', name: 'Object Detector', icon: '🎯', cost: 8000, purchased: false, description: 'Learn to build a project that can find and identify objects in an image.' }
        ],
        potions: [
            { id: 'random_flip', name: 'Flipping Charm', icon: '↔️', description: 'Randomly flips images horizontally.', cost: 50 },
            { id: 'random_rotation', name: 'Potion of Rotation', icon: '🔄', description: 'Randomly rotates images slightly.', cost: 75 },
            { id: 'color_jitter', name: 'Elixir of Brightness', icon: '✨', description: 'Randomly changes brightness, contrast, and saturation.', cost: 100 }
        ],

        init() {
            const self = this; // Store the context of the 'game' object

            fetch(`${this.backendUrl}/get-progress`)
                .then(response => response.json())
                .then(data => {
                    Object.assign(self, data);
                    self.purchasedItems = self.purchasedItems || [];
                    self.shopItems.forEach(item => item.purchased = self.purchasedItems.includes(item.id));
                    self.projectItems.forEach(item => item.purchased = self.purchasedItems.includes(item.id));
                    // Also sync memory crystal
                    self.memoryCrystalActive = self.purchasedItems.includes('memory_crystal');
                    self.renderAll();
                })
                .catch(error => {
                    console.error('Error loading data:', error);
                    self.showNotification('Error: Could not connect to backend!', 'error');
                    self.renderAll(); // Render with defaults even if backend fails
                });

            this.initEventListeners();
            this.initDebugMode(); // Initialize debug mode
            this.initMNISTLab(); // Initialize the MNIST Lab logic (now back to canvas)
        },

        renderAll() {
            this.renderWorldMap();
            this.renderShop();
            this.renderProjectStore();
            this.renderDataGarden();
            this.renderNAS(); // Render NAS screen
            this.updatePlayerStats();
            this.initNavigation();
            this.updateMemoryBar(); // Update memory bar on load
        },

        initEventListeners() {
             const runButton = document.getElementById('run-button');
             const hintButton = document.getElementById('hint-button');
             // Check if buttons exist before adding listeners
             if(runButton && !runButton.dataset.listenerAttached) {
                 runButton.addEventListener('click', () => this.checkAnswer());
                 runButton.dataset.listenerAttached = 'true';
             }
             if(hintButton && !hintButton.dataset.listenerAttached) {
                 hintButton.addEventListener('click', () => this.useHint());
                 hintButton.dataset.listenerAttached = 'true';
             }

             // Add listener for the modal close button
             const modalClose = document.getElementById('modal-close-btn');
             if(modalClose && !modalClose.dataset.listenerAttached) {
                 modalClose.addEventListener('click', () => {
                     document.getElementById('training-result-modal').classList.remove('show');
                 });
                 modalClose.dataset.listenerAttached = 'true';
             }
             // NAS Listeners are initialized in renderNAS
        },

        saveProgressToServer() {
            const dataToSave = {
                playerLevel: this.playerLevel, xp: this.xp, insightPoints: this.insightPoints,
                computeCredits: this.computeCredits, currentChallenge: this.currentChallenge,
                equippedModel: this.equippedModel, equippedProject: this.equippedProject,
                uploadedDatasets: this.uploadedDatasets, purchasedItems: this.purchasedItems,
                memoryCrystalActive: this.memoryCrystalActive
            };
            fetch(`${this.backendUrl}/save-progress`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(dataToSave),
            })
            .then(res => res.json())
            .then(data => console.log('Progress saved:', data.data))
            .catch(error => console.error('Error saving progress:', error));
        },

        renderDataGarden() {
            const plotsContainer = document.getElementById('dataset-plots');
            const potionsContainer = document.getElementById('potions-shed');
            if (!plotsContainer || !potionsContainer) return;

            // --- Render Datasets ---
            plotsContainer.innerHTML = '';
            const allDatasets = [...this.datasets];
            (this.uploadedDatasets || []).forEach(filename => {
                if (!allDatasets.some(ds => ds.name === filename)) {
                    allDatasets.push({ id: filename.toLowerCase().replace(/[^a-z0-9]/g, ''), name: filename, icon: '📁', description: 'User uploaded dataset.' });
                }
            });
            allDatasets.forEach(ds => {
                const plotEl = document.createElement('div');
                plotEl.className = 'plot';
                plotEl.innerHTML = `
                    <div class="plot-icon">${ds.icon}</div>
                    <h3>${ds.name}</h3>
                    <p>${ds.description}</p>
                    <button class="train-button" data-dataset-name="${ds.name}">Train</button>
                `;
                plotsContainer.appendChild(plotEl);
            });
            // --- Upload Plot ---
            const uploadPlot = document.createElement('div');
            uploadPlot.className = 'plot upload-plot';
            uploadPlot.innerHTML = `
                <div class="plot-icon">📤</div>
                <h3>Upload Dataset</h3>
                <p>Select a file to upload.</p>
                <input type="file" id="dataset-upload-input" accept=".zip,.rar,.7z,.png,.jpg,.jpeg" style="display: none;">
            `;
            uploadPlot.addEventListener('click', () => {
                 const input = document.getElementById('dataset-upload-input');
                 if(input) input.click();
             });
            plotsContainer.appendChild(uploadPlot);

            // --- Render Potions ---
            potionsContainer.innerHTML = '<h2 class="potions-title">Potion Brewery</h2>';
            this.potions.forEach(potion => {
                const potionEl = document.createElement('div');
                potionEl.className = 'potion-item';
                potionEl.innerHTML = `
                    <label>
                        <input type="checkbox" class="potion-checkbox" value="${potion.id}" ${this.activePotions.includes(potion.id) ? 'checked' : ''}>
                        <span class="potion-icon">${potion.icon}</span>
                        <strong>${potion.name}</strong> (-${potion.cost} CC)
                    </label>
                    <p class="potion-desc">${potion.description}</p>
                `;
                potionsContainer.appendChild(potionEl);
            });

            this.updateMemoryBar(); // Update memory bar when garden is rendered
            this.initDataGardenUpload();
            this.initTrainButtons();
            this.initPotionSelectors();
        },

        initDataGardenUpload() {
            const fileInput = document.getElementById('dataset-upload-input');
            if (!fileInput || fileInput.dataset.listenerAdded) return;

            fileInput.addEventListener('change', (event) => {
                const file = event.target.files[0];
                if (file) {
                    this.showNotification(`Uploading "${file.name}"...`, 'unlock');
                    const formData = new FormData();
                    formData.append('file', file);

                    fetch(`${this.backendUrl}/upload-dataset`, { method: 'POST', body: formData })
                        .then(response => response.json())
                        .then(result => {
                            if (result.status === 'success') {
                                this.showNotification(result.message, 'success');
                                this.uploadedDatasets = result.datasets;
                                this.renderDataGarden();
                            } else { this.showNotification(`Error: ${result.message}`, 'error'); }
                        })
                        .catch(error => this.showNotification('Upload failed! Server error.', 'error'));
                }
            });
            fileInput.dataset.listenerAdded = 'true';
        },

        initPotionSelectors() {
            document.querySelectorAll('.potion-checkbox').forEach(checkbox => {
                 const newCheckbox = checkbox.cloneNode(true);
                 checkbox.parentNode.replaceChild(newCheckbox, checkbox);

                newCheckbox.addEventListener('change', (event) => {
                    const potionId = event.target.value;
                    if (event.target.checked) {
                        if (!this.activePotions.includes(potionId)) {
                             this.activePotions.push(potionId);
                        }
                    } else {
                        this.activePotions = this.activePotions.filter(id => id !== potionId);
                    }
                     console.log("Activated potions:", this.activePotions);
                });
            });
        },

        initTrainButtons() {
            document.querySelectorAll('.train-button').forEach(button => {
                const newButton = button.cloneNode(true);
                button.parentNode.replaceChild(newButton, button);

                newButton.addEventListener('click', () => {
                    const datasetName = newButton.dataset.datasetName;
                    if (!this.equippedModel && datasetName !== 'MNIST') { // Allow training MNIST without equipping SimpleCNN
                        this.showNotification('You must equip a model first!', 'error'); return;
                    }

                    // --- NEW MEMORY CHECK ---
                    // Use 'SimpleCNN' if no model is equipped (implied for MNIST)
                    const modelToCheck = this.equippedModel || (datasetName === 'MNIST' ? 'SimpleCNN' : null);
                    if (!modelToCheck) { // Should not happen if previous check passed, but safeguard
                         this.showNotification('Error: Cannot determine model for memory check.', 'error'); return;
                    }
                    const modelCost = this.memoryCosts[modelToCheck] || 10;
                    const crystalActive = this.purchasedItems.includes('memory_crystal');
                    if (modelCost > 100 && !crystalActive) {
                        this.showNotification(`Training Failed: Memory Overload! ${modelToCheck} requires ${modelCost}%. Equip a Memory Crystal.`, 'error');
                        return; // Stop the training
                    }
                    // --- END MEMORY CHECK ---

                    let totalPotionCost = 0;
                    this.activePotions.forEach(potionId => {
                        totalPotionCost += this.potions.find(p => p.id === potionId)?.cost || 0;
                    });

                    if (this.computeCredits < totalPotionCost) {
                        this.showNotification(`Not enough Credits for potions! Cost: ${totalPotionCost} C`, 'error');
                        return;
                    }

                    const confirmMsg = `Train ${modelToCheck} on ${datasetName}?` + (totalPotionCost > 0 ? ` Potion cost: ${totalPotionCost} CC.` : '');
                    const confirmTrain = confirm(confirmMsg);
                    if (!confirmTrain) return;


                    this.computeCredits -= totalPotionCost;
                    this.updatePlayerStats();
                    this.saveProgressToServer();

                    this.showNotification(`Training ${modelToCheck} on ${datasetName}... This may take a moment.`, 'unlock');

                    fetch(`${this.backendUrl}/train-model`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            modelName: this.equippedModel, // Backend defaults to SimpleCNN if null
                            datasetName: datasetName,
                            activePotions: this.activePotions
                        })
                    })
                    .then(response => {
                         if (!response.ok) { // Check for HTTP errors first
                            return response.json().then(err => { throw new Error(err.message || `Server error: ${response.statusText}`); });
                         }
                         return response.json();
                    })
                    .then(result => {
                        if (result.status === 'success') {
                            document.getElementById('result-model-name').textContent = modelToCheck;
                            document.getElementById('result-dataset-name').textContent = datasetName;
                            document.getElementById('result-accuracy').textContent = result.accuracy;
                            document.getElementById('result-time').textContent = `${parseFloat(result.time_taken).toFixed(2)} seconds`;
                            document.getElementById('training-result-modal').classList.add('show');
                        } else {
                            // This case might not be reached if using the error check above, but keep as fallback
                            this.showNotification(`Training Failed: ${result.message}`, 'error');
                            this.computeCredits += totalPotionCost; // Refund
                            this.updatePlayerStats();
                            this.saveProgressToServer();
                        }
                    })
                    .catch(error => {
                        console.error('Training fetch error:', error); // Log the actual error
                        this.showNotification(`Training failed! ${error.message}`, 'error');
                        this.computeCredits += totalPotionCost; // Refund
                        this.updatePlayerStats();
                        this.saveProgressToServer();
                    });
                });
            });
        },

        // --- CORRECT RENDER WORLD MAP FUNCTION ---
        renderWorldMap() {
            const container = document.getElementById('world-map-screen');
            if (!container) return;
            // Clear previous content and add title
            container.innerHTML = '<h1 class="screen-title">🗺️ World Map</h1>';

            // Iterate through worlds
            this.worlds.forEach(world => {
                // --- MODIFIED: Always render world, apply locked class later ---
                // Create world element
                const worldEl = document.createElement('div');
                 // Add data attribute for CSS theming
                worldEl.setAttribute('data-world-id', world.id);

                // --- MODIFIED: Determine locked status ---
                const isLocked = !world.unlocked && !this.debugMode;
                worldEl.className = `world ${isLocked ? 'locked' : ''}`; // Apply locked class based on status

                // Filter challenges for this world
                const challengesForWorld = this.challenges.filter(c => c.world === world.id);
                // Count completed challenges in this world
                const completedInWorld = challengesForWorld.filter(c => (this.challenges.indexOf(c) + 1) < this.currentChallenge).length;
                // Determine status text
                const status = (world.unlocked || this.debugMode) ? `${completedInWorld} / ${challengesForWorld.length} Complete` : `Unlocks at Player Level ${world.unlocksAt}`;

                // Generate HTML for level nodes and connecting lines
                const levelsHTML = challengesForWorld.map((challenge, idx) => {
                    const globalIdx = this.challenges.indexOf(challenge) + 1; // Get 1-based index
                    let statusClass = 'locked'; // Default to locked
                    // --- MODIFIED: Apply completed/active ONLY if world is unlocked ---
                    if (!isLocked) {
                        if (globalIdx < this.currentChallenge) {
                            statusClass = 'completed'; // Mark as completed if before current challenge
                        } else if (globalIdx === this.currentChallenge || this.debugMode) {
                            statusClass = 'active'; // Mark as active if it's the current one or debug mode is on
                        }
                    }
                    // Return the HTML for a single level node
                    return `<div class="level-node ${statusClass}" data-challenge="${globalIdx}">${globalIdx}</div>`;
                }).join('<div class="path-line"></div>'); // Join nodes with path lines

                // Set the inner HTML of the world element
                worldEl.innerHTML = `
                    <div class="world-header">
                        <h2 class="world-title">${world.name}</h2>
                        <span class="world-status">${status}</span>
                    </div>
                    <div class="level-path">${levelsHTML}</div>
                `;
                // Add the world element to the container
                container.appendChild(worldEl);
            });
            // Initialize event listeners for the newly created level nodes
            this.initLevelSelectors();
        },
        // --- END CORRECT RENDER WORLD MAP FUNCTION ---


        renderShop() {
            const container = document.getElementById('shop-items');
            if (!container) return;

            container.innerHTML = this.shopItems.map(item => {
                let buttonHtml = '';
                const purchased = this.purchasedItems.includes(item.id) || this.debugMode;

                // --- MODIFIED LOGIC START ---
                // Special case for the Memory Crystal
                if (item.id === 'memory_crystal') {
                    if (!purchased) {
                        buttonHtml = `<button class="buy-button" data-item-id="${item.id}" ${this.computeCredits < item.cost ? 'disabled' : ''}>Buy (${item.cost} C)</button>`;
                    } else {
                        // If purchased, it's just "Active". It cannot be "equipped".
                        buttonHtml = `<button class="equipped-button" disabled>Active</button>`;
                    }
                } else {
                    // Original logic for all other items (models)
                    if (!purchased) {
                        buttonHtml = `<button class="buy-button" data-item-id="${item.id}" ${this.computeCredits < item.cost ? 'disabled' : ''}>Buy (${item.cost} C)</button>`;
                    } else if (this.equippedModel === item.id) {
                        buttonHtml = `<button class="equipped-button" disabled>Equipped</button>`;
                    } else {
                        buttonHtml = `<button class="equip-button" data-item-id="${item.id}">Equip</button>`;
                    }
                }
                // --- MODIFIED LOGIC END ---

                return `
                    <div class="shop-item ${purchased ? 'purchased' : ''}">
                        <div class="shop-icon">${item.icon}</div>
                        <h3>${item.name}</h3>
                        ${item.stats.acc !== 'N/A' ? `<p>Accuracy: ${item.stats.acc} | Speed: ${item.stats.speed}</p>` : `<p>${item.description}</p>`}
                        ${buttonHtml}
                    </div>
                `;
            }).join('');

            this.initShopButtons();
        },

        renderProjectStore() {
            const container = document.getElementById('project-items');
            if (!container) return;

            container.innerHTML = this.projectItems.map(item => {
                 let buttonHtml = '';
                 const purchased = this.purchasedItems.includes(item.id) || this.debugMode;

                 if (!purchased) {
                    buttonHtml = `<button class="buy-button" data-item-id="${item.id}" data-item-type="project" ${this.computeCredits < item.cost ? 'disabled' : ''}>Buy (${item.cost} C)</button>`;
                } else if (this.equippedProject === item.id) {
                    buttonHtml = `<button class="equipped-button" disabled>Equipped</button>`;
                } else {
                    buttonHtml = `<button class="equip-button" data-item-id="${item.id}" data-item-type="project">Equip</button>`;
                }

                return `
                    <div class="project-item ${purchased ? 'purchased' : ''}">
                        <div class="project-icon">${item.icon}</div>
                        <h3>${item.name}</h3>
                        <p>${item.description}</p>
                        ${buttonHtml}
                    </div>
                `;
            }).join('');

            this.initShopButtons();
        },

        initShopButtons() {
            document.querySelectorAll('.buy-button, .equip-button').forEach(button => {
                const newButton = button.cloneNode(true);
                button.parentNode.replaceChild(newButton, button);

                newButton.addEventListener('click', () => {
                    const itemId = newButton.dataset.itemId;
                    const itemType = newButton.dataset.itemType || 'model'; // Default to model
                    const isBuy = newButton.classList.contains('buy-button');

                    if (isBuy) {
                        const collection = itemType === 'project' ? this.projectItems : this.shopItems;
                        const item = collection.find(i => i.id === itemId);
                        if (item && this.computeCredits >= item.cost) {
                            this.computeCredits -= item.cost;
                            // item.purchased = true; // State is managed centrally
                            if (!this.purchasedItems.includes(item.id)) {
                                this.purchasedItems.push(item.id);
                            }
                            // Special case for memory crystal activation
                            if (item.id === 'memory_crystal') {
                                this.memoryCrystalActive = true;
                                this.updateMemoryBar(); // Update bar immediately
                            }

                            this.showNotification(`Purchased ${item.name}!`, 'success');
                            this.updatePlayerStats();
                            // Re-render the relevant store section
                            if (itemType === 'project') this.renderProjectStore(); else this.renderShop();
                            this.saveProgressToServer();
                        } else if (item) {
                            this.showNotification('Not enough Credits!', 'error');
                         } else {
                             console.error("Item not found for purchase:", itemId);
                             this.showNotification('Error: Item not found!', 'error');
                         }
                    } else { // Equip button clicked
                        if (itemType === 'project') {
                            this.equippedProject = itemId;
                            this.renderProjectStore(); // Re-render project store
                        } else {
                            // Ensure it's not the memory crystal being equipped as a model
                            if (itemId !== 'memory_crystal') {
                                this.equippedModel = itemId;
                                this.renderShop(); // Re-render model shop
                                this.updateMemoryBar(); // Update bar on model equip
                            }
                        }

                        // Only show notification if equipping a model/project (not crystal)
                        if (itemId !== 'memory_crystal') {
                            this.showNotification(`Equipped ${itemId}!`, 'success');
                        }
                        this.saveProgressToServer(); // Save the newly equipped item
                    }
                });
            });
        },

        async checkAnswer() {
            const challenge = this.challenges[this.currentChallenge - 1];
            if (!challenge) return;
            const inputField = document.getElementById('code-input');
            if (!inputField) return;
            const userInput = inputField.value.trim();

            this.showNotification('Validating on server...', 'unlock');
            const runButton = document.getElementById('run-button');
             if(runButton) runButton.disabled = true;

            let result;
            try {
                const response = await fetch(`${this.backendUrl}/validate-answer`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ challengeId: this.currentChallenge, userInput: userInput }),
                });
                 // Check for non-OK response first
                 if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ message: `Server error: ${response.statusText}` })); // Provide fallback error
                    throw new Error(errorData.message || `Server error: ${response.status}`);
                 }
                result = await response.json();
            } catch (error) {
                console.error("Validation fetch error:", error);
                this.showNotification(`Validation failed: ${error.message}`, 'error'); // Show specific error
                 if(runButton) runButton.disabled = false;
                return;
            }

            if (result.correct) {
                this.showNotification('Correct!', 'success');
                this.addXp(result.xpReward || 100); // Use default XP if reward not sent
                this.insightPoints += challenge.insightReward || 50; // Use default insight
                this.computeCredits += result.xpReward || 100; // Give credits too
                this.addVis(challenge.vis);

                this.currentChallenge++;
                this.hintUsedThisLevel = false; // Reset hint usage for the next level
                this.saveProgressToServer();

                // Delay before going back to map
                setTimeout(() => {
                    this.updatePlayerStats();
                    this.renderWorldMap(); // Re-render map to update status
                    this.showScreen('world-map-screen');
                    this.showNotification('Challenge Complete!', 'unlock');
                }, 1500);
            } else {
                this.showNotification(result.message || 'Not quite, try again!', 'error'); // Show server message if available
                 if(runButton) runButton.disabled = false;
                 if(inputField) {
                     inputField.style.borderBottom = '2px solid var(--accent-red)';
                     // Optional: Add a shake animation
                     inputField.classList.add('shake');
                     setTimeout(() => {
                         if(inputField) {
                            inputField.style.borderBottom = '';
                            inputField.classList.remove('shake');
                         }
                     }, 1000);
                 }
            }
        },

        addXp(amount) {
            this.xp += amount;
            while (this.xp >= this.xpToNextLevel) { // Use while loop for multiple level ups
                this.xp -= this.xpToNextLevel;
                this.playerLevel++;
                this.xpToNextLevel = Math.floor(this.xpToNextLevel * 1.5); // Increase XP needed
                this.showNotification(`Level Up! Reached Player Level ${this.playerLevel}`, 'levelup');
                this.checkUnlocks(); // Check if new content is unlocked
            }
            this.updatePlayerStats(); // Update UI
            // Avoid saving progress excessively, maybe save only on level up or explicit save action?
            // For now, keep save here for simplicity.
            this.saveProgressToServer();
        },
        checkUnlocks() {
            let updated = false;
            // Check world unlocks
            this.worlds.forEach(world => {
                const shouldBeUnlocked = this.playerLevel >= world.unlocksAt;
                if (shouldBeUnlocked && !world.unlocked) {
                    world.unlocked = true;
                    this.showNotification(`New World Unlocked: ${world.name}!`, 'unlock');
                    updated = true;
                } else if (!shouldBeUnlocked && world.unlocked && world.id !== 1) { // Re-lock if level drops (e.g., debug off)
                    // world.unlocked = false; // Decide if re-locking is desired
                    // updated = true;
                }
            });
            // Check NAS unlock
            const shouldNasBeUnlocked = this.playerLevel >= 6; // NAS unlock level
            if (shouldNasBeUnlocked && !this.nasUnlocked) {
                 this.showNotification('Advanced Feature Unlocked: NAS Lab!', 'levelup');
                 this.nasUnlocked = true;
                 updated = true;
            } else if (!shouldNasBeUnlocked && this.nasUnlocked) {
                 // this.nasUnlocked = false; // Re-lock if level drops
                 // updated = true;
            }

            // If any state changed that affects UI, re-render
             if(updated) {
                 this.renderWorldMap(); // Update world map display (shows locks correctly now)
                 this.renderNAS();      // Update NAS button status/content
                 this.initNavigation(); // Re-init nav to update NAS button lock state
             }
        },
        updatePlayerStats() {
             const insightStat = document.getElementById('insight-stat');
             const creditsStat = document.getElementById('credits-stat');
             const levelStat = document.getElementById('player-level-stat');
             const xpStat = document.getElementById('xp-stat');
             const xpNextStat = document.getElementById('xp-next-stat');
             const xpBarFill = document.getElementById('xp-bar-fill'); // Target the fill element

             if(insightStat) insightStat.textContent = this.insightPoints;
             if(creditsStat) creditsStat.textContent = this.computeCredits;
             if(levelStat) levelStat.textContent = this.playerLevel;
             if(xpStat) xpStat.textContent = this.xp;
             if(xpNextStat) xpNextStat.textContent = this.xpToNextLevel;
             if(xpBarFill) xpBarFill.style.width = `${Math.min(100, (this.xp / this.xpToNextLevel) * 100)}%`; // Ensure width doesn't exceed 100%
        },
        initNavigation() {
            document.querySelectorAll('.nav-button').forEach(button => {
                 // Remove old listeners before adding new ones
                 const newButton = button.cloneNode(true);
                 button.parentNode.replaceChild(newButton, button);

                newButton.addEventListener('click', () => {
                    const screenId = newButton.dataset.screen;

                    // Special check for NAS Lab unlock status
                    if(screenId === 'nas-screen' && !this.nasUnlocked && !this.debugMode) {
                        this.showNotification('NAS Lab is locked! Reach Player Level 6.', 'error');
                        return; // Prevent switching to locked screen
                    }
                    this.showScreen(screenId);
                    // Update active state for buttons
                    document.querySelectorAll('.nav-button').forEach(btn => btn.classList.remove('active'));
                    newButton.classList.add('active');
                });
            });

             // Ensure back button listener is attached only once
             const backButton = document.getElementById('back-to-map-btn');
             if(backButton && !backButton.dataset.listenerAdded) {
                 backButton.addEventListener('click', () => {
                     this.showScreen('world-map-screen');
                     // Optionally, make the World Map nav button active again
                     document.querySelectorAll('.nav-button').forEach(btn => btn.classList.remove('active'));
                     const mapNavButton = document.querySelector('.nav-button[data-screen="world-map-screen"]');
                     if (mapNavButton) mapNavButton.classList.add('active');
                 });
                 backButton.dataset.listenerAdded = 'true'; // Mark as listener added
             }
        },
        initLevelSelectors() {
            document.querySelectorAll('.level-node').forEach(node => {
                // Remove old listeners before adding new ones
                const newNode = node.cloneNode(true);
                node.parentNode.replaceChild(newNode, node);

                // Determine if the level is accessible
                const challengeId = parseInt(newNode.dataset.challenge);
                const isCompleted = newNode.classList.contains('completed');
                const isActive = newNode.classList.contains('active');
                 // A level is clickable if it's active, completed, or if debug mode is on (ignoring visual lock)
                 const canClick = isActive || isCompleted || this.debugMode;


                if (canClick && !isNaN(challengeId)) {
                    newNode.style.cursor = 'pointer'; // Ensure pointer cursor
                    newNode.addEventListener('click', () => {
                         this.loadLevel(challengeId);
                    });
                } else {
                     newNode.style.cursor = 'default'; // Non-clickable cursor
                }
            });
        },
        loadLevel(challengeId) {
            // Ensure challengeId is valid
            if (challengeId < 1 || challengeId > this.challenges.length) {
                console.error("Attempted to load invalid challenge ID:", challengeId);
                this.showNotification("Error: Invalid challenge selected.", "error");
                return;
            }
            // Find the world this challenge belongs to check world unlock status
             const challengeData = this.challenges[challengeId - 1];
             if (!challengeData) {
                  console.error("Challenge data not found for ID:", challengeId); return;
             }
             const worldData = this.worlds.find(w => w.id === challengeData.world);
             const isWorldLocked = worldData && !worldData.unlocked && !this.debugMode;

             // Allow loading only if the world is unlocked AND (it's the current challenge, or completed, or debug mode)
             const canLoad = !isWorldLocked && (challengeId <= this.currentChallenge || this.debugMode);

             if (!canLoad) {
                 // More specific message: is it because the world is locked or the level?
                 if (isWorldLocked) {
                     this.showNotification(`World "${worldData.name}" is locked!`, "error");
                 } else {
                     this.showNotification("Complete previous levels first!", "error");
                 }
                 return;
             }

            // Set the current challenge ONLY if we are moving forward or revisiting the current one
            // This prevents setting currentChallenge back when clicking a completed level
             if (challengeId >= this.currentChallenge) {
                // this.currentChallenge = challengeId; // Decided against this - just load the level requested
             }


            const challenge = challengeData; // Use already fetched data

            this.hintUsedThisLevel = false; // Reset hint status when loading/reloading a level
            const visContainer = document.getElementById('vis-container');
             if(visContainer) visContainer.innerHTML = ''; // Clear previous visualizations

            // Pre-load visualization for completed layers *up to* this challenge
            for (let i = 0; i < challengeId - 1; i++) {
                if (this.challenges[i] && this.challenges[i].vis) {
                     this.addVis(this.challenges[i].vis, true); // true for instant add
                }
            }

             // Get references to UI elements
             const levelTitle = document.getElementById('level-title');
             const instructionsContent = document.getElementById('instructions-content');
             const hintCostEl = document.getElementById('hint-cost');
             const codeBlock = document.getElementById('code-block');
             const runButton = document.getElementById('run-button');

            // Update UI elements with challenge data
             if(levelTitle) levelTitle.textContent = `${challengeId}: ${challenge.title}`;
             if(instructionsContent) {
                 instructionsContent.innerHTML = challenge.instructions; // Set instructions, clears old hints
             }
             if(hintCostEl) hintCostEl.textContent = challenge.hintCost || 50; // Use default cost if missing

             if(codeBlock) {
                 // Replace placeholder [_] with the input field span
                 const codeWithInput = challenge.code.replace('[_]', `<span class="code-input-container"><input type="text" id="code-input" class="code-input-field" autocomplete="off" spellcheck="false" autofocus></span>`);
                 codeBlock.innerHTML = codeWithInput;

                 // Add event listener for Enter key on the input field
                 const inputField = document.getElementById('code-input');
                 if (inputField) {
                     // Remove previous listener if exists
                     inputField.onkeydown = null;
                     inputField.onkeydown = (e) => {
                         if (e.key === 'Enter') {
                             e.preventDefault(); // Prevent default form submission/newline
                             this.checkAnswer();
                         }
                     };
                      // Focus the input field shortly after showing the screen
                     setTimeout(() => {
                         // Check if still on level screen before focusing
                         const currentScreen = document.querySelector('.game-screen.active-screen');
                         if (currentScreen && currentScreen.id === 'level-screen' && document.activeElement !== inputField) {
                            inputField.focus();
                         }
                     }, 100); // Slightly longer delay might help
                 }
             }

             // Enable the run button when loading a level
             if(runButton) runButton.disabled = false;

             this.updateHintButtonState(); // Update hint button availability
             this.showScreen('level-screen'); // Make the level screen visible
        },
        useHint() {
            // Use the actual current challenge ID to find the correct challenge data
            const challengeIndex = this.currentChallenge - 1;
            // Ensure the current challenge index is valid
            if (challengeIndex < 0 || challengeIndex >= this.challenges.length) return;

            const challenge = this.challenges[challengeIndex];
            const hintCost = challenge.hintCost || 50; // Use default cost if not specified

            // Check conditions for using hint
            if (this.hintUsedThisLevel) {
                 this.showNotification('Hint already used for this attempt.', 'error'); // Clarify 'this attempt'
                 return;
             }
             if (this.insightPoints < hintCost) {
                 this.showNotification('Not enough Insight!', 'error');
                return;
            }
             if (!challenge.hint) {
                  console.warn("No hint available for challenge:", this.currentChallenge);
                  this.showNotification("No hint available for this challenge.", "error");
                  // Disable button permanently if no hint exists?
                  this.updateHintButtonState(); // Update state to potentially disable
                  return;
              }


            // Deduct cost, mark hint as used
            this.insightPoints -= hintCost;
            this.hintUsedThisLevel = true; // Mark hint used for this level load/attempt
            this.updatePlayerStats();
            this.updateHintButtonState(); // Disable hint button after use
            this.saveProgressToServer();

            // Display the hint
            const instructions = document.getElementById('instructions-content');
             if(instructions) {
                 // Check if a hint box already exists to prevent duplicates
                 if (!instructions.querySelector('.hint-box')) {
                    const hintBox = document.createElement('div');
                    hintBox.className = "hint-box";
                    hintBox.innerHTML = `<strong style="color: var(--accent-blue);">Hint:</strong> ${challenge.hint}`;
                    instructions.appendChild(hintBox);
                    // Scroll hint into view if needed
                    hintBox.scrollIntoView({ behavior: 'smooth', block: 'end' });
                 }
             }
        },
        updateHintButtonState() {
            const hintButton = document.getElementById('hint-button');
            if (!hintButton) return;
             // Use the actual current challenge ID
            const challengeIndex = this.currentChallenge - 1;
             // Ensure the current challenge index is valid
             if (challengeIndex < 0 || challengeIndex >= this.challenges.length) {
                 hintButton.disabled = true; // Disable if challenge is invalid
                 return;
             }
            const challenge = this.challenges[challengeIndex];
            const hintCost = challenge.hintCost || 50;
            // Disable if hint used, not enough insight, or no hint exists for the current challenge
            hintButton.disabled = this.hintUsedThisLevel || this.insightPoints < hintCost || !challenge.hint;
        },
        addVis(visData, isInstant = false) {
            const visContainer = document.getElementById('vis-container');
            if (!visData || !visContainer) return;

            const items = Array.isArray(visData) ? visData : [visData];

            items.forEach((item, index) => {
                if (!item || !item.type || !item.label) {
                     console.warn("Skipping invalid vis item:", item);
                     return; // Skip invalid items
                 }

                const visEl = document.createElement('div');
                visEl.className = 'layer-vis';
                // Define default style
                let style = { bgColor: 'var(--panel)', height: '30px', textColor: 'var(--text-muted)' };

                // Apply styles based on layer type
                switch(item.type) {
                    case 'conv': style = { bgColor: 'var(--accent-blue)', height: item.size ? `${item.size}px` : '40px', textColor: 'white' }; break;
                    case 'relu': style = { bgColor: 'var(--accent-green)', height: '20px', textColor: 'var(--bg-dark)' }; break;
                    case 'pool': style = { bgColor: 'purple', textColor: 'white', height: item.size ? `${item.size}px` : '35px' }; break;
                    case 'flat': style = { bgColor: '#666', textColor: 'white', height: item.size ? `${item.size}px` : '30px' }; break;
                    case 'linear': style = { bgColor: 'var(--accent-yellow)', textColor: 'var(--bg-dark)', height: item.size ? `${item.size}px` : '30px' }; break;
                    case 'dropout': style = { bgColor: 'var(--accent-red)', textColor: 'white', height: '20px' }; break;
                    case 'data': style = { bgColor: 'teal', textColor: 'white' }; break; // Example for data steps
                    case 'optim': style = { bgColor: 'orange', textColor: 'var(--bg-dark)' }; break; // Example for optimizer steps
                    default: console.warn("Unknown vis type:", item.type); // Log unknown types
                }
                // Apply styles
                visEl.style.backgroundColor = style.bgColor;
                visEl.style.height = style.height; // Use defined height
                visEl.style.color = style.textColor;
                visEl.textContent = item.label;
                visEl.style.fontWeight = 'bold';
                visEl.style.textAlign = 'center';
                visEl.style.borderRadius = '5px';
                visEl.style.padding = '5px';
                visEl.style.marginBottom = '5px';
                visEl.style.width = '90%'; // Use percentage width for responsiveness
                visEl.style.display = 'flex'; // Use flexbox for vertical centering
                visEl.style.alignItems = 'center';
                visEl.style.justifyContent = 'center';
                visEl.style.overflow = 'hidden'; // Prevent text overflow issues
                visEl.style.textOverflow = 'ellipsis';
                visEl.style.whiteSpace = 'nowrap';


                // Add fade-in animation
                visEl.style.opacity = '0'; // Start invisible
                visContainer.appendChild(visEl); // Add to DOM first
                // Trigger fade-in with a slight delay
                setTimeout(() => {
                    visEl.style.transition = 'opacity 0.5s ease';
                    visEl.style.opacity = '1';
                }, isInstant ? 0 : 50 + index * 50); // Stagger animation slightly unless instant
            });
        },
        showScreen(screenId) {
            document.querySelectorAll('.game-screen').forEach(s => s.classList.remove('active-screen'));
            const activeScreen = document.getElementById(screenId);
            if (activeScreen) {
                activeScreen.classList.add('active-screen');
                // Scroll to top when switching screens
                activeScreen.scrollTop = 0;
            } else {
                 console.error("Screen not found:", screenId);
             }
        },
        showNotification(message, type = 'success') {
            const notif = document.getElementById('notification');
            if (!notif) return;
            notif.textContent = message;
            // Set background color based on type
            let bgColor;
            switch(type) {
                case 'error': bgColor = 'var(--accent-red)'; break;
                case 'success': bgColor = 'var(--accent-green)'; break;
                case 'unlock': case 'levelup': bgColor = 'var(--accent-blue)'; break;
                default: bgColor = 'var(--panel)'; // Default color
            }
            notif.style.backgroundColor = bgColor;

            // Show notification
            notif.classList.add('show');

            // Clear previous timeout if exists
            if (this.notificationTimeout) {
                clearTimeout(this.notificationTimeout);
            }

            // Hide after 3 seconds
            this.notificationTimeout = setTimeout(() => {
                    notif.classList.remove('show');
                    this.notificationTimeout = null; // Clear timeout reference
            }, 3000);
        },

        updateMemoryBar() {
            const modelId = this.equippedModel || 'default';
            let cost = this.memoryCosts[modelId] || 10; // Use default cost if model not found
            // --- Use memoryCrystalActive state variable ---
            const crystalActive = this.memoryCrystalActive;

            const fillBar = document.getElementById('memory-bar-fill');
            const fillText = document.getElementById('memory-bar-text');
            const crystalSlot = document.getElementById('crystal-slot');

            if (!fillBar || !fillText || !crystalSlot) return;

            let capacity = 100; // Base capacity
            if (crystalActive) {
                capacity = 1000;
                crystalSlot.textContent = '💎 Memory Crystal: ACTIVE';
                crystalSlot.className = 'slot-active';
            } else {
                crystalSlot.textContent = '💎 Memory Crystal: EMPTY';
                crystalSlot.className = 'slot-empty';
            }

            let fillPercent = (cost / capacity) * 100;
            if (cost > 100 && !crystalActive) {
                fillPercent = 100;
                fillBar.style.background = 'var(--accent-red)';
            } else {
                 fillBar.style.background = 'linear-gradient(90deg, var(--accent-green), var(--accent-yellow), var(--accent-red))';
            }
             fillPercent = Math.max(0, Math.min(100, fillPercent));

            fillBar.style.width = `${fillPercent}%`;
            fillText.textContent = `${cost}% Used / ${capacity}% Capacity`;
        },

        // --- NAS FUNCTIONS ---
        renderNAS() {
            const container = document.getElementById('nas-screen');
            if (!container) return;

            const nasNavButton = document.querySelector('.nav-button[data-screen="nas-screen"]');
            const isNasLocked = !this.nasUnlocked && !this.debugMode;
            if (nasNavButton) {
                 nasNavButton.classList.toggle('locked', isNasLocked);
            }

            if (!isNasLocked || this.debugMode) {
                this.initNASListeners();
                this.updateNASSlidersFromState();
                this.updateNASCost();
            }
        },
        updateNASSlidersFromState() {
             const accuracySlider = document.getElementById('nas-accuracy');
             const speedSlider = document.getElementById('nas-speed');
             const efficiencySlider = document.getElementById('nas-efficiency');
             const accVal = document.getElementById('nas-accuracy-val');
             const speedVal = document.getElementById('nas-speed-val');
             const effVal = document.getElementById('nas-efficiency-val');

             if (accuracySlider) accuracySlider.value = this.nasPreferences.accuracy;
             if (speedSlider) speedSlider.value = this.nasPreferences.speed;
             if (efficiencySlider) efficiencySlider.value = this.nasPreferences.efficiency;
             if (accVal) accVal.textContent = this.nasPreferences.accuracy;
             if (speedVal) speedVal.textContent = this.nasPreferences.speed;
             if (effVal) effVal.textContent = this.nasPreferences.efficiency;
         },

        initNASListeners() {
            const nasButton = document.getElementById('start-nas-button');
            if (nasButton && !nasButton.dataset.listenerAdded) {
                nasButton.addEventListener('click', () => this.startNAS());
                nasButton.dataset.listenerAdded = 'true';
            }

            document.querySelectorAll('.nas-slider').forEach(slider => {
                const newSlider = slider.cloneNode(true);
                slider.parentNode.replaceChild(newSlider, slider);
                newSlider.addEventListener('input', (e) => {
                    const id = e.target.id;
                    const value = e.target.value;
                    const valDisplay = document.getElementById(`${id}-val`);
                    if (valDisplay) valDisplay.textContent = `${value}`;
                    const intValue = parseInt(value);
                    if (id.includes('accuracy')) this.nasPreferences.accuracy = intValue;
                    else if (id.includes('speed')) this.nasPreferences.speed = intValue;
                    else if (id.includes('efficiency')) this.nasPreferences.efficiency = intValue;
                    this.updateNASCost();
                });
            });
        },

        updateNASCost() {
            const baseCost = 5000;
            const accuracyCost = this.nasPreferences.accuracy * 100;
            const speedCost = this.nasPreferences.speed * 50;
            const efficiencyCost = this.nasPreferences.efficiency * 50;
            const totalCost = baseCost + accuracyCost + speedCost + efficiencyCost;
            const costDisplay = document.getElementById('nas-cost-display');
            if(costDisplay) costDisplay.textContent = `${totalCost} CC`;
        },

        startNAS() {
            let cost = 5000; // Default cost
            const baseCost = 5000;
            const accuracyCost = this.nasPreferences.accuracy * 100;
            const speedCost = this.nasPreferences.speed * 50;
            const efficiencyCost = this.nasPreferences.efficiency * 50;
            cost = baseCost + accuracyCost + speedCost + efficiencyCost;

            if (this.computeCredits < cost) {
                this.showNotification('Not enough Compute Credits!', 'error');
                return;
            }

            this.computeCredits -= cost;
            this.updatePlayerStats();
            this.saveProgressToServer();

            const nasButton = document.getElementById('start-nas-button');
            const nasStatus = document.getElementById('nas-status');
            const progressFill = document.getElementById('nas-progress-fill');
            const resultsDisplay = document.getElementById('nas-results-display');

            if (nasButton) nasButton.disabled = true;
            if (nasStatus) nasStatus.textContent = 'Initializing evolution...';
            if (progressFill) progressFill.style.width = '0%';
            if (resultsDisplay) resultsDisplay.innerHTML = '';

            let progress = 0;
            const maxSimulatedProgress = 95;
            const progressIntervalTime = 500;
            const progressInterval = setInterval(() => {
                progress += 5;
                if (progressFill) progressFill.style.width = `${Math.min(progress, maxSimulatedProgress)}%`;
                if (nasStatus) nasStatus.textContent = `Evolving generation... ${Math.min(progress, maxSimulatedProgress)}%`;
                if (progress >= maxSimulatedProgress) {
                    clearInterval(progressInterval);
                }
            }, progressIntervalTime);

            this.showNotification('Starting Neural Architecture Search...', 'unlock');

            fetch(`${this.backendUrl}/nas`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    preferences: this.nasPreferences,
                    potions: this.activePotions
                })
            })
            .then(response => {
                 if (!response.ok) {
                     return response.json().then(err => { throw new Error(err.message || `Server error: ${response.statusText}`); });
                 }
                 return response.json();
            })
            .then(data => {
                clearInterval(progressInterval);
                if (progressFill) progressFill.style.width = '100%';

                if (data.results && data.results.length > 0) {
                    if (nasStatus) nasStatus.textContent = `Evolution complete! Found ${data.total_evaluated || 0} valid architectures.`;
                    this.showNotification('NAS complete! Best architectures found.', 'success');
                    if (resultsDisplay) {
                        resultsDisplay.innerHTML = data.results.map((res, i) => `
                            <div class="nas-result-item">
                                <h3>Rank #${i + 1} (Fitness: ${res.fitness?.toFixed(2) || 'N/A'})</h3>
                                <div class="nas-result-stats">
                                    <div><strong>Accuracy:</strong> ${res.accuracy?.toFixed(2) || 'N/A'}%</div>
                                    <div><strong>Speed:</strong> ${res.inference_time?.toFixed(1) || 'N/A'} ms</div>
                                    <div><strong>Size:</strong> ${res.model_size?.toFixed(2) || 'N/A'} MB</div>
                                </div>
                                <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 10px;">${res.gene_summary || 'No summary'}</p>
                            </div>
                        `).join('');
                         resultsDisplay.parentElement.scrollTop = 0;
                    }
                } else {
                    if (nasStatus) nasStatus.textContent = 'Search failed or found no valid models.';
                    this.showNotification(data.message || 'NAS search failed to find a valid model.', 'error');
                }
                if (nasButton) nasButton.disabled = false;
            })
            .catch(error => {
                console.error("NAS Fetch Error:", error);
                clearInterval(progressInterval);
                 if (progressFill) progressFill.style.width = '0%';
                if (nasStatus) nasStatus.textContent = `Error: ${error.message}`;
                if (nasButton) nasButton.disabled = false;
                this.showNotification(`NAS search failed! ${error.message}`, 'error');
                this.computeCredits += cost; // Refund
                this.updatePlayerStats();
                this.saveProgressToServer();
            });
        },

       
          // --- Debug Mode Function (Corrected Memory Crystal & Item Purchase Logic) ---
        initDebugMode() {
            const debugToggle = document.getElementById('debug-toggle');
            if (!debugToggle || debugToggle.dataset.listenerAttached) return;

            debugToggle.addEventListener('click', () => {
                this.debugMode = !this.debugMode;
                debugToggle.textContent = `Debug Mode: ${this.debugMode ? 'ON' : 'OFF'}`;
                debugToggle.style.backgroundColor = this.debugMode ? 'var(--accent-green)' : 'var(--accent-red)';

                if (this.debugMode) {
                    // --- ON Logic ---
                    this.computeCredits = 50000;
                    this.insightPoints = 1000;
                    this.playerLevel = 10;
                    this.xp = 0;

                    // --- MODIFIED: Explicitly add ALL items to purchasedItems ---
                    this.shopItems.forEach(item => {
                        if (!this.purchasedItems.includes(item.id)) {
                            this.purchasedItems.push(item.id);
                        }
                    });
                    this.projectItems.forEach(item => {
                        if (!this.purchasedItems.includes(item.id)) {
                            this.purchasedItems.push(item.id);
                        }
                    });
                    // --- END MODIFICATION ---

                    // Ensure memory crystal active flag is set
                    this.memoryCrystalActive = this.purchasedItems.includes('memory_crystal');

                    this.updatePlayerStats();
                    this.checkUnlocks(); // Unlock based on level 10
                    this.showNotification('Debug Mode ON: All features unlocked!', 'levelup');
                    // Re-render relevant screens
                    this.renderWorldMap();
                    this.renderShop(); // Should show all items as purchased/active
                    this.renderProjectStore(); // Should show all projects as purchased
                    this.renderNAS(); // Should show NAS unlocked
                    this.updateMemoryBar(); // Update bar *after* setting crystal active
                    this.initNavigation(); // Re-init nav buttons
                } else {
                    // --- OFF Logic ---
                    this.showNotification('Debug Mode OFF: Normal progression restored', 'success');
                    // Fetch saved state from server to restore accurately
                    fetch(`${this.backendUrl}/get-progress`)
                        .then(response => response.ok ? response.json() : Promise.reject('Fetch failed'))
                        .then(data => {
                             // Reset core state before applying fetched data
                            Object.assign(this, {
                                playerLevel: 1, xp: 0, insightPoints: 100, computeCredits: 1000,
                                currentChallenge: 1, equippedModel: null, equippedProject: null,
                                uploadedDatasets: [], purchasedItems: [], memoryCrystalActive: false,
                                nasUnlocked: false, debugMode: false // Ensure debug is off
                            });
                             // Manually reset world locks based on level 1
                            this.worlds.forEach(w => w.unlocked = (w.id === 1));

                            // Apply fetched data
                            Object.assign(this, data);
                            this.purchasedItems = this.purchasedItems || [];
                            this.uploadedDatasets = this.uploadedDatasets || [];
                            // Sync visual state
                            this.shopItems.forEach(item => item.purchased = this.purchasedItems.includes(item.id));
                            this.projectItems.forEach(item => item.purchased = this.purchasedItems.includes(item.id));
                            this.memoryCrystalActive = this.purchasedItems.includes('memory_crystal');
                            // Re-check unlocks based on fetched level
                            this.checkUnlocks();
                            // Re-render everything
                            this.renderAll();
                        })
                        .catch(error => {
                            console.error('Error fetching progress after debug off:', error);
                            this.showNotification('Error restoring progress! Resetting.', 'error');
                            // Fallback reset
                            Object.assign(this, { playerLevel: 1, xp: 0, insightPoints: 100, computeCredits: 1000, currentChallenge: 1, equippedModel: null, equippedProject: null, uploadedDatasets: [], purchasedItems: [], memoryCrystalActive: false, nasUnlocked: false, debugMode: false });
                            this.worlds.forEach(w => w.unlocked = (w.id === 1));
                            this.renderAll();
                        });
                }
            });
             debugToggle.dataset.listenerAttached = 'true';
        },


        // --- REVERTED: MNIST LAB FUNCTIONS (BACK TO DRAWING CANVAS) ---
        initMNISTLab() {
            this.mnistCanvas = document.getElementById('mnist-canvas'); // Target canvas now
            const predictBtn = document.getElementById('predict-digit-btn');
            const clearBtn = document.getElementById('clear-canvas-btn'); // Target canvas clear button

            if (!this.mnistCanvas || !predictBtn || !clearBtn) {
                 console.warn("MNIST Lab canvas elements not found. Skipping initialization.");
                return;
            }
             if (this.mnistCanvas.dataset.listenerAttached) return;

            this.mnistCtx = this.mnistCanvas.getContext('2d');
            this.clearMNISTCanvas();

            this.mnistCtx.strokeStyle = "white";
            this.mnistCtx.lineWidth = 12; // Thinner lines
            this.mnistCtx.lineCap = 'round';
            this.mnistCtx.lineJoin = 'round';

            const startDrawing = (e) => {
                this.updateMNISTResults('?', 'Draw a digit (0-9)');
                this.isDrawing = true;
                this.mnistCtx.beginPath();
                const pos = this.getMousePos(this.mnistCanvas, e);
                this.mnistCtx.moveTo(pos.x, pos.y);
            };
            const draw = (e) => {
                if (!this.isDrawing) return;
                const pos = this.getMousePos(this.mnistCanvas, e);
                this.mnistCtx.lineTo(pos.x, pos.y);
                this.mnistCtx.stroke();
            };
            const stopDrawing = () => {
                if(this.isDrawing) {
                    this.mnistCtx.closePath();
                    this.isDrawing = false;
                }
            };

            this.mnistCanvas.addEventListener('mousedown', startDrawing);
            this.mnistCanvas.addEventListener('mousemove', draw);
            this.mnistCanvas.addEventListener('mouseup', stopDrawing);
            this.mnistCanvas.addEventListener('mouseout', stopDrawing);

            this.mnistCanvas.addEventListener('touchstart', (e) => {
                e.preventDefault();
                if (e.touches.length > 0) startDrawing(e.touches[0]);
            }, { passive: false });
            this.mnistCanvas.addEventListener('touchmove', (e) => {
                e.preventDefault();
                 if (e.touches.length > 0) draw(e.touches[0]);
            }, { passive: false });
            this.mnistCanvas.addEventListener('touchend', stopDrawing);
            this.mnistCanvas.addEventListener('touchcancel', stopDrawing);

            clearBtn.addEventListener('click', () => this.clearMNISTCanvas());
            predictBtn.addEventListener('click', () => this.predictDigit());

             this.mnistCanvas.dataset.listenerAttached = 'true';
        },

        getMousePos(canvas, evt) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const clientX = evt.clientX ?? evt.touches?.[0]?.clientX ?? 0;
            const clientY = evt.clientY ?? evt.touches?.[0]?.clientY ?? 0;
            return {
                x: (clientX - rect.left) * scaleX,
                y: (clientY - rect.top) * scaleY
            };
        },

        clearMNISTCanvas() {
            if (!this.mnistCtx || !this.mnistCanvas) return;
            this.mnistCtx.fillStyle = "black";
            this.mnistCtx.fillRect(0, 0, this.mnistCanvas.width, this.mnistCanvas.height);
            this.updateMNISTResults('?', 'Draw a digit (0-9)');
        },

        updateMNISTResults(digitText, confidenceText) {
            const resultEl = document.getElementById('mnist-result-display'); // Correct ID
            const confidenceEl = document.getElementById('mnist-confidence');
            if(resultEl) resultEl.textContent = digitText;
            if(confidenceEl) confidenceEl.textContent = confidenceText;
        },

        async predictDigit() {
            if (!this.mnistCanvas) return;

             const blankCanvas = document.createElement('canvas');
             blankCanvas.width = this.mnistCanvas.width;
             blankCanvas.height = this.mnistCanvas.height;
             const blankDataURL = blankCanvas.toDataURL('image/png');
             const currentDataURL = this.mnistCanvas.toDataURL('image/png');

             if (currentDataURL === blankDataURL) {
                 this.showNotification("Canvas is empty. Draw a digit first!", "error");
                 this.updateMNISTResults('?', 'Draw a digit (0-9)');
                 return;
             }

            this.updateMNISTResults('...', 'Analyzing...');

            try {
                const response = await fetch(`${this.backendUrl}/predict-digit`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ imageData: currentDataURL }),
                });

                if (!response.ok) {
                    const err = await response.json().catch(() => ({ error: `Server error: ${response.statusText}` }));
                    throw new Error(err.error || `Server error: ${response.statusText}`);
                }
                const result = await response.json();
                if (result.error) {
                    throw new Error(result.error);
                }
                this.updateMNISTResults(result.predicted_digit, `Confidence: ${result.confidence}%`);

            } catch (error) {
                console.error('Error predicting digit:', error);
                this.updateMNISTResults('X', 'Prediction failed.');
                this.showNotification(`Prediction Error: ${error.message}`, 'error');
            }
        }
        // --- END REVERTED MNIST LAB FUNCTIONS ---

    }; // End of game object definition

    // --- Initialize the game ---
    game.init();

}); // End DOMContentLoaded