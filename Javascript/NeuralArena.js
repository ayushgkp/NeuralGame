document.addEventListener('DOMContentLoaded', () => {
    const game = {
        backendUrl: 'http://127.0.0.1:5000',

        // --- PLAYER STATE (will be overwritten by backend) ---
        playerLevel: 1, xp: 0, xpToNextLevel: 100, insightPoints: 100,
        computeCredits: 1000, hintUsedThisLevel: false, equippedModel: null,
        equippedProject: null, uploadedDatasets: [], purchasedItems: [],
        currentChallenge: 1,
        
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
            { id: 'inception', name: 'InceptionV3', icon: '🌀', cost: 3500, purchased: false, stats: { acc: '94%', speed: 'Medium' } }
        ],
        projectItems: [
            { id: 'img_classification', name: 'Image Classifier', icon: '📸', cost: 5000, purchased: false, description: 'A complete blueprint for a state-of-the-art image classification project.' },
            { id: 'object_detection', name: 'Object Detector', icon: '🎯', cost: 8000, purchased: false, description: 'Learn to build a project that can find and identify objects in an image.' }
        ],
        
        init() {
            const self = this; // Store the context of the 'game' object

            fetch(`${this.backendUrl}/get-progress`)
                .then(response => response.json())
                .then(data => {
                    Object.assign(self, data);
                    self.shopItems.forEach(item => item.purchased = self.purchasedItems.includes(item.id));
                    self.projectItems.forEach(item => item.purchased = self.purchasedItems.includes(item.id));
                    self.renderAll();
                })
                .catch(error => {
                    console.error('Error loading data:', error);
                    self.showNotification('Error: Could not connect to backend!', 'error');
                    self.renderAll(); 
                });
            
            this.initEventListeners();
        },
        
        renderAll() {
            this.renderWorldMap(); this.renderShop(); this.renderProjectStore();
            this.renderDataGarden(); this.updatePlayerStats(); this.initNavigation();
        },

        initEventListeners() {
            document.getElementById('run-button').addEventListener('click', () => this.checkAnswer());
            document.getElementById('hint-button').addEventListener('click', () => this.useHint());
        },

        saveProgressToServer() {
            const dataToSave = {
                playerLevel: this.playerLevel, xp: this.xp, insightPoints: this.insightPoints,
                computeCredits: this.computeCredits, currentChallenge: this.currentChallenge,
                equippedModel: this.equippedModel, equippedProject: this.equippedProject,
                uploadedDatasets: this.uploadedDatasets, purchasedItems: this.purchasedItems,
            };
            fetch(`${this.backendUrl}/save-progress`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(dataToSave),
            })
            .then(res => res.json())
            .then(data => console.log('Progress saved:', data.data))
            .catch(error => console.error('Error saving progress:', error));
        },

        // --- NEWLY ADDED FUNCTIONS ---
        renderWorldMap() {
            const container = document.getElementById('world-map-screen');
            if (!container) return;
            
            container.innerHTML = '<h1 class="screen-title">🗺️ World Map</h1>';
            
            this.worlds.forEach(world => {
                const worldEl = document.createElement('div');
                worldEl.className = `world ${world.unlocked ? '' : 'locked'}`;
                
                const challengesForWorld = this.challenges.filter(c => c.world === world.id);
                const completedInWorld = challengesForWorld.filter(c => (this.challenges.indexOf(c) + 1) < this.currentChallenge).length;
                const status = world.unlocked ? `${completedInWorld} / ${challengesForWorld.length} Complete` : `Unlocks at Player Level ${world.unlocksAt}`;

                const levelsHTML = challengesForWorld.map((challenge, idx) => {
                    const globalIdx = this.challenges.indexOf(challenge) + 1;
                    let status = 'locked';
                    if (globalIdx < this.currentChallenge) status = 'completed';
                    else if (globalIdx === this.currentChallenge && world.unlocked) status = 'active';
                    
                    return `<div class="level-node ${status}" data-challenge="${globalIdx}">${globalIdx}</div>`;
                }).join('<div class="path-line"></div>');
                
                worldEl.innerHTML = `
                    <div class="world-header">
                        <h2 class="world-title">${world.name}</h2>
                        <span class="world-status">${status}</span>
                    </div>
                    <div class="level-path">${levelsHTML}</div>
                `;
                
                container.appendChild(worldEl);
            });
            
            this.initLevelSelectors();
        },

        renderShop() {
            const container = document.getElementById('shop-items');
            if (!container) return;
            
            container.innerHTML = this.shopItems.map(item => {
                let buttonHtml = '';
                if (!item.purchased) {
                    buttonHtml = `<button class="buy-button" data-item-id="${item.id}" ${this.computeCredits < item.cost ? 'disabled' : ''}>Buy</button>`;
                } else if (this.equippedModel === item.id) {
                    buttonHtml = `<button class="equipped-button" disabled>Equipped</button>`;
                } else {
                    buttonHtml = `<button class="equip-button" data-item-id="${item.id}">Equip</button>`;
                }

                return `
                    <div class="shop-item ${item.purchased ? 'purchased' : ''}">
                        <div class="shop-icon">${item.icon}</div>
                        <h3>${item.name}</h3>
                        <p>Accuracy: ${item.stats.acc} | Speed: ${item.stats.speed}</p>
                        <p><strong>${item.cost} Credits</strong></p>
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
                 if (!item.purchased) {
                    buttonHtml = `<button class="buy-button" data-item-id="${item.id}" data-item-type="project" ${this.computeCredits < item.cost ? 'disabled' : ''}>Buy</button>`;
                } else if (this.equippedProject === item.id) {
                    buttonHtml = `<button class="equipped-button">Equipped</button>`;
                } else {
                    buttonHtml = `<button class="equip-button" data-item-id="${item.id}" data-item-type="project">Equip</button>`;
                }

                return `
                    <div class="project-item ${item.purchased ? 'purchased' : ''}">
                        <div class="project-icon">${item.icon}</div>
                        <h3>${item.name}</h3>
                        <p>${item.description}</p>
                        <p><strong>${item.cost} Credits</strong></p>
                        ${buttonHtml}
                    </div>
                `;
            }).join('');
            
            this.initShopButtons();
        },
        // --- END NEWLY ADDED FUNCTIONS ---

        renderDataGarden() {
            const container = document.getElementById('dataset-plots');
            container.innerHTML = ''; 

            const allDatasets = [...this.datasets];
            (this.uploadedDatasets || []).forEach(filename => {
                if (!allDatasets.some(ds => ds.name === filename)) {
                    allDatasets.push({
                        id: filename.toLowerCase().replace(/[^a-z0-9]/g, ''),
                        name: filename,
                        icon: '📁',
                        description: 'User uploaded dataset.'
                    });
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
                container.appendChild(plotEl);
            });

            const uploadPlot = document.createElement('div');
            uploadPlot.className = 'plot upload-plot';
            uploadPlot.innerHTML = `
                <div class="plot-icon">📤</div>
                <h3>Upload Dataset</h3>
                <p>Select a file to upload.</p>
                <input type="file" id="dataset-upload-input" accept=".zip,.rar,.7z,.png,.jpg,.jpeg" style="display: none;">
            `;
            uploadPlot.addEventListener('click', () => document.getElementById('dataset-upload-input').click());
            container.appendChild(uploadPlot);
            
            this.initDataGardenUpload();
            this.initTrainButtons();
        },
        
        initDataGardenUpload() {
            const fileInput = document.getElementById('dataset-upload-input');
            if (fileInput.dataset.listenerAdded) return;
            
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
                        .catch(error => this.showNotification('Upload failed! Could not reach server.', 'error'));
                }
            });
            fileInput.dataset.listenerAdded = 'true';
        },

        initTrainButtons() {
            document.querySelectorAll('.train-button').forEach(button => {
                const newButton = button.cloneNode(true);
                button.parentNode.replaceChild(newButton, button);

                newButton.addEventListener('click', () => {
                    const datasetName = newButton.dataset.datasetName;
                    if (!this.equippedModel) {
                        this.showNotification('You must equip a model from the Store first!', 'error');
                        return;
                    }
                    
                    this.showNotification(`Training ${this.equippedModel} on ${datasetName}... This may take a moment.`, 'unlock');
                    
                    fetch(`${this.backendUrl}/train-model`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ modelName: this.equippedModel, datasetName: datasetName })
                    })
                    .then(response => response.json())
                    .then(result => {
                        if (result.status === 'success') {
                            this.showNotification(`Training Complete! Final Accuracy: ${result.accuracy}`, 'success');
                        } else {
                            this.showNotification(`Training Failed: ${result.message}`, 'error');
                        }
                    })
                    .catch(error => {
                        console.error('Training Error:', error);
                        this.showNotification('Training failed! Could not connect to the server.', 'error');
                    });
                });
            });
        },

        initShopButtons() {
            document.querySelectorAll('.buy-button, .equip-button').forEach(button => {
                const newButton = button.cloneNode(true);
                button.parentNode.replaceChild(newButton, button);
                
                newButton.addEventListener('click', () => {
                    const itemId = newButton.dataset.itemId;
                    const itemType = newButton.dataset.itemType || 'model';
                    const isBuy = newButton.classList.contains('buy-button');

                    if (isBuy) {
                        const collection = itemType === 'project' ? this.projectItems : this.shopItems;
                        const item = collection.find(i => i.id === itemId);
                        if (item && this.computeCredits >= item.cost) {
                            this.computeCredits -= item.cost;
                            item.purchased = true;
                            if (!this.purchasedItems.includes(item.id)) {
                                this.purchasedItems.push(item.id);
                            }
                            this.showNotification(`Purchased ${item.name}!`, 'success');
                            this.updatePlayerStats();
                            if (itemType === 'project') this.renderProjectStore(); else this.renderShop();
                            this.saveProgressToServer();
                        } else { this.showNotification('Not enough Credits!', 'error'); }
                    } else { // Equip button
                        if (itemType === 'project') {
                            this.equippedProject = itemId;
                            this.renderProjectStore();
                        } else {
                            this.equippedModel = itemId;
                            this.renderShop();
                        }
                        this.showNotification(`Equipped ${itemId}!`, 'success');
                        this.saveProgressToServer(); 
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
            document.getElementById('run-button').disabled = true;

            let result;
            try {
                const response = await fetch(`${this.backendUrl}/validate-answer`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ challengeId: this.currentChallenge, userInput: userInput }),
                });
                result = await response.json();
            } catch (error) {
                this.showNotification('Could not connect to server!', 'error');
                document.getElementById('run-button').disabled = false;
                return;
            }

            if (result.correct) {
                this.showNotification('Correct!', 'success');
                this.addXp(result.xpReward); 
                this.insightPoints += challenge.insightReward;
                this.computeCredits += result.xpReward;
                this.addVis(challenge.vis);
                
                this.currentChallenge++; 
                this.saveProgressToServer();

                setTimeout(() => {
                    this.updatePlayerStats();
                    this.renderWorldMap();
                    this.showScreen('world-map-screen');
                    this.showNotification('Challenge Complete!', 'unlock');
                }, 1500);
            } else {
                this.showNotification('Not quite, try again!', 'error');
                document.getElementById('run-button').disabled = false;
                inputField.style.borderBottom = '2px solid var(--accent-red)';
                setTimeout(() => { inputField.style.borderBottom = ''; }, 1000);
            }
        },
        
        addXp(amount) {
            this.xp += amount;
            if (this.xp >= this.xpToNextLevel) {
                this.xp -= this.xpToNextLevel;
                this.playerLevel++;
                this.xpToNextLevel = Math.floor(this.xpToNextLevel * 1.5);
                this.showNotification(`Level Up! Reached Player Level ${this.playerLevel}`, 'levelup');
                this.checkUnlocks();
                this.saveProgressToServer();
            }
            this.updatePlayerStats();
        },
        checkUnlocks() {
            this.worlds.forEach(world => {
                if (!world.unlocked && this.playerLevel >= world.unlocksAt) {
                    world.unlocked = true;
                    this.showNotification(`New World Unlocked: ${world.name}!`, 'unlock');
                    this.renderWorldMap();
                }
            });
        },
        updatePlayerStats() {
            document.getElementById('insight-stat').textContent = this.insightPoints;
            document.getElementById('credits-stat').textContent = this.computeCredits;
            document.getElementById('player-level-stat').textContent = this.playerLevel;
            document.getElementById('xp-stat').textContent = this.xp;
            document.getElementById('xp-next-stat').textContent = this.xpToNextLevel;
            document.getElementById('xp-bar').style.width = `${(this.xp / this.xpToNextLevel) * 100}%`;
        },
        initNavigation() {
            document.querySelectorAll('.nav-button').forEach(button => {
                button.addEventListener('click', () => {
                    const screenId = button.dataset.screen;
                    this.showScreen(screenId);
                    document.querySelectorAll('.nav-button').forEach(btn => btn.classList.remove('active'));
                    button.classList.add('active');
                });
            });
            document.getElementById('back-to-map-btn').addEventListener('click', () => this.showScreen('world-map-screen'));
        },
        initLevelSelectors() {
            document.querySelectorAll('.level-node.active').forEach(node => {
                const newNode = node.cloneNode(true);
                node.parentNode.replaceChild(newNode, node);
                newNode.addEventListener('click', () => this.loadLevel(parseInt(newNode.dataset.challenge)));
            });
        },
        loadLevel(challengeId) {
            const challenge = this.challenges[challengeId - 1];
            if (!challenge) return;
            this.hintUsedThisLevel = false;
            const visContainer = document.getElementById('vis-container');
            visContainer.innerHTML = '';
            for (let i = 0; i < challengeId - 1; i++) {
                if (this.challenges[i].vis) this.addVis(this.challenges[i].vis, true); 
            }
            document.getElementById('level-title').textContent = `${challengeId}: ${challenge.title}`;
            document.getElementById('instructions-content').innerHTML = challenge.instructions;
            document.getElementById('hint-cost').textContent = challenge.hintCost;
            this.updateHintButtonState();
            
            const codeWithInput = challenge.code.replace('[_]', `<span class="code-input-container"><input type="text" id="code-input" class="code-input-field" autocomplete="off" spellcheck="false" autofocus></span>`);
            document.getElementById('code-block').innerHTML = codeWithInput;

            document.getElementById('code-input').addEventListener('keydown', (e) => {
                if (e.key === 'Enter') { e.preventDefault(); this.checkAnswer(); }
            });
            this.showScreen('level-screen');
            setTimeout(() => { const input = document.getElementById('code-input'); if (input) input.focus(); }, 50);
        },
        useHint() {
            const challenge = this.challenges[this.currentChallenge - 1];
            if (!challenge || this.hintUsedThisLevel || this.insightPoints < challenge.hintCost) {
                if(!challenge || this.insightPoints < challenge.hintCost) this.showNotification('Not enough Insight Points!', 'error');
                return;
            }
            this.insightPoints -= challenge.hintCost;
            this.hintUsedThisLevel = true;
            this.updatePlayerStats();
            this.updateHintButtonState();
            this.saveProgressToServer();

            const instructions = document.getElementById('instructions-content');
            const hintBox = document.createElement('div');
            hintBox.className = "hint-box";
            hintBox.innerHTML = `<strong style="color: var(--accent-blue);">Hint:</strong> ${challenge.hint}`;
            instructions.appendChild(hintBox);
        },
        updateHintButtonState() {
            const hintButton = document.getElementById('hint-button');
            if (!hintButton) return;
            const challenge = this.challenges[this.currentChallenge - 1];
            hintButton.disabled = !challenge || this.hintUsedThisLevel || this.insightPoints < challenge.hintCost;
        },
        addVis(visData, isInstant = false) {
            const visContainer = document.getElementById('vis-container');
            if (!visData || !visContainer) return;
            
            const items = Array.isArray(visData) ? visData : [visData];
            
            items.forEach((item, index) => {
                const visEl = document.createElement('div');
                visEl.className = 'layer-vis';
                let style = { bgColor: 'var(--accent-blue)', height: '40px', textColor: 'var(--bg-dark)' };
                switch(item.type) {
                    case 'relu': style = { bgColor: 'var(--accent-green)', height: '20px', textColor: 'var(--bg-dark)' }; break;
                    case 'pool': style = { bgColor: 'purple', textColor: 'white' }; break;
                    case 'flat': style = { bgColor: '#666', textColor: 'white' }; break;
                    case 'linear': style = { bgColor: 'var(--accent-yellow)', textColor: 'var(--bg-dark)' }; break;
                    case 'dropout': style = { bgColor: 'var(--accent-red)', textColor: 'white', height: '20px' }; break;
                    case 'data': case 'teal': style = { bgColor: 'teal', textColor: 'white' }; break;
                    case 'optim': style = { bgColor: 'orange', textColor: 'var(--bg-dark)' }; break;
                }
                visEl.style.backgroundColor = style.bgColor;
                visEl.style.height = style.height;
                visEl.style.color = style.textColor;
                visEl.textContent = item.label;
                
                visEl.classList.add('fade-in');
                setTimeout(() => visContainer.appendChild(visEl), isInstant ? 0 : 100 * index);
            });
        },
        showScreen(screenId) {
            document.querySelectorAll('.game-screen').forEach(s => s.classList.remove('active-screen'));
            const activeScreen = document.getElementById(screenId);
            if (activeScreen) activeScreen.classList.add('active-screen');
        },
        showNotification(message, type = 'success') {
            const notif = document.getElementById('notification');
            if (!notif) return;
            notif.textContent = message;
            notif.className = 'show';
            notif.style.backgroundColor = type === 'error' ? 'var(--accent-red)' : type === 'success' ? 'var(--accent-green)' : 'var(--accent-blue)';
        }
    };
    game.init();
});