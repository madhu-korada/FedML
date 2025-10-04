# IoT Anomaly Detection Model & Federated Aggregation Analysis

## 🧠 **Model Architecture: Deep Autoencoder**

### **Model Overview**
The system uses a **Deep Autoencoder** for anomaly detection in IoT network traffic. The model is defined in `model/autoencoder.py`:

```python
class AutoEncoder(nn.Module):
    def __init__(self, output_dim=115):  # 115 network traffic features
```

### **Architecture Details**

#### **Encoder (Compression Path)**
```
Input: 115 features → 86 → 57 → 38 → 28 (bottleneck)
```

1. **Layer 1**: `115 → 86` (75% of input) + Tanh activation
2. **Layer 2**: `86 → 57` (50% of input) + Tanh activation  
3. **Layer 3**: `57 → 38` (33% of input) + Tanh activation
4. **Layer 4**: `38 → 28` (25% of input) - **Bottleneck layer**

#### **Decoder (Reconstruction Path)**
```
Bottleneck: 28 → 38 → 57 → 86 → 115 (reconstructed output)
```

1. **Layer 1**: `28 → 38` (33% of input) + Tanh activation
2. **Layer 2**: `38 → 57` (50% of input) + Tanh activation
3. **Layer 3**: `57 → 86` (75% of input) + Tanh activation
4. **Layer 4**: `86 → 115` (full reconstruction)

### **Key Characteristics**

- **Symmetric Architecture**: Encoder mirrors decoder structure
- **Compression Ratio**: ~4:1 (115 → 28 features in bottleneck)
- **Activation Function**: Tanh throughout (smooth, bounded [-1,1])
- **No Dropout**: Commented out (could be enabled for regularization)
- **Total Parameters**: ~20,000 parameters (relatively lightweight)

### **Anomaly Detection Principle**

1. **Training Phase**: Learn to reconstruct **normal/benign** traffic patterns
2. **Detection Phase**: High reconstruction error indicates **anomalous** traffic
3. **Threshold**: MSE > threshold → Anomaly detected

---

## 🔄 **Federated Learning Aggregation**

### **Aggregation Algorithm: FedAvg (Federated Averaging)**

The system uses the **FedAvg** algorithm for model aggregation, implemented in the FedML framework.

### **How FedAvg Works**

#### **Step 1: Local Training**
Each IoT device (client) trains the autoencoder locally:

```python
# In FedDetectTrainer.train()
criterion = nn.MSELoss()  # Reconstruction loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

# Train on benign traffic only
for epoch in range(args.epochs):
    for batch_idx, x in enumerate(train_data):
        decode = model(x)  # Reconstruct input
        loss = criterion(decode, x)  # MSE loss
        loss.backward()
        optimizer.step()
```

#### **Step 2: Model Parameter Extraction**
```python
# Extract local model parameters
def get_model_params(self):
    return self.model.cpu().state_dict()
```

#### **Step 3: Weighted Aggregation**
The server aggregates parameters using weighted averaging:

```python
# FedAvg aggregation formula
for k in avg_params.keys():
    for i in range(len(clients)):
        local_sample_number, local_model_params = client_updates[i]
        w = local_sample_number / total_training_samples  # Weight by data size
        if i == 0:
            avg_params[k] = local_model_params[k] * w
        else:
            avg_params[k] += local_model_params[k] * w
```

#### **Step 4: Global Model Distribution**
```python
# Update all clients with aggregated model
def set_model_params(self, model_parameters):
    self.model.load_state_dict(model_parameters)
```

### **Aggregation Configuration**

From `fedml_config.yaml`:
```yaml
train_args:
  federated_optimizer: "FedAvg"
  client_num_in_total: 9        # 9 IoT devices
  client_num_per_round: 9       # All clients participate
  comm_round: 10                # 10 communication rounds
  epochs: 1                     # 1 local epoch per round
  batch_size: 10
  learning_rate: 0.03
```

---

## 🎯 **Anomaly Detection Mechanism**

### **Threshold Calculation**
The system uses a sophisticated global threshold calculation:

```python
def _get_threshold_global(self, args, device):
    # Use samples 5000-8000 from benign data for threshold calculation
    benign_th = benign_data[5000:8000]
    
    # Calculate MSE for all devices
    mse_global = torch.cat(mse_list).mean(dim=1)
    
    # Threshold = Mean + 3 * Standard Deviation
    threshold_global = torch.mean(mse_global) + 3 * torch.std(mse_global)
    
    return threshold_global
```

### **Detection Process**
```python
# During testing
diff = threshold_func(model(x), x)  # MSE between input and reconstruction
mse = diff.mean(dim=1)

# Classification
if mse > threshold:
    prediction = "ANOMALY/ATTACK"
else:
    prediction = "NORMAL/BENIGN"
```

### **Evaluation Metrics**
The system tracks comprehensive metrics:

- **True Positive (TP)**: Correctly detected attacks
- **True Negative (TN)**: Correctly identified benign traffic  
- **False Positive (FP)**: Benign traffic flagged as attack
- **False Negative (FN)**: Missed attacks

**Derived Metrics**:
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **True Positive Rate (TPR)**: TP / (TP + FN)
- **False Positive Rate (FPR)**: FP / (FP + TN)

---

## 🏗️ **System Architecture**

### **Federated Learning Flow**

1. **Initialization**:
   ```python
   model = AutoEncoder(output_dim=115)
   trainer = FedDetectTrainer(model, args)
   aggregator = FedDetectAggregator(model, args)
   ```

2. **Training Loop** (10 rounds):
   - Each device trains locally on benign traffic
   - Devices send model parameters to server
   - Server aggregates using FedAvg
   - Updated global model sent back to devices

3. **Evaluation**:
   - Global threshold calculated from benign data
   - Model tested on attack traffic
   - Metrics computed and logged

### **Data Distribution**

- **Training Data**: Benign traffic (first 5000 samples per device)
- **Threshold Data**: Benign traffic (samples 5000-8000 per device)  
- **Test Data**: Attack traffic (Gafgyt + Mirai attacks)

### **Privacy Preservation**

- **No Raw Data Sharing**: Only model parameters exchanged
- **Local Training**: Each device keeps its data locally
- **Aggregated Learning**: Global model benefits from all devices without data exposure

---

## 🔍 **Model Strengths & Characteristics**

### **Strengths**
1. **Lightweight**: Only ~20K parameters, suitable for IoT devices
2. **Unsupervised**: Learns from benign traffic only (no labeled attacks needed)
3. **Federated**: Preserves privacy while leveraging collective knowledge
4. **Device-Agnostic**: Same architecture works across different IoT devices
5. **Robust Threshold**: Statistical approach (μ + 3σ) for anomaly detection

### **Design Choices**
1. **Tanh Activation**: Smooth gradients, bounded output
2. **Symmetric Architecture**: Ensures proper reconstruction capability  
3. **Progressive Compression**: Gradual dimensionality reduction (115→86→57→38→28)
4. **MSE Loss**: Appropriate for reconstruction tasks
5. **Adam Optimizer**: Adaptive learning rates for stable training

### **Federated Learning Benefits**
1. **Collective Intelligence**: Model learns from 9 different device types
2. **Privacy Preservation**: Raw network traffic stays on devices
3. **Scalability**: Can easily add more IoT devices
4. **Robustness**: Global model more robust than individual device models

---

## 📊 **Expected Performance**

Based on the architecture and approach:

- **Detection Capability**: Should detect novel attack patterns not seen during training
- **False Positive Rate**: Controlled by threshold tuning (3σ approach)
- **Generalization**: Benefits from diverse IoT device data in federated setting
- **Efficiency**: Lightweight model suitable for resource-constrained IoT devices

The system represents a practical implementation of federated anomaly detection for IoT security, balancing privacy, efficiency, and detection performance.

