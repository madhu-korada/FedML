# Simple IoT Autoencoder Training

This directory contains a standalone training system for the IoT anomaly detection autoencoder that doesn't require the full FedML framework complexity.

## 🚀 Quick Start

### **Easiest Way - Interactive Menu**
```bash
python run_training.py
# Shows menu with options 1-5 for different training modes
```

### **Command Line Options**
```bash
# Quick training (5 rounds, synthetic data)
python run_training.py quick

# Full training (10 rounds, real data)
python run_training.py full

# Demo with visualization
python run_training.py demo

# Compare different configurations
python run_training.py compare
```

### **Advanced Usage**
```bash
# Full control with command line arguments
python simple_train.py --rounds 15 --epochs 2 --lr 0.01 --use_real_data --save_model --visualize

# Quick synthetic training
python simple_train.py --rounds 5 --use_real_data

# Save trained model
python simple_train.py --save_model --model_path ./my_model.pt
```

## 📋 **What This Training System Does**

### **Simulates Federated Learning**
Instead of actual distributed training, it:
1. **Loads data** from each IoT device (9 devices total)
2. **Trains locally** on each device's benign traffic
3. **Aggregates models** using FedAvg algorithm (weighted averaging)
4. **Repeats** for multiple communication rounds

### **Training Process**
```
Round 1: Device1 → Local Model1
         Device2 → Local Model2
         ...
         Device9 → Local Model9
         ↓
         FedAvg Aggregation → Global Model

Round 2: Global Model → Each Device
         Local Training → Updated Local Models
         ↓
         FedAvg Aggregation → Updated Global Model
         
... (repeat for N rounds)
```

## 🎯 **Training Modes**

### **1. Quick Training** (`quick`)
- **5 communication rounds**
- **Synthetic data** (fast, no downloads needed)
- **Basic logging**
- Perfect for testing and development

### **2. Full Training** (`full`)
- **10 communication rounds**
- **Real IoT data** (downloads if needed)
- **Model evaluation** with attack detection
- **Saves trained model**
- Production-quality training

### **3. Demo Training** (`demo`)
- **8 communication rounds**
- **3 selected devices** (faster)
- **Real data + visualizations**
- **Training plots** and metrics
- Great for presentations and learning

### **4. Configuration Comparison** (`compare`)
- **Tests multiple configurations**:
  - Low learning rate (0.01)
  - High learning rate (0.05)  
  - More rounds (15)
- **Compares final losses**
- **Finds best configuration**

### **5. Custom Training**
- **User-defined parameters**
- **Interactive configuration**
- **Flexible options**

## 📊 **Sample Output**

```
🚀 Starting IoT Autoencoder Training
============================================================
Configuration:
  Communication rounds: 10
  Local epochs: 1
  Learning rate: 0.03
  Data type: Real
  Device: cuda
============================================================

============================================================
Communication Round 1/10
============================================================
Training on Danmini_Doorbell...
  Danmini_Doorbell - Epoch 1/1, Loss: 0.045623
Training on Ecobee_Thermostat...
  Ecobee_Thermostat - Epoch 1/1, Loss: 0.052341
...
Performing federated averaging...
Aggregated 9 models with 45000 total samples
Round 1 completed in 12.34s
Global loss: 0.048234

... (continues for all rounds)

✅ Training completed!
Final loss: 0.012456

📊 Evaluation Results:
  Anomaly threshold: 0.034567
  Attack detection rate: 87.50%

💾 Model saved to: ./trained_models/iot_autoencoder.pt
```

## 🔧 **Command Line Options**

| Option | Description | Default |
|--------|-------------|---------|
| `--rounds` | Communication rounds | 10 |
| `--epochs` | Local epochs per round | 1 |
| `--lr` | Learning rate | 0.03 |
| `--data_dir` | Data directory path | ./data_og |
| `--use_real_data` | Use real IoT data | False (synthetic) |
| `--save_model` | Save trained model | False |
| `--model_path` | Model save path | ./trained_models/iot_autoencoder.pt |
| `--visualize` | Show training plots | False |
| `--evaluate` | Evaluate after training | False |
| `--devices` | Specific devices to use | All 9 devices |

## 📈 **Training Visualizations**

When using `--visualize` or demo mode, you get:

1. **Global Loss Curve**: Loss reduction over communication rounds
2. **Device-Specific Losses**: Individual device training progress
3. **Participation Chart**: Number of devices per round
4. **Loss Distribution**: Histogram of all training losses

## 🎛️ **Configuration Examples**

### **Fast Development**
```bash
python simple_train.py --rounds 3 --epochs 1
# 3 rounds, synthetic data, quick testing
```

### **High-Quality Training**
```bash
python simple_train.py --rounds 20 --epochs 2 --lr 0.01 --use_real_data --save_model --evaluate
# 20 rounds, real data, thorough training
```

### **Specific Devices**
```bash
python simple_train.py --devices Danmini_Doorbell Ecobee_Thermostat --use_real_data
# Train only on doorbell and thermostat
```

### **Learning Rate Tuning**
```bash
python simple_train.py --lr 0.001 --rounds 15 --use_real_data
# Lower learning rate, more rounds
```

## 📁 **File Structure**

```
iot/anomaly_detection_for_cybersecurity/
├── simple_train.py              # Main training script
├── run_training.py              # Interactive runner
├── model/
│   └── autoencoder.py           # Model definition
├── data_og/                     # Real IoT data (if available)
├── trained_models/              # Saved models
│   └── iot_autoencoder.pt
└── training_results.png         # Training visualizations
```

## 🔍 **What's Different from Original FedML**

### **Simplified**
- ✅ **Single Python file** (no complex framework setup)
- ✅ **No MPI/distributed computing** required
- ✅ **Simulated federated learning** (sequential training)
- ✅ **Easy debugging** and modification

### **Maintained**
- ✅ **Same model architecture** (AutoEncoder)
- ✅ **Same training logic** (MSE loss, Adam optimizer)
- ✅ **Same FedAvg aggregation** (weighted averaging)
- ✅ **Same data preprocessing** (normalization)

### **Enhanced**
- ✅ **Better logging** and progress tracking
- ✅ **Visualization support** 
- ✅ **Flexible configuration**
- ✅ **Model evaluation** built-in

## 🎯 **Use Cases**

### **Research & Development**
- **Quick prototyping** of federated learning ideas
- **Algorithm testing** without distributed setup
- **Parameter tuning** and experimentation

### **Education & Learning**
- **Understanding federated learning** concepts
- **Autoencoder training** demonstration
- **IoT security** research

### **Production Testing**
- **Model validation** before deployment
- **Baseline performance** measurement
- **Configuration optimization**

## ⚡ **Performance Notes**

### **Training Speed**
- **Synthetic data**: ~30 seconds for 10 rounds
- **Real data**: ~2-5 minutes for 10 rounds (depends on data loading)
- **GPU acceleration**: Automatically uses CUDA if available

### **Memory Usage**
- **Model size**: ~20K parameters (~80KB)
- **Data loading**: Loads one device at a time (memory efficient)
- **Batch processing**: 32 samples per batch (configurable)

### **Scalability**
- **Devices**: Tested with all 9 IoT devices
- **Rounds**: Tested up to 50 communication rounds
- **Data size**: Handles up to 5000 samples per device

## 🚨 **Troubleshooting**

### **Common Issues**

1. **"No module named 'model.autoencoder'"**
   ```bash
   # Make sure you're in the correct directory
   cd iot/anomaly_detection_for_cybersecurity
   python simple_train.py
   ```

2. **"Data directory not found"**
   ```bash
   # Use synthetic data for testing
   python simple_train.py  # (without --use_real_data)
   
   # Or specify data directory
   python simple_train.py --data_dir /path/to/your/data
   ```

3. **"CUDA out of memory"**
   ```bash
   # The script automatically falls back to CPU
   # Or reduce batch size in the code (line ~200)
   ```

4. **Slow training**
   ```bash
   # Use fewer devices for testing
   python simple_train.py --devices Danmini_Doorbell Ecobee_Thermostat
   
   # Or use synthetic data
   python simple_train.py  # (faster than real data)
   ```

## 🎉 **Getting Started Checklist**

- [ ] Navigate to the IoT directory
- [ ] Run `python run_training.py` for interactive menu
- [ ] Try quick training first (`python run_training.py quick`)
- [ ] Experiment with real data (`python simple_train.py --use_real_data`)
- [ ] Save your trained model (`--save_model`)
- [ ] Visualize results (`--visualize`)

This training system gives you all the power of the original federated learning setup in a simple, easy-to-use package!
