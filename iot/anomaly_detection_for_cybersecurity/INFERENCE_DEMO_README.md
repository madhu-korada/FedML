# IoT Autoencoder Model Inference Demo

This directory contains tools for demonstrating and inspecting the forward pass and inference process of the IoT anomaly detection autoencoder model.

## 🚀 Quick Start

### Simple Usage
```bash
# Interactive demo with menu
python run_inference_demo.py

# Or run specific demos directly
python run_inference_demo.py basic    # Basic synthetic data demo
python run_inference_demo.py real     # Real IoT data demo  
python run_inference_demo.py visual   # Demo with visualizations
python run_inference_demo.py step     # Step-by-step breakdown
python run_inference_demo.py arch     # Architecture analysis
```

### Advanced Usage
```bash
# Full control with command line options
python model_inference_demo.py --use_real_data --visualize --num_samples 20

# Quick synthetic demo
python model_inference_demo.py --num_samples 5

# Real data with visualizations
python model_inference_demo.py --use_real_data --visualize
```

## 📋 What You'll See

### 1. **Forward Pass Analysis**
- **Input processing**: How 115 network features are processed
- **Encoder compression**: 115 → 86 → 57 → 38 → 28 (bottleneck)
- **Decoder reconstruction**: 28 → 38 → 57 → 86 → 115 (output)
- **Reconstruction errors**: MSE between input and output

### 2. **Anomaly Detection Process**
- **Threshold calculation**: μ + 3σ from benign data
- **Classification logic**: MSE > threshold = Anomaly
- **Performance metrics**: Accuracy, Precision, Recall, F1-Score

### 3. **Model Architecture Inspection**
- **Parameter count**: ~20,000 parameters
- **Layer dimensions**: Detailed breakdown of each layer
- **Compression ratio**: 4:1 compression in bottleneck
- **Activation patterns**: Tanh activation analysis

## 🔍 Demo Modes

### **Basic Demo** (`basic`)
- Uses synthetic network traffic data
- Shows basic forward pass and reconstruction
- Quick demonstration of core concepts

### **Real Data Demo** (`real`)
- Loads actual IoT device data (Danmini_Doorbell)
- Demonstrates with real benign and attack traffic
- Shows realistic reconstruction errors

### **Visual Demo** (`visual`)
- Creates comprehensive visualizations
- Plots reconstruction error distributions
- Shows feature-wise analysis
- Saves plots to `./exploration_results/`

### **Step-by-Step Demo** (`step`)
- Breaks down each inference step
- Detailed logging of intermediate results
- Educational walkthrough of the process

### **Architecture Analysis** (`arch`)
- Analyzes model structure and parameters
- Tests different aspects of the architecture
- Provides technical insights

## 📊 Sample Output

```
==============================================================
FORWARD PASS RESULTS - BENIGN DATA
==============================================================
Input shape: (5, 115)
Encoded shape: (5, 28)
Reconstructed shape: (5, 115)
Mean reconstruction error: 0.023456
Std reconstruction error: 0.004321
Min sample error: 0.018234
Max sample error: 0.029876

==============================================================
ANOMALY DETECTION RESULTS
==============================================================
Threshold: 0.036456
True Positives (attacks detected): 4
True Negatives (benign classified correctly): 5
False Positives (benign classified as attack): 0
False Negatives (attacks missed): 1

Accuracy: 0.9000
Precision: 1.0000
Recall: 0.8000
F1-Score: 0.8889
```

## 🎯 Key Insights You'll Gain

### **Model Behavior**
- How the autoencoder compresses network traffic features
- What the bottleneck representation looks like
- How reconstruction quality differs between benign and attack traffic

### **Anomaly Detection Logic**
- Why higher reconstruction error indicates anomalies
- How the statistical threshold (μ + 3σ) works
- Trade-offs between false positives and false negatives

### **Feature Analysis**
- Which network features are most important for reconstruction
- How different attack types affect reconstruction patterns
- Feature-wise error distributions

### **Performance Characteristics**
- Model accuracy on different data types
- Computational efficiency (forward pass timing)
- Memory usage and parameter efficiency

## 🛠️ Customization Options

### **Data Sources**
```python
# Use different IoT devices
demo = ModelInferenceDemo(data_dir="./data_og")
benign_data, attack_data = demo.load_real_data(device_name="Ecobee_Thermostat")

# Create custom synthetic data
benign_data, attack_data = demo.create_synthetic_data(num_samples=50)
```

### **Model Variants**
```python
# Load trained model weights
demo = ModelInferenceDemo(model_path="./model_file_cache/global_model.pt")

# Use different model configurations
model = AutoEncoder(output_dim=115)  # Standard
```

### **Threshold Tuning**
```python
# Different threshold strategies
threshold = mean_error + 2 * std_error  # More sensitive
threshold = mean_error + 4 * std_error  # Less sensitive
```

## 📈 Visualization Outputs

When using `--visualize` or the visual demo, you'll get:

1. **Reconstruction Error Distribution**: Histogram comparing benign vs attack errors
2. **Sample-wise Errors**: Scatter plot of individual sample errors
3. **Feature Reconstruction**: Line plots showing original vs reconstructed features
4. **Bottleneck Analysis**: Bar chart of encoded representations
5. **Feature-wise Errors**: Analysis of which features have highest reconstruction errors
6. **Attack vs Benign Comparison**: Side-by-side feature comparisons

## 🔧 Requirements

```bash
pip install torch numpy pandas matplotlib seaborn
```

## 💡 Educational Use Cases

- **Understanding Autoencoders**: See how compression and reconstruction works
- **Anomaly Detection**: Learn threshold-based classification
- **Federated Learning**: Understand the model used in FL scenarios
- **IoT Security**: Explore network traffic analysis for cybersecurity
- **Deep Learning**: Practical example of unsupervised learning

## 🚨 Troubleshooting

### Common Issues

1. **"No module named 'model.autoencoder'"**
   - Make sure you're running from the correct directory
   - The script should be in the same directory as the `model/` folder

2. **"Data directory not found"**
   - Check that `./data_og/` exists with IoT device folders
   - Use `--data_dir` to specify a different path

3. **"CUDA out of memory"**
   - The model automatically uses CPU if GPU is unavailable
   - Reduce `--num_samples` if processing large batches

### Performance Notes

- **Synthetic data**: Instant generation, good for testing
- **Real data**: Requires dataset download, more realistic results
- **Visualizations**: May take a few seconds to generate plots
- **Large samples**: Use `--num_samples 100+` for statistical significance

This demo provides a hands-on way to understand how the IoT anomaly detection model works internally and how it makes predictions on network traffic data!
