# IoT Anomaly Detection Dataset Exploration

This directory contains tools for exploring the N-BaIoT dataset used in federated learning for IoT anomaly detection.

## Dataset Overview

The **N-BaIoT** (Network-based detection of IoT Botnet attacks using deep autoencoders) dataset contains network traffic data from 9 different IoT devices:

### Devices
1. **Danmini_Doorbell** - Smart doorbell
2. **Ecobee_Thermostat** - Smart thermostat  
3. **Ennio_Doorbell** - Smart doorbell
4. **Philips_B120N10_Baby_Monitor** - Baby monitor
5. **Provision_PT_737E_Security_Camera** - Security camera
6. **Provision_PT_838_Security_Camera** - Security camera
7. **Samsung_SNH_1011_N_Webcam** - Webcam
8. **SimpleHome_XCS7_1002_WHT_Security_Camera** - Security camera
9. **SimpleHome_XCS7_1003_WHT_Security_Camera** - Security camera

### Attack Types
- **Gafgyt** attacks: combo, junk, scan, tcp, udp
- **Mirai** attacks: ack, scan, syn, udp, udpplain

Note: Some devices (Ennio_Doorbell, Samsung_SNH_1011_N_Webcam) only have Gafgyt attacks.

## Data Exploration Script

### Usage

```bash
# Full exploration with all visualizations
python explore_data.py

# Quick exploration (summary only)
python explore_data.py --quick

# Specify custom data directory
python explore_data.py --data_dir /path/to/your/data

# Custom sample size for analysis
python explore_data.py --sample_size 2000
```

### Features

The exploration script provides:

1. **Dataset Overview**
   - Sample counts per device
   - Attack type distribution
   - Device category analysis
   - Overall traffic distribution

2. **Feature Analysis**
   - Feature value distributions
   - Correlation analysis
   - Missing value detection
   - Variance analysis

3. **Attack Pattern Analysis**
   - Comparison between benign and malicious traffic
   - Different attack type characteristics
   - PCA visualization (if scikit-learn available)

4. **Comprehensive Summary Report**
   - Device statistics
   - Attack type breakdown
   - Federated learning setup details

### Output

Results are saved to `./exploration_results/` including:
- `dataset_overview.png` - Overall dataset statistics
- `feature_statistics.png` - Feature analysis plots
- `attack_pattern_analysis.png` - Attack vs benign comparison
- Console output with detailed summary

### Requirements

```bash
pip install pandas numpy matplotlib seaborn pathlib

# Optional for PCA visualization
pip install scikit-learn
```

## Data Structure

```
data_og/
├── Device_Name/
│   ├── benign_traffic.csv      # Normal network traffic
│   ├── gafgyt_attacks/
│   │   ├── combo.csv
│   │   ├── junk.csv
│   │   ├── scan.csv
│   │   ├── tcp.csv
│   │   └── udp.csv
│   └── mirai_attacks/          # (if available)
│       ├── ack.csv
│       ├── scan.csv
│       ├── syn.csv
│       ├── udp.csv
│       └── udpplain.csv
├── min_dataset.txt             # Normalization parameters
└── max_dataset.txt
```

## Features

Each CSV file contains 115 network traffic features derived from:
- **MI** (Mutual Information) statistics
- **H** (Entropy) statistics  
- **HH** (Host-Host) flow statistics
- **HpHp** (Host-Port Host-Port) flow statistics

Features are calculated at different time windows (L5, L3, L1, L0.1, L0.01 seconds).

## Federated Learning Context

In the federated learning setup:
- Each IoT device acts as a **client**
- **Training data**: Benign traffic (for autoencoder training)
- **Test data**: Attack traffic (for anomaly detection)
- **Model**: Deep autoencoder for anomaly detection
- **Aggregation**: FedAvg algorithm

The goal is to train a federated anomaly detection model that can identify botnet attacks across different IoT devices while preserving data privacy.

## Example Analysis

```python
from explore_data import IoTDataExplorer

# Initialize explorer
explorer = IoTDataExplorer(data_dir="./data_og")

# Get device information
device_info = explorer.get_device_info()

# Run specific analysis
explorer.analyze_feature_statistics(sample_size=1000)
explorer.compare_attack_patterns()

# Generate summary
explorer.generate_summary_report(device_info)
```

## Related Files

- `data_loader.py` - Main data loading logic used in training
- `fedml_config.yaml` - Federated learning configuration
- `fed_detect_trainer.py` - Training logic
- `fed_detect_aggregator.py` - Aggregation logic
