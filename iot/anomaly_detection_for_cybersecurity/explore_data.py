#!/usr/bin/env python3
"""
IoT Anomaly Detection Dataset Explorer

This script explores the N-BaIoT dataset used for federated learning-based
anomaly detection in IoT devices. It analyzes network traffic patterns from
9 different IoT devices and their associated attack patterns.

Dataset: N-BaIoT (Network-based detection of IoT Botnet attacks using deep autoencoders)
Paper: https://arxiv.org/abs/1805.03409
"""

import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IoTDataExplorer:
    """Comprehensive IoT dataset explorer for the N-BaIoT dataset."""
    
    def __init__(self, data_dir="./data_og"):
        """Initialize the data explorer.
        
        Args:
            data_dir (str): Path to the data directory containing device folders
        """
        self.data_dir = Path(data_dir)
        self.device_list = [
            "Danmini_Doorbell",
            "Ecobee_Thermostat", 
            "Ennio_Doorbell",
            "Philips_B120N10_Baby_Monitor",
            "Provision_PT_737E_Security_Camera",
            "Provision_PT_838_Security_Camera",
            "Samsung_SNH_1011_N_Webcam",
            "SimpleHome_XCS7_1002_WHT_Security_Camera",
            "SimpleHome_XCS7_1003_WHT_Security_Camera",
        ]
        
        # Load normalization parameters
        self.load_normalization_params()
        
        # Create output directory for plots
        self.output_dir = Path("./exploration_results")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized IoT Data Explorer with {len(self.device_list)} devices")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_normalization_params(self):
        """Load min/max normalization parameters."""
        try:
            min_file = self.data_dir / "min_dataset.txt"
            max_file = self.data_dir / "max_dataset.txt"
            
            if min_file.exists() and max_file.exists():
                self.min_dataset = np.loadtxt(min_file)
                self.max_dataset = np.loadtxt(max_file)
                logger.info(f"Loaded normalization parameters: {len(self.min_dataset)} features")
            else:
                logger.warning("Normalization files not found. Will compute from data.")
                self.min_dataset = None
                self.max_dataset = None
        except Exception as e:
            logger.error(f"Error loading normalization parameters: {e}")
            self.min_dataset = None
            self.max_dataset = None
    
    def get_device_info(self):
        """Get basic information about each device and its data."""
        device_info = {}
        
        for device in self.device_list:
            device_path = self.data_dir / device
            if not device_path.exists():
                logger.warning(f"Device directory not found: {device}")
                continue
                
            info = {
                'device_name': device,
                'benign_samples': 0,
                'attack_samples': 0,
                'attack_types': [],
                'has_mirai': False,
                'has_gafgyt': False
            }
            
            # Check benign traffic
            benign_file = device_path / "benign_traffic.csv"
            if benign_file.exists():
                try:
                    benign_df = pd.read_csv(benign_file)
                    info['benign_samples'] = len(benign_df)
                    info['feature_count'] = len(benign_df.columns)
                except Exception as e:
                    logger.error(f"Error reading benign data for {device}: {e}")
            
            # Check attack types
            gafgyt_path = device_path / "gafgyt_attacks"
            mirai_path = device_path / "mirai_attacks"
            
            if gafgyt_path.exists():
                info['has_gafgyt'] = True
                gafgyt_files = list(gafgyt_path.glob("*.csv"))
                info['attack_types'].extend([f"gafgyt_{f.stem}" for f in gafgyt_files])
                
                # Count gafgyt samples
                for attack_file in gafgyt_files:
                    try:
                        attack_df = pd.read_csv(attack_file)
                        info['attack_samples'] += len(attack_df)
                    except Exception as e:
                        logger.error(f"Error reading {attack_file}: {e}")
            
            if mirai_path.exists():
                info['has_mirai'] = True
                mirai_files = list(mirai_path.glob("*.csv"))
                info['attack_types'].extend([f"mirai_{f.stem}" for f in mirai_files])
                
                # Count mirai samples
                for attack_file in mirai_files:
                    try:
                        attack_df = pd.read_csv(attack_file)
                        info['attack_samples'] += len(attack_df)
                    except Exception as e:
                        logger.error(f"Error reading {attack_file}: {e}")
            
            device_info[device] = info
            logger.info(f"Processed device: {device} - Benign: {info['benign_samples']}, Attacks: {info['attack_samples']}")
        
        return device_info
    
    def create_dataset_overview(self, device_info):
        """Create overview visualizations of the dataset."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('IoT Dataset Overview - N-BaIoT', fontsize=16, fontweight='bold')
        
        # 1. Sample counts per device
        devices = list(device_info.keys())
        benign_counts = [device_info[d]['benign_samples'] for d in devices]
        attack_counts = [device_info[d]['attack_samples'] for d in devices]
        
        x = np.arange(len(devices))
        width = 0.35
        
        axes[0,0].bar(x - width/2, benign_counts, width, label='Benign', color='green', alpha=0.7)
        axes[0,0].bar(x + width/2, attack_counts, width, label='Attack', color='red', alpha=0.7)
        axes[0,0].set_title('Sample Counts per Device')
        axes[0,0].set_xlabel('IoT Devices')
        axes[0,0].set_ylabel('Number of Samples')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels([d.replace('_', '\n') for d in devices], rotation=45, ha='right')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Attack type distribution
        attack_type_counts = {}
        for device, info in device_info.items():
            for attack_type in info['attack_types']:
                attack_family = attack_type.split('_')[0]  # gafgyt or mirai
                if attack_family not in attack_type_counts:
                    attack_type_counts[attack_family] = 0
                attack_type_counts[attack_family] += 1
        
        if attack_type_counts:
            axes[0,1].pie(attack_type_counts.values(), labels=attack_type_counts.keys(), 
                         autopct='%1.1f%%', startangle=90)
            axes[0,1].set_title('Attack Family Distribution')
        
        # 3. Device categories
        device_categories = {
            'Doorbell': ['Danmini_Doorbell', 'Ennio_Doorbell'],
            'Camera': ['Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera',
                      'Samsung_SNH_1011_N_Webcam', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
                      'SimpleHome_XCS7_1003_WHT_Security_Camera'],
            'Thermostat': ['Ecobee_Thermostat'],
            'Baby Monitor': ['Philips_B120N10_Baby_Monitor']
        }
        
        category_counts = {}
        for category, device_names in device_categories.items():
            category_counts[category] = len([d for d in device_names if d in device_info])
        
        axes[1,0].bar(category_counts.keys(), category_counts.values(), 
                     color=['blue', 'orange', 'green', 'purple'], alpha=0.7)
        axes[1,0].set_title('Device Categories')
        axes[1,0].set_ylabel('Number of Devices')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Attack vs Benign ratio
        total_benign = sum(benign_counts)
        total_attack = sum(attack_counts)
        
        axes[1,1].pie([total_benign, total_attack], 
                     labels=['Benign Traffic', 'Attack Traffic'],
                     colors=['green', 'red'], 
                     autopct='%1.1f%%', 
                     startangle=90)
        axes[1,1].set_title('Overall Traffic Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def analyze_feature_statistics(self, sample_size=1000):
        """Analyze feature statistics across devices."""
        logger.info("Analyzing feature statistics...")
        
        # Sample data from each device
        all_benign_data = []
        all_attack_data = []
        device_labels = []
        
        for device in self.device_list[:3]:  # Analyze first 3 devices for speed
            device_path = self.data_dir / device
            if not device_path.exists():
                continue
                
            # Load benign data
            benign_file = device_path / "benign_traffic.csv"
            if benign_file.exists():
                try:
                    benign_df = pd.read_csv(benign_file)
                    # Sample data
                    sample_benign = benign_df.sample(min(sample_size, len(benign_df)))
                    all_benign_data.append(sample_benign)
                    device_labels.extend([device] * len(sample_benign))
                except Exception as e:
                    logger.error(f"Error loading benign data for {device}: {e}")
            
            # Load attack data (sample from gafgyt attacks)
            gafgyt_path = device_path / "gafgyt_attacks"
            if gafgyt_path.exists():
                attack_files = list(gafgyt_path.glob("*.csv"))
                if attack_files:
                    try:
                        # Load first attack file
                        attack_df = pd.read_csv(attack_files[0])
                        sample_attack = attack_df.sample(min(sample_size//2, len(attack_df)))
                        all_attack_data.append(sample_attack)
                    except Exception as e:
                        logger.error(f"Error loading attack data for {device}: {e}")
        
        if not all_benign_data:
            logger.error("No benign data loaded!")
            return
        
        # Combine benign data
        combined_benign = pd.concat(all_benign_data, ignore_index=True)
        
        # Feature analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Feature Statistics Analysis', fontsize=16, fontweight='bold')
        
        # 1. Feature value distributions (first 10 features)
        feature_subset = combined_benign.iloc[:, :10]
        axes[0,0].boxplot([feature_subset[col].dropna() for col in feature_subset.columns])
        axes[0,0].set_title('Feature Value Distributions (First 10 Features)')
        axes[0,0].set_xlabel('Features')
        axes[0,0].set_ylabel('Values')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Feature correlation heatmap (sample of features)
        feature_sample = combined_benign.iloc[:, :20].corr()
        sns.heatmap(feature_sample, ax=axes[0,1], cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        axes[0,1].set_title('Feature Correlation Matrix (First 20 Features)')
        
        # 3. Missing values analysis
        missing_counts = combined_benign.isnull().sum()
        non_zero_missing = missing_counts[missing_counts > 0]
        
        if len(non_zero_missing) > 0:
            axes[1,0].bar(range(len(non_zero_missing)), non_zero_missing.values)
            axes[1,0].set_title('Missing Values per Feature')
            axes[1,0].set_xlabel('Features with Missing Values')
            axes[1,0].set_ylabel('Count of Missing Values')
        else:
            axes[1,0].text(0.5, 0.5, 'No Missing Values Found', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Missing Values Analysis')
        
        # 4. Feature variance analysis
        feature_vars = combined_benign.var().sort_values(ascending=False)
        top_vars = feature_vars.head(20)
        
        axes[1,1].bar(range(len(top_vars)), top_vars.values)
        axes[1,1].set_title('Top 20 Features by Variance')
        axes[1,1].set_xlabel('Features (ranked by variance)')
        axes[1,1].set_ylabel('Variance')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("FEATURE STATISTICS SUMMARY")
        print("="*60)
        print(f"Total features: {len(combined_benign.columns)}")
        print(f"Total samples analyzed: {len(combined_benign)}")
        print(f"Features with missing values: {len(non_zero_missing)}")
        print(f"Average feature correlation: {feature_sample.values[np.triu_indices_from(feature_sample.values, k=1)].mean():.4f}")
        
        return combined_benign
    
    def compare_attack_patterns(self, sample_size=500):
        """Compare different attack patterns."""
        logger.info("Analyzing attack patterns...")
        
        # Focus on one device with both attack types
        target_device = "Danmini_Doorbell"  # Has both gafgyt and mirai
        device_path = self.data_dir / target_device
        
        if not device_path.exists():
            logger.error(f"Target device {target_device} not found!")
            return
        
        # Load benign data
        benign_file = device_path / "benign_traffic.csv"
        benign_data = pd.read_csv(benign_file).sample(sample_size)
        
        # Load attack data
        attack_data = {}
        
        # Gafgyt attacks
        gafgyt_path = device_path / "gafgyt_attacks"
        if gafgyt_path.exists():
            for attack_file in gafgyt_path.glob("*.csv"):
                attack_type = f"gafgyt_{attack_file.stem}"
                try:
                    attack_df = pd.read_csv(attack_file)
                    attack_data[attack_type] = attack_df.sample(min(sample_size//2, len(attack_df)))
                except Exception as e:
                    logger.error(f"Error loading {attack_file}: {e}")
        
        # Mirai attacks
        mirai_path = device_path / "mirai_attacks"
        if mirai_path.exists():
            for attack_file in mirai_path.glob("*.csv"):
                attack_type = f"mirai_{attack_file.stem}"
                try:
                    attack_df = pd.read_csv(attack_file)
                    attack_data[attack_type] = attack_df.sample(min(sample_size//2, len(attack_df)))
                except Exception as e:
                    logger.error(f"Error loading {attack_file}: {e}")
        
        if not attack_data:
            logger.error("No attack data loaded!")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Attack Pattern Analysis - {target_device}', fontsize=16, fontweight='bold')
        
        # 1. Feature means comparison
        feature_means = {'benign': benign_data.mean()}
        for attack_type, data in attack_data.items():
            feature_means[attack_type] = data.mean()
        
        # Plot first 10 features
        feature_indices = range(10)
        x = np.arange(len(feature_indices))
        width = 0.15
        
        colors = ['green', 'red', 'blue', 'orange', 'purple', 'brown']
        for i, (data_type, means) in enumerate(feature_means.items()):
            axes[0,0].bar(x + i*width, means.iloc[:10], width, 
                         label=data_type, alpha=0.7, color=colors[i % len(colors)])
        
        axes[0,0].set_title('Feature Means Comparison (First 10 Features)')
        axes[0,0].set_xlabel('Features')
        axes[0,0].set_ylabel('Mean Values')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Feature variance comparison
        feature_vars = {'benign': benign_data.var()}
        for attack_type, data in attack_data.items():
            feature_vars[attack_type] = data.var()
        
        for i, (data_type, variances) in enumerate(feature_vars.items()):
            axes[0,1].bar(x + i*width, variances.iloc[:10], width, 
                         label=data_type, alpha=0.7, color=colors[i % len(colors)])
        
        axes[0,1].set_title('Feature Variance Comparison (First 10 Features)')
        axes[0,1].set_xlabel('Features')
        axes[0,1].set_ylabel('Variance')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. PCA visualization (if possible)
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Combine all data for PCA
            all_data = [benign_data]
            labels = ['benign'] * len(benign_data)
            
            for attack_type, data in attack_data.items():
                all_data.append(data)
                labels.extend([attack_type] * len(data))
            
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Standardize and apply PCA
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(combined_data.fillna(0))
            
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            # Plot PCA results
            unique_labels = list(set(labels))
            for i, label in enumerate(unique_labels):
                mask = [l == label for l in labels]
                axes[1,0].scatter(pca_result[mask, 0], pca_result[mask, 1], 
                                 label=label, alpha=0.6, color=colors[i % len(colors)])
            
            axes[1,0].set_title('PCA Visualization (First 2 Components)')
            axes[1,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            axes[1,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
        except ImportError:
            axes[1,0].text(0.5, 0.5, 'PCA requires scikit-learn\npip install scikit-learn', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('PCA Visualization (scikit-learn required)')
        
        # 4. Sample size comparison
        sample_sizes = {'benign': len(benign_data)}
        for attack_type, data in attack_data.items():
            sample_sizes[attack_type] = len(data)
        
        axes[1,1].bar(sample_sizes.keys(), sample_sizes.values(), 
                     color=colors[:len(sample_sizes)], alpha=0.7)
        axes[1,1].set_title('Sample Sizes by Traffic Type')
        axes[1,1].set_ylabel('Number of Samples')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'attack_pattern_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print attack pattern summary
        print("\n" + "="*60)
        print("ATTACK PATTERN ANALYSIS SUMMARY")
        print("="*60)
        print(f"Device analyzed: {target_device}")
        print(f"Benign samples: {len(benign_data)}")
        for attack_type, data in attack_data.items():
            print(f"{attack_type} samples: {len(data)}")
        
        return attack_data
    
    def generate_summary_report(self, device_info):
        """Generate a comprehensive summary report."""
        print("\n" + "="*80)
        print("IoT ANOMALY DETECTION DATASET SUMMARY REPORT")
        print("="*80)
        
        print(f"\nDataset: N-BaIoT (Network-based detection of IoT Botnet attacks)")
        print(f"Data Directory: {self.data_dir}")
        print(f"Total Devices: {len(device_info)}")
        
        # Device summary
        print(f"\nDEVICE BREAKDOWN:")
        print("-" * 50)
        total_benign = 0
        total_attack = 0
        
        for device, info in device_info.items():
            print(f"{device}:")
            print(f"  - Benign samples: {info['benign_samples']:,}")
            print(f"  - Attack samples: {info['attack_samples']:,}")
            print(f"  - Features: {info.get('feature_count', 'N/A')}")
            print(f"  - Attack types: {len(info['attack_types'])}")
            print(f"  - Has Mirai: {info['has_mirai']}")
            print(f"  - Has Gafgyt: {info['has_gafgyt']}")
            print()
            
            total_benign += info['benign_samples']
            total_attack += info['attack_samples']
        
        print(f"OVERALL STATISTICS:")
        print("-" * 50)
        print(f"Total benign samples: {total_benign:,}")
        print(f"Total attack samples: {total_attack:,}")
        print(f"Total samples: {total_benign + total_attack:,}")
        print(f"Benign/Attack ratio: {total_benign/total_attack:.2f}:1" if total_attack > 0 else "No attack data")
        
        # Attack type summary
        attack_families = set()
        attack_subtypes = set()
        
        for info in device_info.values():
            for attack_type in info['attack_types']:
                family, subtype = attack_type.split('_', 1)
                attack_families.add(family)
                attack_subtypes.add(subtype)
        
        print(f"\nATTACK ANALYSIS:")
        print("-" * 50)
        print(f"Attack families: {', '.join(sorted(attack_families))}")
        print(f"Attack subtypes: {', '.join(sorted(attack_subtypes))}")
        
        # Device categories
        device_categories = {
            'Doorbell': ['Danmini_Doorbell', 'Ennio_Doorbell'],
            'Security Camera': ['Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera',
                               'Samsung_SNH_1011_N_Webcam', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
                               'SimpleHome_XCS7_1003_WHT_Security_Camera'],
            'Thermostat': ['Ecobee_Thermostat'],
            'Baby Monitor': ['Philips_B120N10_Baby_Monitor']
        }
        
        print(f"\nDEVICE CATEGORIES:")
        print("-" * 50)
        for category, devices in device_categories.items():
            available_devices = [d for d in devices if d in device_info]
            print(f"{category}: {len(available_devices)} devices")
            for device in available_devices:
                print(f"  - {device}")
        
        print(f"\nFEATURE INFORMATION:")
        print("-" * 50)
        if device_info:
            sample_device = next(iter(device_info.values()))
            if 'feature_count' in sample_device:
                print(f"Features per sample: {sample_device['feature_count']}")
                print("Feature types: Network traffic statistics (MI, H, HH, HpHp)")
                print("Normalization: Min-Max scaling applied")
        
        print(f"\nFEDERATED LEARNING SETUP:")
        print("-" * 50)
        print(f"Clients: {len(device_info)} (one per device)")
        print("Training data: Benign traffic (autoencoder)")
        print("Test data: Attack traffic (anomaly detection)")
        print("Model: Deep Autoencoder")
        print("Aggregation: FedAvg")
        
        print("="*80)
    
    def run_full_exploration(self):
        """Run the complete data exploration pipeline."""
        logger.info("Starting comprehensive IoT dataset exploration...")
        
        # 1. Get device information
        logger.info("Step 1: Gathering device information...")
        device_info = self.get_device_info()
        
        if not device_info:
            logger.error("No device data found! Please check data directory.")
            return
        
        # 2. Create dataset overview
        logger.info("Step 2: Creating dataset overview...")
        self.create_dataset_overview(device_info)
        
        # 3. Analyze feature statistics
        logger.info("Step 3: Analyzing feature statistics...")
        self.analyze_feature_statistics()
        
        # 4. Compare attack patterns
        logger.info("Step 4: Comparing attack patterns...")
        self.compare_attack_patterns()
        
        # 5. Generate summary report
        logger.info("Step 5: Generating summary report...")
        self.generate_summary_report(device_info)
        
        logger.info(f"Exploration complete! Results saved to: {self.output_dir}")
        print(f"\nExploration results saved to: {self.output_dir}")


def main():
    """Main function to run the data exploration."""
    parser = argparse.ArgumentParser(description='Explore IoT Anomaly Detection Dataset')
    parser.add_argument('--data_dir', type=str, default='./data_og',
                       help='Path to the data directory')
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='Sample size for analysis')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick exploration (fewer visualizations)')
    
    args = parser.parse_args()
    
    # Initialize explorer
    explorer = IoTDataExplorer(data_dir=args.data_dir)
    
    if args.quick:
        # Quick exploration
        device_info = explorer.get_device_info()
        explorer.generate_summary_report(device_info)
    else:
        # Full exploration
        explorer.run_full_exploration()


if __name__ == "__main__":
    main()
