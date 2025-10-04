#!/usr/bin/env python3
"""
Simple IoT Autoencoder Training Script

This script provides a standalone way to train the IoT anomaly detection autoencoder
without the full federated learning framework. It simulates the federated training
process by training on multiple IoT devices sequentially and aggregating the results.

Usage:
    python simple_train.py                    # Basic training
    python simple_train.py --epochs 10       # More epochs
    python simple_train.py --save_model      # Save trained model
    python simple_train.py --use_real_data   # Use real IoT data
    python simple_train.py --visualize       # Show training plots
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
from collections import OrderedDict
import time

# Add current directory to path for imports
sys.path.append('.')

# Import model
from model.autoencoder import AutoEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleIoTTrainer:
    """Simple trainer for IoT autoencoder without federated learning complexity."""
    
    def __init__(self, data_dir="./data_og", device=None):
        """Initialize the trainer.
        
        Args:
            data_dir (str): Path to data directory
            device (torch.device): Device to train on
        """
        self.data_dir = Path(data_dir)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # IoT device list
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
        
        # Initialize model
        self.model = AutoEncoder(output_dim=115)
        self.model.to(self.device)
        
        # Load normalization parameters
        self.load_normalization_params()
        
        # Training history
        self.training_history = {
            'global_losses': [],
            'device_losses': {device: [] for device in self.device_list},
            'aggregation_history': []
        }
        
        logger.info(f"Initialized trainer on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_normalization_params(self):
        """Load min/max normalization parameters."""
        try:
            min_file = self.data_dir / "min_dataset.txt"
            max_file = self.data_dir / "max_dataset.txt"
            
            if min_file.exists() and max_file.exists():
                self.min_dataset = np.loadtxt(min_file)
                self.max_dataset = np.loadtxt(max_file)
                logger.info(f"Loaded normalization parameters for {len(self.min_dataset)} features")
            else:
                logger.warning("Normalization files not found. Creating dummy parameters.")
                self.min_dataset = np.zeros(115)
                self.max_dataset = np.ones(115)
        except Exception as e:
            logger.error(f"Error loading normalization parameters: {e}")
            self.min_dataset = np.zeros(115)
            self.max_dataset = np.ones(115)
    
    def load_device_data(self, device_name, use_real_data=True, num_samples=5000):
        """Load data for a specific IoT device.
        
        Args:
            device_name (str): Name of the IoT device
            use_real_data (bool): Whether to use real data or synthetic
            num_samples (int): Number of samples to load
            
        Returns:
            torch.utils.data.DataLoader: Data loader for training
        """
        if use_real_data:
            device_path = self.data_dir / device_name
            
            if not device_path.exists():
                logger.warning(f"Device directory not found: {device_path}. Using synthetic data.")
                return self.create_synthetic_data(device_name, num_samples)
            
            try:
                # Load benign traffic data
                benign_file = device_path / "benign_traffic.csv"
                if not benign_file.exists():
                    logger.warning(f"Benign traffic file not found for {device_name}. Using synthetic data.")
                    return self.create_synthetic_data(device_name, num_samples)
                
                benign_data = pd.read_csv(benign_file)
                benign_data = benign_data[:num_samples]  # Limit samples
                benign_data = np.array(benign_data)
                
                # Handle NaN values
                benign_data[np.isnan(benign_data)] = 0
                
                # Normalize data
                benign_data = (benign_data - self.min_dataset) / (self.max_dataset - self.min_dataset + 1e-8)
                benign_data = np.clip(benign_data, 0, 1)  # Ensure [0,1] range
                
                # Create data loader
                dataset = torch.utils.data.TensorDataset(torch.FloatTensor(benign_data))
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=32, shuffle=True, num_workers=0
                )
                
                logger.info(f"Loaded {len(benign_data)} samples for {device_name}")
                return dataloader
                
            except Exception as e:
                logger.error(f"Error loading data for {device_name}: {e}")
                return self.create_synthetic_data(device_name, num_samples)
        else:
            return self.create_synthetic_data(device_name, num_samples)
    
    def create_synthetic_data(self, device_name, num_samples=5000):
        """Create synthetic training data for a device.
        
        Args:
            device_name (str): Name of the device (for seeding)
            num_samples (int): Number of samples to generate
            
        Returns:
            torch.utils.data.DataLoader: Synthetic data loader
        """
        # Set seed based on device name for reproducibility
        np.random.seed(hash(device_name) % 2**32)
        
        # Generate synthetic benign traffic patterns
        data = []
        for i in range(num_samples):
            # Create realistic IoT traffic patterns
            sample = np.random.normal(0.5, 0.1, 115)
            
            # Add device-specific patterns
            if "Doorbell" in device_name:
                sample[:20] = np.random.uniform(0.3, 0.7, 20)  # Doorbell patterns
            elif "Camera" in device_name:
                sample[20:60] = np.random.uniform(0.4, 0.8, 40)  # Camera patterns
            elif "Thermostat" in device_name:
                sample[60:] = np.random.normal(0.6, 0.05, 55)  # Thermostat patterns
            
            # Ensure values are in [0, 1] range
            sample = np.clip(sample, 0, 1)
            data.append(sample)
        
        data = np.array(data)
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(data))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=True, num_workers=0
        )
        
        logger.info(f"Created {num_samples} synthetic samples for {device_name}")
        return dataloader
    
    def train_local_model(self, device_name, dataloader, epochs=1, learning_rate=0.03):
        """Train the model locally on one device's data.
        
        Args:
            device_name (str): Name of the device
            dataloader: Data loader for training
            epochs (int): Number of local epochs
            learning_rate (float): Learning rate
            
        Returns:
            tuple: (model_state_dict, training_loss, num_samples)
        """
        logger.info(f"Training on {device_name}...")
        
        # Create local copy of model
        local_model = AutoEncoder(output_dim=115)
        local_model.load_state_dict(self.model.state_dict())
        local_model.to(self.device)
        local_model.train()
        
        # Set up training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)
        
        epoch_losses = []
        total_samples = 0
        
        for epoch in range(epochs):
            batch_losses = []
            
            for batch_idx, (data,) in enumerate(dataloader):
                data = data.to(self.device).float()
                total_samples += len(data)
                
                # Forward pass
                optimizer.zero_grad()
                reconstructed = local_model(data)
                loss = criterion(reconstructed, data)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                batch_losses.append(loss.item())
            
            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)
            
            logger.info(f"  {device_name} - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
        
        avg_loss = np.mean(epoch_losses)
        self.training_history['device_losses'][device_name].append(avg_loss)
        
        return local_model.state_dict(), avg_loss, total_samples
    
    def federated_averaging(self, local_models_info):
        """Perform FedAvg aggregation of local models.
        
        Args:
            local_models_info (list): List of (state_dict, loss, num_samples) tuples
            
        Returns:
            OrderedDict: Aggregated model parameters
        """
        logger.info("Performing federated averaging...")
        
        # Calculate total samples for weighting
        total_samples = sum(info[2] for info in local_models_info)
        
        # Initialize aggregated parameters
        aggregated_params = None
        
        for state_dict, loss, num_samples in local_models_info:
            weight = num_samples / total_samples
            
            if aggregated_params is None:
                # Initialize with first model
                aggregated_params = OrderedDict()
                for key, param in state_dict.items():
                    aggregated_params[key] = param * weight
            else:
                # Add weighted parameters
                for key, param in state_dict.items():
                    aggregated_params[key] += param * weight
        
        # Record aggregation info
        avg_loss = np.mean([info[1] for info in local_models_info])
        self.training_history['aggregation_history'].append({
            'participating_devices': len(local_models_info),
            'total_samples': total_samples,
            'average_loss': avg_loss
        })
        
        logger.info(f"Aggregated {len(local_models_info)} models with {total_samples} total samples")
        return aggregated_params
    
    def train_federated(self, communication_rounds=10, local_epochs=1, learning_rate=0.03, 
                       use_real_data=True, participating_devices=None):
        """Train the model using simulated federated learning.
        
        Args:
            communication_rounds (int): Number of communication rounds
            local_epochs (int): Local epochs per round
            learning_rate (float): Learning rate
            use_real_data (bool): Whether to use real data
            participating_devices (list): List of devices to use (None = all)
            
        Returns:
            dict: Training results
        """
        logger.info("Starting federated training simulation...")
        logger.info(f"Communication rounds: {communication_rounds}")
        logger.info(f"Local epochs per round: {local_epochs}")
        logger.info(f"Learning rate: {learning_rate}")
        
        devices_to_use = participating_devices or self.device_list
        logger.info(f"Participating devices: {len(devices_to_use)}")
        
        # Load data for all devices
        device_dataloaders = {}
        for device_name in devices_to_use:
            try:
                dataloader = self.load_device_data(device_name, use_real_data)
                device_dataloaders[device_name] = dataloader
            except Exception as e:
                logger.error(f"Failed to load data for {device_name}: {e}")
        
        if not device_dataloaders:
            raise ValueError("No device data could be loaded!")
        
        logger.info(f"Successfully loaded data for {len(device_dataloaders)} devices")
        
        # Training loop
        for round_num in range(communication_rounds):
            logger.info(f"\n{'='*60}")
            logger.info(f"Communication Round {round_num + 1}/{communication_rounds}")
            logger.info(f"{'='*60}")
            
            round_start_time = time.time()
            local_models_info = []
            
            # Local training on each device
            for device_name, dataloader in device_dataloaders.items():
                try:
                    state_dict, loss, num_samples = self.train_local_model(
                        device_name, dataloader, local_epochs, learning_rate
                    )
                    local_models_info.append((state_dict, loss, num_samples))
                except Exception as e:
                    logger.error(f"Training failed for {device_name}: {e}")
            
            if not local_models_info:
                logger.error("No local models trained successfully!")
                break
            
            # Federated averaging
            aggregated_params = self.federated_averaging(local_models_info)
            
            # Update global model
            self.model.load_state_dict(aggregated_params)
            
            # Calculate global loss
            global_loss = np.mean([info[1] for info in local_models_info])
            self.training_history['global_losses'].append(global_loss)
            
            round_time = time.time() - round_start_time
            logger.info(f"Round {round_num + 1} completed in {round_time:.2f}s")
            logger.info(f"Global loss: {global_loss:.6f}")
        
        logger.info("\n🎉 Federated training completed!")
        
        return {
            'final_loss': self.training_history['global_losses'][-1],
            'training_history': self.training_history,
            'model_state_dict': self.model.state_dict()
        }
    
    def evaluate_model(self, use_real_data=True):
        """Evaluate the trained model on test data.
        
        Args:
            use_real_data (bool): Whether to use real attack data
            
        Returns:
            dict: Evaluation results
        """
        logger.info("Evaluating trained model...")
        
        self.model.eval()
        
        # Calculate threshold from benign data
        threshold = self.calculate_threshold(use_real_data)
        
        # Test on attack data
        attack_results = self.test_on_attacks(use_real_data, threshold)
        
        return {
            'threshold': threshold,
            'attack_results': attack_results
        }
    
    def calculate_threshold(self, use_real_data=True):
        """Calculate anomaly detection threshold from benign data.
        
        Args:
            use_real_data (bool): Whether to use real data
            
        Returns:
            float: Anomaly threshold
        """
        logger.info("Calculating anomaly threshold...")
        
        all_errors = []
        
        for device_name in self.device_list[:3]:  # Use first 3 devices for speed
            try:
                if use_real_data:
                    device_path = self.data_dir / device_name
                    if device_path.exists():
                        benign_file = device_path / "benign_traffic.csv"
                        if benign_file.exists():
                            # Load threshold data (samples 5000-8000 as in original)
                            benign_data = pd.read_csv(benign_file)
                            if len(benign_data) > 8000:
                                threshold_data = benign_data[5000:8000].values
                            else:
                                threshold_data = benign_data[-1000:].values  # Last 1000 samples
                            
                            # Normalize
                            threshold_data[np.isnan(threshold_data)] = 0
                            threshold_data = (threshold_data - self.min_dataset) / (self.max_dataset - self.min_dataset + 1e-8)
                            threshold_data = np.clip(threshold_data, 0, 1)
                        else:
                            continue
                    else:
                        continue
                else:
                    # Use synthetic data
                    threshold_data = np.random.normal(0.5, 0.1, (1000, 115))
                    threshold_data = np.clip(threshold_data, 0, 1)
                
                # Calculate reconstruction errors
                with torch.no_grad():
                    data_tensor = torch.FloatTensor(threshold_data).to(self.device)
                    reconstructed = self.model(data_tensor)
                    errors = torch.mean((reconstructed - data_tensor) ** 2, dim=1)
                    all_errors.extend(errors.cpu().numpy())
                    
            except Exception as e:
                logger.warning(f"Error calculating threshold for {device_name}: {e}")
        
        if not all_errors:
            logger.warning("No threshold data available, using default threshold")
            return 0.1
        
        # Calculate threshold: mean + 3 * std
        mean_error = np.mean(all_errors)
        std_error = np.std(all_errors)
        threshold = mean_error + 3 * std_error
        
        logger.info(f"Calculated threshold: {threshold:.6f} (mean: {mean_error:.6f}, std: {std_error:.6f})")
        return threshold
    
    def test_on_attacks(self, use_real_data=True, threshold=0.1):
        """Test the model on attack data.
        
        Args:
            use_real_data (bool): Whether to use real attack data
            threshold (float): Anomaly threshold
            
        Returns:
            dict: Test results
        """
        logger.info("Testing on attack data...")
        
        results = {
            'total_attacks': 0,
            'detected_attacks': 0,
            'detection_rate': 0.0,
            'average_attack_error': 0.0
        }
        
        attack_errors = []
        
        for device_name in self.device_list[:3]:  # Test on first 3 devices
            try:
                if use_real_data:
                    device_path = self.data_dir / device_name
                    if not device_path.exists():
                        continue
                    
                    # Load attack data
                    gafgyt_path = device_path / "gafgyt_attacks"
                    if gafgyt_path.exists():
                        attack_files = list(gafgyt_path.glob("*.csv"))
                        if attack_files:
                            attack_data = pd.read_csv(attack_files[0])[:500]  # First 500 samples
                            attack_data = np.array(attack_data)
                            
                            # Different normalization for attack data (as in original)
                            attack_data = (attack_data - attack_data.mean()) / (attack_data.std() + 1e-8)
                            attack_data = np.nan_to_num(attack_data, 0)
                        else:
                            continue
                    else:
                        continue
                else:
                    # Create synthetic attack data
                    attack_data = np.random.normal(0.5, 0.3, (500, 115))  # Higher variance
                    # Add anomalous patterns
                    for i in range(len(attack_data)):
                        spike_indices = np.random.choice(115, 10, replace=False)
                        attack_data[i, spike_indices] = np.random.uniform(0.8, 1.0, 10)
                    attack_data = np.clip(attack_data, 0, 1)
                
                # Test on attack data
                with torch.no_grad():
                    data_tensor = torch.FloatTensor(attack_data).to(self.device)
                    reconstructed = self.model(data_tensor)
                    errors = torch.mean((reconstructed - data_tensor) ** 2, dim=1)
                    attack_errors.extend(errors.cpu().numpy())
                
            except Exception as e:
                logger.warning(f"Error testing attacks for {device_name}: {e}")
        
        if attack_errors:
            results['total_attacks'] = len(attack_errors)
            results['detected_attacks'] = sum(1 for error in attack_errors if error > threshold)
            results['detection_rate'] = results['detected_attacks'] / results['total_attacks']
            results['average_attack_error'] = np.mean(attack_errors)
            
            logger.info(f"Attack detection results:")
            logger.info(f"  Total attacks: {results['total_attacks']}")
            logger.info(f"  Detected attacks: {results['detected_attacks']}")
            logger.info(f"  Detection rate: {results['detection_rate']:.2%}")
            logger.info(f"  Average attack error: {results['average_attack_error']:.6f}")
        
        return results
    
    def visualize_training(self, save_path=None):
        """Visualize training results.
        
        Args:
            save_path (str): Path to save plots (optional)
        """
        if not self.training_history['global_losses']:
            logger.warning("No training history to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('IoT Autoencoder Training Results', fontsize=16, fontweight='bold')
        
        # 1. Global loss over communication rounds
        axes[0,0].plot(self.training_history['global_losses'], 'b-', linewidth=2)
        axes[0,0].set_title('Global Loss Over Communication Rounds')
        axes[0,0].set_xlabel('Communication Round')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Device-specific losses
        for device, losses in self.training_history['device_losses'].items():
            if losses:  # Only plot if device has loss history
                axes[0,1].plot(losses, label=device.replace('_', ' '), alpha=0.7)
        axes[0,1].set_title('Device-Specific Losses')
        axes[0,1].set_xlabel('Communication Round')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Participating devices per round
        participating = [info['participating_devices'] for info in self.training_history['aggregation_history']]
        if participating:
            axes[1,0].bar(range(len(participating)), participating, alpha=0.7)
            axes[1,0].set_title('Participating Devices per Round')
            axes[1,0].set_xlabel('Communication Round')
            axes[1,0].set_ylabel('Number of Devices')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Loss distribution
        all_losses = []
        for losses in self.training_history['device_losses'].values():
            all_losses.extend(losses)
        
        if all_losses:
            axes[1,1].hist(all_losses, bins=20, alpha=0.7, color='green')
            axes[1,1].set_title('Loss Distribution')
            axes[1,1].set_xlabel('Loss Value')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training plots saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, save_path):
        """Save the trained model.
        
        Args:
            save_path (str): Path to save the model
        """
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'normalization_params': {
                'min_dataset': self.min_dataset,
                'max_dataset': self.max_dataset
            }
        }, save_path)
        
        logger.info(f"Model saved to: {save_path}")


def main():
    """Main function to run the training."""
    parser = argparse.ArgumentParser(description='Simple IoT Autoencoder Training')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Local epochs per communication round')
    parser.add_argument('--rounds', type=int, default=10,
                       help='Number of communication rounds')
    parser.add_argument('--lr', type=float, default=0.03,
                       help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='./data_og',
                       help='Path to data directory')
    parser.add_argument('--use_real_data', action='store_true',
                       help='Use real IoT data instead of synthetic')
    parser.add_argument('--save_model', action='store_true',
                       help='Save the trained model')
    parser.add_argument('--model_path', type=str, default='./trained_models/iot_autoencoder.pt',
                       help='Path to save the model')
    parser.add_argument('--visualize', action='store_true',
                       help='Show training visualizations')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model after training')
    parser.add_argument('--devices', nargs='+', default=None,
                       help='Specific devices to train on (default: all)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SimpleIoTTrainer(data_dir=args.data_dir)
    
    print(f"🚀 Starting IoT Autoencoder Training")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Communication rounds: {args.rounds}")
    print(f"  Local epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Data type: {'Real' if args.use_real_data else 'Synthetic'}")
    print(f"  Device: {trainer.device}")
    print(f"{'='*60}")
    
    try:
        # Train the model
        results = trainer.train_federated(
            communication_rounds=args.rounds,
            local_epochs=args.epochs,
            learning_rate=args.lr,
            use_real_data=args.use_real_data,
            participating_devices=args.devices
        )
        
        print(f"\n✅ Training completed!")
        print(f"Final loss: {results['final_loss']:.6f}")
        
        # Evaluate model
        if args.evaluate:
            eval_results = trainer.evaluate_model(use_real_data=args.use_real_data)
            print(f"\n📊 Evaluation Results:")
            print(f"  Anomaly threshold: {eval_results['threshold']:.6f}")
            if eval_results['attack_results']['total_attacks'] > 0:
                print(f"  Attack detection rate: {eval_results['attack_results']['detection_rate']:.2%}")
        
        # Save model
        if args.save_model:
            trainer.save_model(args.model_path)
        
        # Visualize results
        if args.visualize:
            plot_path = './training_results.png' if args.visualize else None
            trainer.visualize_training(save_path=plot_path)
        
        print(f"\n🎉 All done!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
