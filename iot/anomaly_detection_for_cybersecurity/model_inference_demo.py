#!/usr/bin/env python3
"""
IoT Anomaly Detection Model Inference Demo

This script demonstrates the forward pass and inference process of the autoencoder
model used for IoT anomaly detection. It shows how the model processes network
traffic data and makes anomaly predictions.

Usage:
    python model_inference_demo.py
    python model_inference_demo.py --use_real_data
    python model_inference_demo.py --visualize
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging

# Add the model directory to path
sys.path.append('.')
from model.autoencoder import AutoEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelInferenceDemo:
    """Demonstrates autoencoder forward pass and anomaly detection inference."""
    
    def __init__(self, model_path=None, data_dir="./data_og"):
        """Initialize the demo.
        
        Args:
            model_path (str): Path to trained model weights (optional)
            data_dir (str): Path to data directory
        """
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = AutoEncoder(output_dim=115)
        self.model.to(self.device)
        
        # Load trained weights if available
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded trained model from {model_path}")
        else:
            logger.info("Using randomly initialized model (for demonstration)")
        
        # Load normalization parameters
        self.load_normalization_params()
        
        # Set model to evaluation mode
        self.model.eval()
        
        logger.info(f"Model initialized on device: {self.device}")
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
                logger.warning("Normalization files not found. Using dummy values.")
                self.min_dataset = np.zeros(115)
                self.max_dataset = np.ones(115)
        except Exception as e:
            logger.error(f"Error loading normalization parameters: {e}")
            self.min_dataset = np.zeros(115)
            self.max_dataset = np.ones(115)
    
    def create_synthetic_data(self, num_samples=10):
        """Create synthetic network traffic data for demonstration.
        
        Args:
            num_samples (int): Number of samples to generate
            
        Returns:
            tuple: (benign_data, attack_data) as numpy arrays
        """
        logger.info(f"Creating {num_samples} synthetic samples...")
        
        # Create benign traffic (normal patterns)
        # Simulate typical IoT network patterns
        benign_data = []
        for i in range(num_samples):
            sample = np.random.normal(0.5, 0.1, 115)  # Normal distribution around 0.5
            # Add some structure to make it more realistic
            sample[:20] = np.random.uniform(0.3, 0.7, 20)  # MI features
            sample[20:40] = np.random.uniform(0.2, 0.8, 20)  # H features
            sample[40:80] = np.random.normal(0.4, 0.15, 40)  # HH features
            sample[80:] = np.random.normal(0.6, 0.1, 35)  # HpHp features
            
            # Ensure values are in [0, 1] range (normalized)
            sample = np.clip(sample, 0, 1)
            benign_data.append(sample)
        
        # Create attack traffic (anomalous patterns)
        attack_data = []
        for i in range(num_samples):
            sample = np.random.normal(0.5, 0.3, 115)  # Higher variance
            # Add anomalous spikes
            spike_indices = np.random.choice(115, 10, replace=False)
            sample[spike_indices] = np.random.uniform(0.8, 1.0, 10)
            
            # Add some zero values (common in attacks)
            zero_indices = np.random.choice(115, 5, replace=False)
            sample[zero_indices] = 0
            
            # Ensure values are in [0, 1] range
            sample = np.clip(sample, 0, 1)
            attack_data.append(sample)
        
        return np.array(benign_data), np.array(attack_data)
    
    def load_real_data(self, device_name="Danmini_Doorbell", num_samples=10):
        """Load real data from the dataset.
        
        Args:
            device_name (str): Name of IoT device
            num_samples (int): Number of samples to load
            
        Returns:
            tuple: (benign_data, attack_data) as numpy arrays
        """
        device_path = self.data_dir / device_name
        
        if not device_path.exists():
            logger.error(f"Device directory not found: {device_path}")
            return self.create_synthetic_data(num_samples)
        
        try:
            # Load benign data
            benign_file = device_path / "benign_traffic.csv"
            benign_df = pd.read_csv(benign_file)
            benign_data = benign_df.sample(min(num_samples, len(benign_df))).values
            
            # Normalize benign data
            benign_data = (benign_data - self.min_dataset) / (self.max_dataset - self.min_dataset)
            benign_data = np.nan_to_num(benign_data, 0)
            
            # Load attack data
            attack_data = None
            gafgyt_path = device_path / "gafgyt_attacks"
            if gafgyt_path.exists():
                attack_files = list(gafgyt_path.glob("*.csv"))
                if attack_files:
                    attack_df = pd.read_csv(attack_files[0])  # Load first attack type
                    attack_data = attack_df.sample(min(num_samples, len(attack_df))).values
                    # Note: Attack data uses different normalization in original code
                    attack_data = (attack_data - attack_data.mean()) / (attack_data.std() + 1e-8)
                    attack_data = np.nan_to_num(attack_data, 0)
            
            if attack_data is None:
                logger.warning("No attack data found, creating synthetic attack data")
                _, attack_data = self.create_synthetic_data(num_samples)
            
            logger.info(f"Loaded real data from {device_name}")
            return benign_data[:num_samples], attack_data[:num_samples]
            
        except Exception as e:
            logger.error(f"Error loading real data: {e}")
            return self.create_synthetic_data(num_samples)
    
    def demonstrate_forward_pass(self, input_data, data_type="benign"):
        """Demonstrate the forward pass through the autoencoder.
        
        Args:
            input_data (np.ndarray): Input network traffic data
            data_type (str): Type of data ("benign" or "attack")
            
        Returns:
            dict: Results including reconstructions, losses, etc.
        """
        logger.info(f"Demonstrating forward pass on {data_type} data...")
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_data).to(self.device)
        
        results = {}
        
        with torch.no_grad():
            # Forward pass through encoder
            encoded = self.model.enc(input_tensor)
            logger.info(f"Encoder output shape: {encoded.shape}")
            logger.info(f"Compression ratio: {input_tensor.shape[1]} -> {encoded.shape[1]} ({encoded.shape[1]/input_tensor.shape[1]:.2%})")
            
            # Forward pass through decoder
            decoded = self.model.dec(encoded)
            logger.info(f"Decoder output shape: {decoded.shape}")
            
            # Full forward pass
            reconstructed = self.model(input_tensor)
            
            # Calculate reconstruction error (MSE)
            mse_loss = nn.MSELoss(reduction='none')
            reconstruction_errors = mse_loss(reconstructed, input_tensor)
            sample_errors = reconstruction_errors.mean(dim=1)
            
            results = {
                'input': input_tensor.cpu().numpy(),
                'encoded': encoded.cpu().numpy(),
                'reconstructed': reconstructed.cpu().numpy(),
                'reconstruction_errors': reconstruction_errors.cpu().numpy(),
                'sample_errors': sample_errors.cpu().numpy(),
                'mean_error': sample_errors.mean().item(),
                'std_error': sample_errors.std().item()
            }
        
        # Print detailed results
        print(f"\n{'='*60}")
        print(f"FORWARD PASS RESULTS - {data_type.upper()} DATA")
        print(f"{'='*60}")
        print(f"Input shape: {results['input'].shape}")
        print(f"Encoded shape: {results['encoded'].shape}")
        print(f"Reconstructed shape: {results['reconstructed'].shape}")
        print(f"Mean reconstruction error: {results['mean_error']:.6f}")
        print(f"Std reconstruction error: {results['std_error']:.6f}")
        print(f"Min sample error: {results['sample_errors'].min():.6f}")
        print(f"Max sample error: {results['sample_errors'].max():.6f}")
        
        return results
    
    def calculate_anomaly_threshold(self, benign_results):
        """Calculate anomaly detection threshold from benign data.
        
        Args:
            benign_results (dict): Results from benign data forward pass
            
        Returns:
            float: Anomaly threshold
        """
        sample_errors = benign_results['sample_errors']
        mean_error = np.mean(sample_errors)
        std_error = np.std(sample_errors)
        
        # Threshold = mean + 3 * std (as used in the original code)
        threshold = mean_error + 3 * std_error
        
        logger.info(f"Calculated anomaly threshold: {threshold:.6f}")
        logger.info(f"  Mean error: {mean_error:.6f}")
        logger.info(f"  Std error: {std_error:.6f}")
        
        return threshold
    
    def perform_anomaly_detection(self, benign_results, attack_results, threshold):
        """Perform anomaly detection and calculate metrics.
        
        Args:
            benign_results (dict): Results from benign data
            attack_results (dict): Results from attack data
            threshold (float): Anomaly threshold
            
        Returns:
            dict: Detection metrics
        """
        logger.info("Performing anomaly detection...")
        
        benign_errors = benign_results['sample_errors']
        attack_errors = attack_results['sample_errors']
        
        # Classify samples
        benign_predictions = benign_errors > threshold  # Should be mostly False
        attack_predictions = attack_errors > threshold  # Should be mostly True
        
        # Calculate metrics
        true_negatives = np.sum(~benign_predictions)  # Benign correctly classified
        false_positives = np.sum(benign_predictions)   # Benign wrongly classified as attack
        true_positives = np.sum(attack_predictions)    # Attack correctly classified
        false_negatives = np.sum(~attack_predictions)  # Attack wrongly classified as benign
        
        total = len(benign_errors) + len(attack_errors)
        accuracy = (true_positives + true_negatives) / total
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'threshold': threshold,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
        # Print results
        print(f"\n{'='*60}")
        print(f"ANOMALY DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"Threshold: {threshold:.6f}")
        print(f"True Positives (attacks detected): {true_positives}")
        print(f"True Negatives (benign classified correctly): {true_negatives}")
        print(f"False Positives (benign classified as attack): {false_positives}")
        print(f"False Negatives (attacks missed): {false_negatives}")
        print(f"")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        
        return metrics
    
    def visualize_results(self, benign_results, attack_results, threshold):
        """Create visualizations of the inference results.
        
        Args:
            benign_results (dict): Results from benign data
            attack_results (dict): Results from attack data
            threshold (float): Anomaly threshold
        """
        logger.info("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('IoT Autoencoder Model Inference Analysis', fontsize=16, fontweight='bold')
        
        # 1. Reconstruction error distribution
        axes[0,0].hist(benign_results['sample_errors'], bins=20, alpha=0.7, label='Benign', color='green')
        axes[0,0].hist(attack_results['sample_errors'], bins=20, alpha=0.7, label='Attack', color='red')
        axes[0,0].axvline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.4f}')
        axes[0,0].set_xlabel('Reconstruction Error (MSE)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Reconstruction Error Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Sample-wise reconstruction errors
        sample_indices = range(len(benign_results['sample_errors']))
        axes[0,1].scatter(sample_indices, benign_results['sample_errors'], 
                         color='green', alpha=0.7, label='Benign')
        axes[0,1].scatter(sample_indices, attack_results['sample_errors'], 
                         color='red', alpha=0.7, label='Attack')
        axes[0,1].axhline(threshold, color='black', linestyle='--', label=f'Threshold')
        axes[0,1].set_xlabel('Sample Index')
        axes[0,1].set_ylabel('Reconstruction Error')
        axes[0,1].set_title('Sample-wise Reconstruction Errors')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Feature reconstruction comparison (first sample)
        feature_indices = range(min(50, benign_results['input'].shape[1]))  # Show first 50 features
        axes[0,2].plot(feature_indices, benign_results['input'][0][:len(feature_indices)], 
                      'g-', label='Original Benign', alpha=0.7)
        axes[0,2].plot(feature_indices, benign_results['reconstructed'][0][:len(feature_indices)], 
                      'g--', label='Reconstructed Benign', alpha=0.7)
        axes[0,2].set_xlabel('Feature Index')
        axes[0,2].set_ylabel('Feature Value')
        axes[0,2].set_title('Feature Reconstruction (Benign Sample)')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Encoded representation (bottleneck features)
        encoded_benign = benign_results['encoded'][0]
        encoded_attack = attack_results['encoded'][0]
        bottleneck_indices = range(len(encoded_benign))
        
        axes[1,0].bar([i-0.2 for i in bottleneck_indices], encoded_benign, 
                     width=0.4, alpha=0.7, label='Benign', color='green')
        axes[1,0].bar([i+0.2 for i in bottleneck_indices], encoded_attack, 
                     width=0.4, alpha=0.7, label='Attack', color='red')
        axes[1,0].set_xlabel('Bottleneck Feature Index')
        axes[1,0].set_ylabel('Encoded Value')
        axes[1,0].set_title('Encoded Representation (Bottleneck Layer)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Feature-wise reconstruction errors
        feature_errors_benign = np.mean(benign_results['reconstruction_errors'], axis=0)
        feature_errors_attack = np.mean(attack_results['reconstruction_errors'], axis=0)
        
        axes[1,1].plot(feature_errors_benign[:50], 'g-', alpha=0.7, label='Benign')
        axes[1,1].plot(feature_errors_attack[:50], 'r-', alpha=0.7, label='Attack')
        axes[1,1].set_xlabel('Feature Index')
        axes[1,1].set_ylabel('Mean Reconstruction Error')
        axes[1,1].set_title('Feature-wise Reconstruction Errors (First 50 Features)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Attack vs Benign comparison (first sample)
        axes[1,2].plot(feature_indices, attack_results['input'][0][:len(feature_indices)], 
                      'r-', label='Original Attack', alpha=0.7)
        axes[1,2].plot(feature_indices, attack_results['reconstructed'][0][:len(feature_indices)], 
                      'r--', label='Reconstructed Attack', alpha=0.7)
        axes[1,2].set_xlabel('Feature Index')
        axes[1,2].set_ylabel('Feature Value')
        axes[1,2].set_title('Feature Reconstruction (Attack Sample)')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path("./exploration_results")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'model_inference_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Visualizations saved to: {output_dir / 'model_inference_analysis.png'}")
    
    def run_complete_demo(self, use_real_data=False, visualize=False, num_samples=10):
        """Run the complete inference demonstration.
        
        Args:
            use_real_data (bool): Whether to use real data or synthetic
            visualize (bool): Whether to create visualizations
            num_samples (int): Number of samples to process
        """
        logger.info("Starting complete model inference demonstration...")
        
        # Load data
        if use_real_data:
            benign_data, attack_data = self.load_real_data(num_samples=num_samples)
        else:
            benign_data, attack_data = self.create_synthetic_data(num_samples=num_samples)
        
        # Demonstrate forward pass on benign data
        benign_results = self.demonstrate_forward_pass(benign_data, "benign")
        
        # Demonstrate forward pass on attack data
        attack_results = self.demonstrate_forward_pass(attack_data, "attack")
        
        # Calculate threshold
        threshold = self.calculate_anomaly_threshold(benign_results)
        
        # Perform anomaly detection
        metrics = self.perform_anomaly_detection(benign_results, attack_results, threshold)
        
        # Create visualizations if requested
        if visualize:
            self.visualize_results(benign_results, attack_results, threshold)
        
        print(f"\n{'='*60}")
        print(f"DEMONSTRATION COMPLETE")
        print(f"{'='*60}")
        print(f"Data type: {'Real' if use_real_data else 'Synthetic'}")
        print(f"Samples processed: {num_samples} benign + {num_samples} attack")
        print(f"Model performance: {metrics['accuracy']:.2%} accuracy")
        
        return {
            'benign_results': benign_results,
            'attack_results': attack_results,
            'threshold': threshold,
            'metrics': metrics
        }


def main():
    """Main function to run the inference demonstration."""
    parser = argparse.ArgumentParser(description='IoT Autoencoder Model Inference Demo')
    parser.add_argument('--use_real_data', action='store_true',
                       help='Use real data instead of synthetic')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to process')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str, default='./data_og',
                       help='Path to data directory')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = ModelInferenceDemo(model_path=args.model_path, data_dir=args.data_dir)
    
    # Run demonstration
    results = demo.run_complete_demo(
        use_real_data=args.use_real_data,
        visualize=args.visualize,
        num_samples=args.num_samples
    )
    
    print(f"\nDemo completed successfully!")
    if args.visualize:
        print(f"Check ./exploration_results/ for visualization outputs.")


if __name__ == "__main__":
    main()
