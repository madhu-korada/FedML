#!/usr/bin/env python3
"""
Swarm Learning for IoT Anomaly Detection
Replaces federated learning with blockchain-based swarm learning
"""

import argparse
import logging
import os
import sys
import time
import torch
import yaml
from typing import Dict, List, Any

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import load_data
from model.autoencoder import AutoEncoder
from blockchain.ledger import SwarmBlockchain
from swarm.node import SwarmNode
from swarm.trainer import SwarmTrainer


class SwarmLearningRunner:
    """
    Main runner for Swarm Learning IoT Anomaly Detection
    """
    
    def __init__(self, args):
        self.args = args
        self.logger = self._setup_logging()
        self.device = self._setup_device()
        
        # Initialize components
        self.blockchain = None
        self.swarm_node = None
        self.trainer = None
        self.model = None
        self.dataset = None
        
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        if self.args.using_gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.args.gpu_id}")
            self.logger.info(f"Using GPU: {device}")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU")
        
        return device
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_dir = getattr(self.args, 'log_file_dir', './log')
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, f'swarm_node_{self.args.node_id}.log')),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(f"SwarmRunner-{self.args.node_id}")
    
    def initialize_components(self):
        """Initialize all swarm learning components"""
        self.logger.info("Initializing swarm learning components...")
        
        # 1. Initialize blockchain
        self.blockchain = SwarmBlockchain(difficulty=self.args.blockchain_difficulty)
        
        # 2. Initialize swarm node
        self.swarm_node = SwarmNode(
            node_id=self.args.node_id,
            host=self.args.host,
            port=self.args.port + int(self.args.node_id.split('_')[-1]),  # Unique port per node
            blockchain=self.blockchain
        )
        
        # 3. Load data
        self.dataset, output_dim = load_data(self.args)
        
        # 4. Initialize model
        self.model = AutoEncoder(output_dim)
        
        # 5. Initialize trainer
        self.trainer = SwarmTrainer(
            model=self.model,
            node=self.swarm_node,
            device=self.device,
            args=self.args
        )
        
        self.logger.info("All components initialized successfully")
    
    def start_swarm_node(self):
        """Start the swarm node for network communication"""
        self.logger.info("Starting swarm node...")
        self.swarm_node.start()
        
        # Discover peers if bootstrap peers are provided
        if hasattr(self.args, 'bootstrap_peers') and self.args.bootstrap_peers:
            bootstrap_peers = [
                (peer['host'], peer['port']) 
                for peer in self.args.bootstrap_peers
            ]
            self.swarm_node.discover_peers(bootstrap_peers)
            
            # Wait for peer discovery
            time.sleep(5)
            
            network_stats = self.swarm_node.get_network_stats()
            self.logger.info(f"Connected to {network_stats['connected_peers']} peers")
    
    def run_swarm_learning(self):
        """Execute the main swarm learning process"""
        self.logger.info("Starting swarm learning process...")
        
        # Get training data for this node
        train_data = self._get_local_training_data()
        test_data = self._get_local_test_data()
        
        if train_data is None:
            self.logger.error("No training data available for this node")
            return
        
        # Run swarm learning rounds
        for round_num in range(1, self.args.comm_round + 1):
            self.logger.info(f"=== Swarm Learning Round {round_num}/{self.args.comm_round} ===")
            
            # Participate in swarm round
            success = self.trainer.participate_in_swarm_round(train_data, round_num)
            
            if success:
                self.logger.info(f"Round {round_num} completed successfully")
                
                # Evaluate model if test data is available
                if test_data is not None and round_num % self.args.evaluation_frequency == 0:
                    self._evaluate_model(test_data, round_num)
                
                # Log statistics
                self._log_round_statistics(round_num)
                
            else:
                self.logger.warning(f"Round {round_num} failed - no consensus reached")
            
            # Wait between rounds
            if round_num < self.args.comm_round:
                time.sleep(self.args.round_wait_time)
        
        self.logger.info("Swarm learning process completed")
    
    def _get_local_training_data(self):
        """Get training data for the current node"""
        if self.args.rank == 0:
            # Server node - use aggregated data or no training
            return None
        
        node_idx = self.args.rank - 1
        train_data_local_dict = self.dataset[5]  # train_data_local_dict
        
        if node_idx in train_data_local_dict:
            return train_data_local_dict[node_idx]
        
        return None
    
    def _get_local_test_data(self):
        """Get test data for the current node"""
        if self.args.rank == 0:
            # Server node - use global test data
            test_data_local_dict = self.dataset[6]  # test_data_local_dict
            return test_data_local_dict
        
        node_idx = self.args.rank - 1
        test_data_local_dict = self.dataset[6]  # test_data_local_dict
        
        if node_idx in test_data_local_dict:
            return {node_idx: test_data_local_dict[node_idx]}
        
        return None
    
    def _evaluate_model(self, test_data, round_num: int):
        """Evaluate model performance"""
        self.logger.info(f"Evaluating model at round {round_num}")
        
        if isinstance(test_data, dict):
            # Multiple test datasets
            for node_idx, data_loader in test_data.items():
                metrics = self.trainer.evaluate_on_attack_data(data_loader)
                self.logger.info(f"Node {node_idx} - Detection Rate: {metrics.get('detection_rate', 0):.3f}")
        else:
            # Single test dataset
            metrics = self.trainer.evaluate_on_attack_data(test_data)
            self.logger.info(f"Detection Rate: {metrics.get('detection_rate', 0):.3f}")
    
    def _log_round_statistics(self, round_num: int):
        """Log statistics for the current round"""
        stats = self.trainer.get_swarm_statistics()
        
        self.logger.info(f"Round {round_num} Statistics:")
        self.logger.info(f"  - Connected peers: {stats['network_stats']['connected_peers']}")
        self.logger.info(f"  - Blockchain blocks: {stats['blockchain_stats']['total_blocks']}")
        self.logger.info(f"  - Total transactions: {stats['blockchain_stats']['total_transactions']}")
        self.logger.info(f"  - Model parameters: {stats['node_stats']['model_parameters']}")
        
        if 'final_loss' in stats['node_stats']['performance_metrics']:
            loss = stats['node_stats']['performance_metrics']['final_loss']
            self.logger.info(f"  - Final training loss: {loss:.6f}")
    
    def save_final_results(self):
        """Save final model and results"""
        results_dir = os.path.join(self.args.log_file_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save model checkpoint
        model_path = os.path.join(results_dir, f'final_model_{self.args.node_id}.pt')
        self.trainer.save_model_checkpoint(model_path)
        
        # Save blockchain state
        blockchain_path = os.path.join(results_dir, f'blockchain_{self.args.node_id}.json')
        blockchain_stats = self.blockchain.get_chain_stats()
        
        import json
        with open(blockchain_path, 'w') as f:
            json.dump(blockchain_stats, f, indent=2)
        
        # Save final statistics
        stats_path = os.path.join(results_dir, f'final_stats_{self.args.node_id}.json')
        final_stats = self.trainer.get_swarm_statistics()
        
        with open(stats_path, 'w') as f:
            json.dump(final_stats, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_dir}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.swarm_node:
            self.swarm_node.stop()
        self.logger.info("Cleanup completed")
    
    def run(self):
        """Main execution method"""
        try:
            # Initialize all components
            self.initialize_components()
            
            # Start swarm node
            self.start_swarm_node()
            
            # Run swarm learning
            self.run_swarm_learning()
            
            # Save results
            self.save_final_results()
            
        except Exception as e:
            self.logger.error(f"Error in swarm learning execution: {e}")
            raise
        
        finally:
            # Cleanup
            self.cleanup()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_args_from_config(config: Dict[str, Any], node_id: str, rank: int) -> argparse.Namespace:
    """Create arguments namespace from configuration"""
    args = argparse.Namespace()
    
    # Common args
    common_args = config.get('common_args', {})
    for key, value in common_args.items():
        setattr(args, key, value)
    
    # Data args
    data_args = config.get('data_args', {})
    for key, value in data_args.items():
        setattr(args, key, value)
    
    # Model args
    model_args = config.get('model_args', {})
    for key, value in model_args.items():
        setattr(args, key, value)
    
    # Training args
    train_args = config.get('train_args', {})
    for key, value in train_args.items():
        setattr(args, key, value)
    
    # Device args
    device_args = config.get('device_args', {})
    for key, value in device_args.items():
        setattr(args, key, value)
    
    # Swarm-specific args
    swarm_args = config.get('swarm_args', {})
    for key, value in swarm_args.items():
        setattr(args, key, value)
    
    # Tracking args
    tracking_args = config.get('tracking_args', {})
    for key, value in tracking_args.items():
        setattr(args, key, value)
    
    # Set node-specific args
    args.node_id = node_id
    args.rank = rank
    
    # Set default swarm args if not specified
    if not hasattr(args, 'blockchain_difficulty'):
        args.blockchain_difficulty = 2
    if not hasattr(args, 'consensus_wait_time'):
        args.consensus_wait_time = 10
    if not hasattr(args, 'round_wait_time'):
        args.round_wait_time = 5
    if not hasattr(args, 'evaluation_frequency'):
        args.evaluation_frequency = 2
    if not hasattr(args, 'host'):
        args.host = "localhost"
    if not hasattr(args, 'port'):
        args.port = 8000
    if not hasattr(args, 'gpu_id'):
        args.gpu_id = 0
    
    return args


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Swarm Learning for IoT Anomaly Detection")
    parser.add_argument("--cf", type=str, required=True, help="Configuration file path")
    parser.add_argument("--node_id", type=str, required=True, help="Node ID")
    parser.add_argument("--rank", type=int, required=True, help="Node rank (0 for server, 1+ for clients)")
    
    cmd_args = parser.parse_args()
    
    # Load configuration
    config = load_config(cmd_args.cf)
    
    # Create full args from config
    args = create_args_from_config(config, cmd_args.node_id, cmd_args.rank)
    
    # Run swarm learning
    runner = SwarmLearningRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
