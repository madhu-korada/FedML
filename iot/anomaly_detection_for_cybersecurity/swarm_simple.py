#!/usr/bin/env python3
"""
Simplified Swarm Learning for IoT Anomaly Detection
Version without PyTorch dependencies for initial testing
"""

import argparse
import logging
import os
import sys
import time
import yaml
import json
import numpy as np
from typing import Dict, List, Any

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from blockchain.ledger import SwarmBlockchain, ModelUpdate
    from swarm.node import SwarmNode
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative imports...")
    # Alternative import method
    import importlib.util
    
    # Import blockchain module
    blockchain_spec = importlib.util.spec_from_file_location(
        "ledger", 
        os.path.join(os.path.dirname(__file__), "blockchain", "ledger.py")
    )
    blockchain_module = importlib.util.module_from_spec(blockchain_spec)
    blockchain_spec.loader.exec_module(blockchain_module)
    SwarmBlockchain = blockchain_module.SwarmBlockchain
    ModelUpdate = blockchain_module.ModelUpdate
    
    # Import swarm node module
    node_spec = importlib.util.spec_from_file_location(
        "node", 
        os.path.join(os.path.dirname(__file__), "swarm", "node.py")
    )
    node_module = importlib.util.module_from_spec(node_spec)
    node_spec.loader.exec_module(node_module)
    SwarmNode = node_module.SwarmNode


class SimpleSwarmRunner:
    """
    Simplified Swarm Learning Runner for testing
    """
    
    def __init__(self, args):
        self.args = args
        self.logger = self._setup_logging()
        
        # Initialize components
        self.blockchain = None
        self.swarm_node = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_dir = getattr(self.args, 'log_file_dir', './log/swarm')
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, f'simple_node_{self.args.node_id}.log')),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(f"SimpleSwarmRunner-{self.args.node_id}")
    
    def initialize_components(self):
        """Initialize swarm learning components"""
        self.logger.info("Initializing simplified swarm learning components...")
        
        try:
            # 1. Initialize blockchain
            difficulty = getattr(self.args, 'blockchain_difficulty', 2)
            self.blockchain = SwarmBlockchain(difficulty=difficulty)
            self.logger.info("Blockchain initialized successfully")
            
            # 2. Initialize swarm node
            host = getattr(self.args, 'host', 'localhost')
            base_port = getattr(self.args, 'port', 8000)
            
            # Calculate unique port for this node
            if self.args.node_id == "coordinator":
                port = base_port
            else:
                node_num = int(self.args.node_id.split('_')[-1]) if '_' in self.args.node_id else self.args.rank
                port = base_port + node_num
            
            self.swarm_node = SwarmNode(
                node_id=self.args.node_id,
                host=host,
                port=port,
                blockchain=self.blockchain
            )
            self.logger.info(f"Swarm node initialized on {host}:{port}")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def start_swarm_node(self):
        """Start the swarm node"""
        try:
            self.logger.info("Starting swarm node...")
            self.swarm_node.start()
            
            # Add bootstrap peers if provided
            if hasattr(self.args, 'bootstrap_peers') and self.args.bootstrap_peers:
                time.sleep(2)  # Give server time to start
                bootstrap_peers = [
                    (peer['host'], peer['port']) 
                    for peer in self.args.bootstrap_peers
                    if peer['port'] != self.swarm_node.port  # Don't connect to self
                ]
                
                if bootstrap_peers:
                    self.logger.info(f"Discovering peers: {bootstrap_peers}")
                    self.swarm_node.discover_peers(bootstrap_peers)
                    time.sleep(3)  # Wait for discovery
            
            # Log network stats
            network_stats = self.swarm_node.get_network_stats()
            self.logger.info(f"Network stats: {network_stats}")
            
        except Exception as e:
            self.logger.error(f"Error starting swarm node: {e}")
            raise
    
    def run_simulation(self):
        """Run a simplified simulation"""
        self.logger.info("Starting simplified swarm simulation...")
        
        try:
            # Simulate some rounds
            rounds = getattr(self.args, 'comm_round', 5)
            
            for round_num in range(1, rounds + 1):
                self.logger.info(f"=== Round {round_num}/{rounds} ===")
                
                # Simulate model update
                self._simulate_model_update(round_num)
                
                # Wait between rounds
                time.sleep(2)
                
                # Show blockchain stats
                stats = self.blockchain.get_chain_stats()
                self.logger.info(f"Blockchain stats: {stats}")
                
                # Try consensus every few rounds
                if round_num % 2 == 0:
                    self._attempt_consensus(round_num)
            
            self.logger.info("Simulation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in simulation: {e}")
            raise
    
    def _simulate_model_update(self, round_num: int):
        """Simulate a model update"""
        try:
            # Create dummy model weights
            dummy_weights = {
                'layer1': np.random.randn(10, 5).tolist(),
                'layer2': np.random.randn(5, 1).tolist()
            }
            
            # Serialize weights
            serialized_weights = self.blockchain.serialize_model_weights(dummy_weights)
            
            # Create performance metrics
            performance = {
                'accuracy': 0.7 + np.random.random() * 0.2,
                'loss': 0.1 + np.random.random() * 0.3,
                'convergence_rate': 0.5 + np.random.random() * 0.4
            }
            
            # Broadcast update
            self.swarm_node.broadcast_model_update(
                model_state=dummy_weights,
                performance_metrics=performance,
                training_data_size=1000 + np.random.randint(0, 500)
            )
            
            self.logger.info(f"Model update broadcasted for round {round_num}")
            
        except Exception as e:
            self.logger.error(f"Error in model update: {e}")
    
    def _attempt_consensus(self, round_num: int):
        """Attempt to reach consensus"""
        try:
            self.logger.info(f"Attempting consensus for round {round_num}")
            
            # Wait for transactions
            time.sleep(1)
            
            # Try consensus
            consensus = self.swarm_node.request_consensus()
            
            if consensus:
                self.logger.info(f"Consensus reached with {len(consensus['consensus_updates'])} updates")
            else:
                self.logger.info("No consensus reached")
                
        except Exception as e:
            self.logger.error(f"Error in consensus: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.swarm_node:
                self.swarm_node.stop()
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Error in cleanup: {e}")
    
    def run(self):
        """Main execution method"""
        try:
            self.logger.info(f"Starting Simple Swarm Runner for node {self.args.node_id}")
            
            # Initialize components
            self.initialize_components()
            
            # Start swarm node
            self.start_swarm_node()
            
            # Run simulation
            self.run_simulation()
            
        except Exception as e:
            self.logger.error(f"Error in execution: {e}")
            raise
        
        finally:
            # Cleanup
            self.cleanup()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        # Return default config
        return {
            'swarm_args': {
                'blockchain_difficulty': 1,
                'host': 'localhost',
                'port': 8000,
                'bootstrap_peers': []
            },
            'train_args': {
                'comm_round': 5
            },
            'tracking_args': {
                'log_file_dir': './log/swarm'
            }
        }


def create_args_from_config(config: Dict[str, Any], node_id: str, rank: int) -> argparse.Namespace:
    """Create arguments namespace from configuration"""
    args = argparse.Namespace()
    
    # Set node-specific args
    args.node_id = node_id
    args.rank = rank
    
    # Swarm args
    swarm_args = config.get('swarm_args', {})
    for key, value in swarm_args.items():
        setattr(args, key, value)
    
    # Training args
    train_args = config.get('train_args', {})
    for key, value in train_args.items():
        setattr(args, key, value)
    
    # Tracking args
    tracking_args = config.get('tracking_args', {})
    for key, value in tracking_args.items():
        setattr(args, key, value)
    
    # Set defaults
    if not hasattr(args, 'blockchain_difficulty'):
        args.blockchain_difficulty = 1
    if not hasattr(args, 'host'):
        args.host = "localhost"
    if not hasattr(args, 'port'):
        args.port = 8000
    if not hasattr(args, 'comm_round'):
        args.comm_round = 5
    if not hasattr(args, 'log_file_dir'):
        args.log_file_dir = './log/swarm'
    
    return args


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Simple Swarm Learning Test")
    parser.add_argument("--cf", type=str, required=True, help="Configuration file path")
    parser.add_argument("--node_id", type=str, required=True, help="Node ID")
    parser.add_argument("--rank", type=int, required=True, help="Node rank")
    
    cmd_args = parser.parse_args()
    
    print(f"Starting node {cmd_args.node_id} with rank {cmd_args.rank}")
    
    # Load configuration
    config = load_config(cmd_args.cf)
    
    # Create full args from config
    args = create_args_from_config(config, cmd_args.node_id, cmd_args.rank)
    
    # Run simple swarm learning
    runner = SimpleSwarmRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
