#!/usr/bin/env python3
"""
Simple Demo of Swarm Learning Components
This script demonstrates the core blockchain and swarm learning functionality
without requiring the full dataset or network setup.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from blockchain.ledger import SwarmBlockchain, ModelUpdate
from swarm.node import SwarmNode


def create_dummy_model_weights():
    """Create dummy model weights for demonstration"""
    return {
        'layer1.weight': torch.randn(10, 5),
        'layer1.bias': torch.randn(10),
        'layer2.weight': torch.randn(1, 10),
        'layer2.bias': torch.randn(1)
    }


def demo_blockchain():
    """Demonstrate blockchain functionality"""
    print("=== Blockchain Demo ===")
    
    # Create blockchain
    blockchain = SwarmBlockchain(difficulty=1)  # Low difficulty for demo
    
    # Create some model updates
    for i in range(3):
        model_weights = create_dummy_model_weights()
        serialized_weights = blockchain.serialize_model_weights(model_weights)
        
        update = ModelUpdate(
            node_id=f"demo_node_{i}",
            model_weights=serialized_weights,
            performance_metrics={'accuracy': 0.8 + i * 0.05, 'loss': 0.3 - i * 0.05},
            training_data_size=1000 + i * 200,
            timestamp=time.time(),
            round_number=1
        )
        
        success = blockchain.add_transaction(update)
        print(f"Added transaction from node_{i}: {success}")
    
    # Mine a block
    print("Mining block...")
    block = blockchain.mine_block("demo_miner")
    if block:
        print(f"Successfully mined block {block.index} with hash: {block.hash[:10]}...")
    
    # Demonstrate consensus
    print("Checking consensus...")
    consensus = blockchain.reach_consensus(1)
    if consensus:
        print(f"Consensus reached with {len(consensus['consensus_updates'])} updates")
        for update in consensus['consensus_updates']:
            print(f"  - Node {update.node_id}: accuracy={update.performance_metrics.get('accuracy', 0):.3f}")
    
    # Show blockchain stats
    stats = blockchain.get_chain_stats()
    print(f"Blockchain stats: {stats}")
    print()


def demo_swarm_node():
    """Demonstrate swarm node functionality (without actual networking)"""
    print("=== Swarm Node Demo ===")
    
    # Create blockchain and node
    blockchain = SwarmBlockchain(difficulty=1)
    node = SwarmNode("demo_node", "localhost", 9999, blockchain)
    
    # Simulate model update broadcast
    model_state = create_dummy_model_weights()
    performance_metrics = {
        'accuracy': 0.85,
        'loss': 0.25,
        'convergence_rate': 0.8
    }
    
    print("Broadcasting model update...")
    node.broadcast_model_update(model_state, performance_metrics, 1500)
    
    # Check node stats
    stats = node.get_network_stats()
    print(f"Node stats: {stats}")
    print()


def demo_model_aggregation():
    """Demonstrate model aggregation functionality"""
    print("=== Model Aggregation Demo ===")
    
    blockchain = SwarmBlockchain(difficulty=1)
    
    # Create multiple model updates with different performance
    models = []
    for i in range(3):
        model_weights = create_dummy_model_weights()
        serialized_weights = blockchain.serialize_model_weights(model_weights)
        
        # Vary performance metrics
        accuracy = 0.7 + i * 0.1
        convergence = 0.6 + i * 0.15
        
        update = ModelUpdate(
            node_id=f"node_{i}",
            model_weights=serialized_weights,
            performance_metrics={
                'accuracy_estimate': accuracy,
                'convergence_rate': convergence,
                'final_loss': 0.4 - i * 0.1
            },
            training_data_size=1000 + i * 300,
            timestamp=time.time(),
            round_number=1
        )
        
        blockchain.add_transaction(update)
        models.append((model_weights, update))
    
    # Mine block and get consensus
    blockchain.mine_block("aggregator")
    consensus = blockchain.reach_consensus(1)
    
    if consensus:
        print("Model aggregation simulation:")
        updates = consensus['consensus_updates']
        
        # Show weights for each model (first parameter only for brevity)
        for i, update in enumerate(updates):
            model_state = blockchain.deserialize_model_weights(update.model_weights)
            first_param = model_state['layer1.weight'][0, 0].item()
            accuracy = update.performance_metrics.get('accuracy_estimate', 0)
            print(f"  Model {i}: first_param={first_param:.3f}, accuracy={accuracy:.3f}")
        
        # Simulate weighted aggregation
        weights = []
        for update in updates:
            accuracy = update.performance_metrics.get('accuracy_estimate', 0.5)
            convergence = update.performance_metrics.get('convergence_rate', 0.5)
            data_size = update.training_data_size
            
            weight = accuracy * convergence * np.log(1 + data_size)
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        print(f"  Aggregation weights: {weights}")
        
        # Show weighted first parameter
        weighted_param = 0
        for i, update in enumerate(updates):
            model_state = blockchain.deserialize_model_weights(update.model_weights)
            first_param = model_state['layer1.weight'][0, 0].item()
            weighted_param += first_param * weights[i]
        
        print(f"  Weighted aggregated first parameter: {weighted_param:.3f}")
    
    print()


def demo_reputation_system():
    """Demonstrate node reputation tracking"""
    print("=== Reputation System Demo ===")
    
    blockchain = SwarmBlockchain(difficulty=1)
    
    # Simulate multiple rounds of updates from different nodes
    nodes = ['node_A', 'node_B', 'node_C']
    
    for round_num in range(1, 4):
        print(f"Round {round_num}:")
        
        for i, node_id in enumerate(nodes):
            # Vary performance: node_A improves, node_B is consistent, node_C degrades
            if node_id == 'node_A':
                accuracy = 0.6 + round_num * 0.1
            elif node_id == 'node_B':
                accuracy = 0.8
            else:  # node_C
                accuracy = 0.9 - round_num * 0.1
            
            model_weights = create_dummy_model_weights()
            serialized_weights = blockchain.serialize_model_weights(model_weights)
            
            update = ModelUpdate(
                node_id=node_id,
                model_weights=serialized_weights,
                performance_metrics={'accuracy': accuracy},
                training_data_size=1000,
                timestamp=time.time(),
                round_number=round_num
            )
            
            blockchain.add_transaction(update)
        
        # Mine block
        blockchain.mine_block(f"miner_round_{round_num}")
        
        # Show reputations
        for node_id in nodes:
            reputation = blockchain.get_node_reputation(node_id)
            print(f"  {node_id} reputation: {reputation:.3f}")
        
        print()


def main():
    """Run all demonstrations"""
    print("Swarm Learning Blockchain Demo")
    print("==============================")
    print()
    
    demo_blockchain()
    demo_swarm_node()
    demo_model_aggregation()
    demo_reputation_system()
    
    print("Demo completed successfully!")
    print("To run the full simulation: ./run_swarm_simulation.sh 9")


if __name__ == "__main__":
    main()
