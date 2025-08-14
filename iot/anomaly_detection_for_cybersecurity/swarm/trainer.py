import logging
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any
from collections import OrderedDict

from blockchain.ledger import SwarmBlockchain, ModelUpdate
from swarm.node import SwarmNode


class SwarmTrainer:
    """
    Swarm Learning Trainer for Anomaly Detection
    Integrates with blockchain for decentralized model training and consensus
    """
    
    def __init__(self, 
                 model: nn.Module,
                 node: SwarmNode,
                 device: torch.device,
                 args: Any):
        self.model = model
        self.node = node
        self.device = device
        self.args = args
        
        # Training state
        self.local_model_state = None
        self.training_history = []
        self.current_round = 0
        self.consensus_threshold = 0.7
        
        # Performance tracking
        self.performance_metrics = {}
        
        # Setup logging
        self.logger = logging.getLogger(f"SwarmTrainer-{node.node_id}")
        
    def train_local_model(self, train_data, epochs: int = None) -> Dict[str, float]:
        """
        Train local model on device-specific data
        Returns performance metrics
        """
        if epochs is None:
            epochs = self.args.epochs
            
        self.model.to(self.device)
        self.model.train()
        
        # Setup training
        criterion = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.learning_rate
        )
        
        epoch_losses = []
        training_start_time = time.time()
        
        self.logger.info(f"Starting local training for {epochs} epochs")
        
        for epoch in range(epochs):
            batch_losses = []
            
            for batch_idx, x in enumerate(train_data):
                x = x.to(self.device).float()
                
                optimizer.zero_grad()
                decoded = self.model(x)
                loss = criterion(decoded, x)
                loss.backward()
                optimizer.step()
                
                batch_losses.append(loss.item())
            
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)
            
            self.logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")
        
        training_time = time.time() - training_start_time
        
        # Calculate performance metrics
        final_loss = sum(epoch_losses[-3:]) / min(3, len(epoch_losses))  # Average of last 3 epochs
        convergence_rate = self._calculate_convergence_rate(epoch_losses)
        
        metrics = {
            'final_loss': final_loss,
            'convergence_rate': convergence_rate,
            'training_time': training_time,
            'epochs_trained': epochs,
            'data_size': len(train_data) * self.args.batch_size
        }
        
        # Store local model state
        self.local_model_state = self.model.cpu().state_dict()
        self.performance_metrics = metrics
        
        self.logger.info(f"Local training completed. Final loss: {final_loss:.6f}")
        
        return metrics
    
    def _calculate_convergence_rate(self, losses: List[float]) -> float:
        """Calculate convergence rate based on loss reduction"""
        if len(losses) < 2:
            return 0.0
        
        # Calculate rate of loss reduction
        initial_loss = losses[0]
        final_loss = losses[-1]
        
        if initial_loss == 0:
            return 0.0
        
        improvement = (initial_loss - final_loss) / initial_loss
        return max(0.0, min(1.0, improvement))  # Clamp between 0 and 1
    
    def participate_in_swarm_round(self, train_data, round_number: int) -> bool:
        """
        Participate in a complete swarm learning round
        """
        self.current_round = round_number
        self.node.current_round = round_number
        
        self.logger.info(f"Starting swarm learning round {round_number}")
        
        # Step 1: Train local model
        metrics = self.train_local_model(train_data)
        
        # Step 2: Broadcast model update to network
        self.broadcast_model_update(metrics)
        
        # Step 3: Wait for other nodes and attempt consensus
        time.sleep(self.args.consensus_wait_time)  # Wait for other nodes
        
        # Step 4: Participate in consensus
        consensus_result = self.participate_in_consensus()
        
        # Step 5: Update local model with consensus if successful
        if consensus_result:
            self.update_model_from_consensus(consensus_result)
            return True
        else:
            self.logger.warning(f"No consensus reached for round {round_number}")
            return False
    
    def broadcast_model_update(self, metrics: Dict[str, float]):
        """Broadcast local model update to the swarm network"""
        if self.local_model_state is None:
            self.logger.error("No local model state to broadcast")
            return
        
        # Add additional metrics
        enhanced_metrics = metrics.copy()
        enhanced_metrics.update({
            'node_id': self.node.node_id,
            'round_number': self.current_round,
            'model_size': self._calculate_model_size(),
            'accuracy_estimate': self._estimate_accuracy()
        })
        
        # Broadcast through swarm node
        self.node.broadcast_model_update(
            model_state=self.local_model_state,
            performance_metrics=enhanced_metrics,
            training_data_size=metrics['data_size']
        )
        
        self.logger.info("Broadcasted model update to swarm network")
    
    def participate_in_consensus(self) -> Optional[Dict]:
        """Participate in consensus mechanism"""
        # Request consensus from blockchain
        consensus_result = self.node.request_consensus()
        
        if consensus_result:
            consensus_updates = consensus_result['consensus_updates']
            total_weight = consensus_result['total_weight']
            consensus_weight = consensus_result['consensus_weight']
            
            self.logger.info(f"Consensus reached with {len(consensus_updates)} updates")
            self.logger.info(f"Consensus weight: {consensus_weight:.2f}/{total_weight:.2f}")
            
            return consensus_result
        
        return None
    
    def update_model_from_consensus(self, consensus_result: Dict):
        """Update local model using consensus from swarm"""
        consensus_updates = consensus_result['consensus_updates']
        
        if not consensus_updates:
            return
        
        # Aggregate models using weighted average
        aggregated_state = self._aggregate_model_states(consensus_updates)
        
        if aggregated_state:
            # Update local model
            self.model.load_state_dict(aggregated_state)
            self.local_model_state = aggregated_state
            
            self.logger.info("Updated local model from swarm consensus")
        else:
            self.logger.warning("Failed to aggregate consensus models")
    
    def _aggregate_model_states(self, updates: List[ModelUpdate]) -> Optional[OrderedDict]:
        """Aggregate multiple model states using weighted averaging"""
        try:
            # Calculate weights based on performance and reputation
            weights = []
            model_states = []
            
            for update in updates:
                # Deserialize model weights
                model_state = self.node.blockchain.deserialize_model_weights(update.model_weights)
                model_states.append(model_state)
                
                # Calculate weight based on performance metrics
                accuracy = update.performance_metrics.get('accuracy_estimate', 0.5)
                convergence = update.performance_metrics.get('convergence_rate', 0.5)
                data_size = update.training_data_size
                reputation = self.node.blockchain.get_node_reputation(update.node_id)
                
                weight = accuracy * convergence * np.log(1 + data_size) * reputation
                weights.append(weight)
            
            if not weights or sum(weights) == 0:
                return None
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Aggregate model parameters
            aggregated_state = OrderedDict()
            
            # Get parameter names from first model
            param_names = list(model_states[0].keys())
            
            for param_name in param_names:
                # Weighted average of parameters
                weighted_params = []
                
                for i, model_state in enumerate(model_states):
                    param = model_state[param_name]
                    weighted_param = param * weights[i]
                    weighted_params.append(weighted_param)
                
                # Sum weighted parameters
                aggregated_param = torch.stack(weighted_params).sum(dim=0)
                aggregated_state[param_name] = aggregated_param
            
            self.logger.info(f"Aggregated {len(model_states)} models with weights: {weights}")
            
            return aggregated_state
            
        except Exception as e:
            self.logger.error(f"Error aggregating model states: {e}")
            return None
    
    def _calculate_model_size(self) -> int:
        """Calculate model size in parameters"""
        if self.local_model_state is None:
            return 0
        
        total_params = 0
        for param_tensor in self.local_model_state.values():
            total_params += param_tensor.numel()
        
        return total_params
    
    def _estimate_accuracy(self) -> float:
        """Estimate model accuracy based on reconstruction loss"""
        # Convert loss to accuracy estimate (inverse relationship)
        if 'final_loss' in self.performance_metrics:
            loss = self.performance_metrics['final_loss']
            # Simple conversion: higher loss = lower accuracy
            accuracy = max(0.0, min(1.0, 1.0 - loss))
            return accuracy
        return 0.5  # Default neutral accuracy
    
    def evaluate_on_attack_data(self, test_data) -> Dict[str, float]:
        """
        Evaluate model on attack data for anomaly detection
        """
        if self.local_model_state is None:
            self.logger.warning("No model state available for evaluation")
            return {}
        
        self.model.load_state_dict(self.local_model_state)
        self.model.to(self.device)
        self.model.eval()
        
        threshold_func = nn.MSELoss(reduction='none')
        
        # Calculate threshold using current model on normal data
        threshold = self._calculate_detection_threshold()
        
        true_positive = 0
        false_negative = 0
        total_attacks = 0
        
        with torch.no_grad():
            for batch_idx, x in enumerate(test_data):
                x = x.to(self.device).float()
                
                # Get reconstruction
                reconstructed = self.model(x)
                
                # Calculate reconstruction error
                errors = threshold_func(reconstructed, x)
                mse_per_sample = errors.mean(dim=1)
                
                # Classify as anomaly if error > threshold
                anomalies = mse_per_sample > threshold
                
                # Count detections (all test data are attacks)
                true_positive += anomalies.sum().item()
                false_negative += (~anomalies).sum().item()
                total_attacks += x.size(0)
        
        # Calculate metrics
        detection_rate = true_positive / total_attacks if total_attacks > 0 else 0.0
        miss_rate = false_negative / total_attacks if total_attacks > 0 else 0.0
        
        evaluation_metrics = {
            'detection_rate': detection_rate,
            'miss_rate': miss_rate,
            'true_positive': true_positive,
            'false_negative': false_negative,
            'total_attacks': total_attacks,
            'threshold': threshold.item()
        }
        
        self.logger.info(f"Evaluation - Detection Rate: {detection_rate:.3f}, Miss Rate: {miss_rate:.3f}")
        
        return evaluation_metrics
    
    def _calculate_detection_threshold(self) -> torch.Tensor:
        """Calculate threshold for anomaly detection"""
        # Use mean + 3*std of reconstruction errors on normal data
        # This is a simplified version - in practice you'd use validation data
        
        # Default threshold based on typical autoencoder behavior
        return torch.tensor(0.1)  # Adjust based on your data characteristics
    
    def get_training_history(self) -> List[Dict]:
        """Get training history for analysis"""
        return self.training_history
    
    def save_model_checkpoint(self, filepath: str):
        """Save current model state"""
        if self.local_model_state:
            torch.save({
                'model_state_dict': self.local_model_state,
                'performance_metrics': self.performance_metrics,
                'current_round': self.current_round,
                'node_id': self.node.node_id
            }, filepath)
            
            self.logger.info(f"Saved model checkpoint to {filepath}")
    
    def load_model_checkpoint(self, filepath: str):
        """Load model state from checkpoint"""
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            self.local_model_state = checkpoint['model_state_dict']
            self.performance_metrics = checkpoint.get('performance_metrics', {})
            self.current_round = checkpoint.get('current_round', 0)
            
            # Load into model
            self.model.load_state_dict(self.local_model_state)
            
            self.logger.info(f"Loaded model checkpoint from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
    
    def get_swarm_statistics(self) -> Dict[str, Any]:
        """Get comprehensive swarm learning statistics"""
        network_stats = self.node.get_network_stats()
        blockchain_stats = self.node.blockchain.get_chain_stats()
        
        return {
            'node_stats': {
                'node_id': self.node.node_id,
                'current_round': self.current_round,
                'model_parameters': self._calculate_model_size(),
                'performance_metrics': self.performance_metrics
            },
            'network_stats': network_stats,
            'blockchain_stats': blockchain_stats,
            'training_history': len(self.training_history)
        }
