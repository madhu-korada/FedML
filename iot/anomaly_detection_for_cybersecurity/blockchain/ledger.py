import hashlib
import json
import time
import pickle
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Handle torch import gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, using basic serialization")


@dataclass
class ModelUpdate:
    """Represents a model update transaction"""
    node_id: str
    model_weights: bytes  # Serialized model weights
    performance_metrics: Dict[str, float]
    training_data_size: int
    timestamp: float
    round_number: int
    signature: Optional[str] = None


@dataclass
class Block:
    """Represents a block in the blockchain"""
    index: int
    timestamp: float
    transactions: List[ModelUpdate]
    previous_hash: str
    merkle_root: str
    nonce: int = 0
    hash: Optional[str] = None


class SwarmBlockchain:
    """
    Blockchain implementation for swarm learning
    Stores model updates, tracks consensus, and maintains integrity
    """
    
    def __init__(self, difficulty: int = 2):
        self.chain: List[Block] = []
        self.pending_transactions: List[ModelUpdate] = []
        self.difficulty = difficulty
        self.mining_reward = 1.0
        self.consensus_threshold = 0.51  # 51% consensus required
        self.node_stakes: Dict[str, float] = defaultdict(float)
        self.create_genesis_block()
    
    def create_genesis_block(self) -> None:
        """Create the first block in the blockchain"""
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            transactions=[],
            previous_hash="0",
            merkle_root="0",
            nonce=0
        )
        genesis_block.hash = self.calculate_hash(genesis_block)
        self.chain.append(genesis_block)
    
    def calculate_hash(self, block: Block) -> str:
        """Calculate SHA-256 hash of a block"""
        block_string = json.dumps({
            'index': block.index,
            'timestamp': block.timestamp,
            'transactions': [asdict(tx) for tx in block.transactions],
            'previous_hash': block.previous_hash,
            'merkle_root': block.merkle_root,
            'nonce': block.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def calculate_merkle_root(self, transactions: List[ModelUpdate]) -> str:
        """Calculate Merkle root of transactions"""
        if not transactions:
            return "0"
        
        tx_hashes = []
        for tx in transactions:
            tx_string = json.dumps(asdict(tx), sort_keys=True)
            tx_hashes.append(hashlib.sha256(tx_string.encode()).hexdigest())
        
        while len(tx_hashes) > 1:
            new_hashes = []
            for i in range(0, len(tx_hashes), 2):
                if i + 1 < len(tx_hashes):
                    combined = tx_hashes[i] + tx_hashes[i + 1]
                else:
                    combined = tx_hashes[i] + tx_hashes[i]
                new_hashes.append(hashlib.sha256(combined.encode()).hexdigest())
            tx_hashes = new_hashes
        
        return tx_hashes[0]
    
    def add_transaction(self, transaction: ModelUpdate) -> bool:
        """Add a model update transaction to pending pool"""
        # Validate transaction
        if self.validate_transaction(transaction):
            self.pending_transactions.append(transaction)
            return True
        return False
    
    def validate_transaction(self, transaction: ModelUpdate) -> bool:
        """Validate a model update transaction"""
        # Check if node exists and has sufficient stake
        if transaction.node_id not in self.node_stakes:
            self.node_stakes[transaction.node_id] = 1.0  # Initial stake
        
        # Check performance metrics are reasonable
        if 'accuracy' in transaction.performance_metrics:
            if not (0.0 <= transaction.performance_metrics['accuracy'] <= 1.0):
                return False
        
        # Check model weights are not empty
        if len(transaction.model_weights) == 0:
            return False
        
        return True
    
    def mine_block(self, miner_id: str) -> Block:
        """Mine a new block with pending transactions"""
        if not self.pending_transactions:
            return None
        
        # Create new block
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            previous_hash=self.chain[-1].hash,
            merkle_root=self.calculate_merkle_root(self.pending_transactions)
        )
        
        # Proof of Work
        new_block = self.proof_of_work(new_block)
        
        # Add block to chain
        if self.validate_block(new_block):
            self.chain.append(new_block)
            # Reward miner
            self.node_stakes[miner_id] += self.mining_reward
            # Clear pending transactions
            self.pending_transactions = []
            return new_block
        
        return None
    
    def proof_of_work(self, block: Block) -> Block:
        """Simple Proof of Work implementation"""
        target = "0" * self.difficulty
        
        while not block.hash or not block.hash.startswith(target):
            block.nonce += 1
            block.hash = self.calculate_hash(block)
        
        return block
    
    def validate_block(self, block: Block) -> bool:
        """Validate a block before adding to chain"""
        # Check if hash is correct
        if block.hash != self.calculate_hash(block):
            return False
        
        # Check if previous hash matches
        if block.previous_hash != self.chain[-1].hash:
            return False
        
        # Check proof of work
        if not block.hash.startswith("0" * self.difficulty):
            return False
        
        # Validate all transactions in block
        for tx in block.transactions:
            if not self.validate_transaction(tx):
                return False
        
        return True
    
    def get_latest_model_updates(self, round_number: int) -> List[ModelUpdate]:
        """Get all model updates for a specific round"""
        updates = []
        for block in self.chain:
            for tx in block.transactions:
                if tx.round_number == round_number:
                    updates.append(tx)
        return updates
    
    def get_node_reputation(self, node_id: str) -> float:
        """Calculate node reputation based on contribution history"""
        total_contributions = 0
        successful_contributions = 0
        
        for block in self.chain:
            for tx in block.transactions:
                if tx.node_id == node_id:
                    total_contributions += 1
                    # Consider contribution successful if performance is above threshold
                    if tx.performance_metrics.get('accuracy', 0) > 0.7:
                        successful_contributions += 1
        
        if total_contributions == 0:
            return 0.5  # Default reputation for new nodes
        
        reputation = successful_contributions / total_contributions
        return reputation
    
    def reach_consensus(self, round_number: int) -> Optional[Dict[str, Any]]:
        """Reach consensus on the best model for current round"""
        updates = self.get_latest_model_updates(round_number)
        
        if not updates:
            return None
        
        # Weight updates by node reputation and stake
        weighted_updates = []
        total_weight = 0
        
        for update in updates:
            reputation = self.get_node_reputation(update.node_id)
            stake = self.node_stakes[update.node_id]
            weight = reputation * stake * update.training_data_size
            
            weighted_updates.append({
                'update': update,
                'weight': weight
            })
            total_weight += weight
        
        # Select updates that represent majority consensus
        consensus_updates = []
        cumulative_weight = 0
        
        # Sort by performance (e.g., accuracy)
        weighted_updates.sort(
            key=lambda x: x['update'].performance_metrics.get('accuracy', 0),
            reverse=True
        )
        
        for weighted_update in weighted_updates:
            cumulative_weight += weighted_update['weight']
            consensus_updates.append(weighted_update['update'])
            
            if cumulative_weight / total_weight >= self.consensus_threshold:
                break
        
        return {
            'consensus_updates': consensus_updates,
            'total_weight': total_weight,
            'consensus_weight': cumulative_weight
        }
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        total_transactions = sum(len(block.transactions) for block in self.chain)
        active_nodes = len(self.node_stakes)
        
        return {
            'total_blocks': len(self.chain),
            'total_transactions': total_transactions,
            'active_nodes': active_nodes,
            'difficulty': self.difficulty,
            'latest_block_hash': self.chain[-1].hash if self.chain else None
        }
    
    def serialize_model_weights(self, model_state_dict: Dict) -> bytes:
        """Serialize model weights for storage in blockchain"""
        if TORCH_AVAILABLE and any(str(type(v)).startswith("<class 'torch") for v in model_state_dict.values()):
            # Convert torch tensors to lists for serialization
            serializable_dict = {}
            for key, value in model_state_dict.items():
                if hasattr(value, 'detach'):
                    serializable_dict[key] = value.detach().cpu().numpy().tolist()
                else:
                    serializable_dict[key] = value
            return pickle.dumps(serializable_dict)
        else:
            return pickle.dumps(model_state_dict)
    
    def deserialize_model_weights(self, weights_bytes: bytes) -> Dict:
        """Deserialize model weights from blockchain storage"""
        model_dict = pickle.loads(weights_bytes)
        
        if TORCH_AVAILABLE:
            # Convert back to torch tensors if torch is available
            torch_dict = {}
            for key, value in model_dict.items():
                if isinstance(value, list):
                    torch_dict[key] = torch.tensor(value)
                else:
                    torch_dict[key] = value
            return torch_dict
        else:
            return model_dict
