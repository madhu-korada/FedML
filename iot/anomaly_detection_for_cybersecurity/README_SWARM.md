# Swarm Learning for IoT Anomaly Detection

This implementation replaces the original federated learning approach with a blockchain-based swarm learning system for IoT anomaly detection using cybersecurity data.

## Overview

Swarm Learning is a decentralized machine learning approach that combines the benefits of federated learning with blockchain technology. Instead of relying on a central server for coordination, nodes in the network use a blockchain ledger to:

- Record model updates and training contributions
- Achieve consensus on the best model through voting mechanisms
- Maintain trust and reputation scores for participating nodes
- Ensure transparency and immutability of the learning process

## Architecture

### Core Components

1. **Blockchain Ledger** (`blockchain/ledger.py`)
   - Manages model update transactions
   - Implements proof-of-work consensus
   - Tracks node reputation and stakes
   - Provides consensus mechanisms

2. **Swarm Node** (`swarm/node.py`)
   - Handles peer-to-peer communication
   - Manages network discovery and connectivity
   - Broadcasts model updates to peers
   - Participates in consensus voting

3. **Swarm Trainer** (`swarm/trainer.py`)
   - Integrates local training with blockchain
   - Aggregates models based on consensus
   - Tracks performance metrics
   - Evaluates anomaly detection capabilities

4. **Main Runner** (`swarm_iot.py`)
   - Orchestrates the entire swarm learning process
   - Manages configuration and initialization
   - Coordinates training rounds and evaluation

## Key Features

### Blockchain Integration
- **Immutable Training Records**: All model updates are recorded on the blockchain
- **Consensus-based Aggregation**: Models are aggregated based on network consensus
- **Reputation System**: Nodes build reputation based on contribution quality
- **Stake-based Voting**: Node influence is weighted by stake and reputation

### Decentralized Architecture
- **No Central Server**: All nodes are peers in the network
- **Fault Tolerance**: System continues operating even if some nodes fail
- **Dynamic Discovery**: Nodes can join and leave the network dynamically
- **Scalable Communication**: Efficient P2P messaging protocol

### Security Features
- **Model Integrity**: Blockchain ensures model updates cannot be tampered with
- **Byzantine Fault Tolerance**: System handles malicious or faulty nodes
- **Trust Metrics**: Reputation system identifies reliable contributors
- **Transparent Process**: All decisions are recorded and verifiable

## Usage

### Quick Start

1. **Run the simulation with default settings:**
   ```bash
   ./run_swarm_simulation.sh 9
   ```

2. **Monitor the progress:**
   ```bash
   tail -f log/swarm/*.log
   ```

3. **Check results:**
   ```bash
   ls results/swarm/
   ```

### Configuration

Edit `config_swarm/swarm_config.yaml` to customize:

- **Blockchain parameters**: difficulty, consensus threshold, mining reward
- **Network settings**: host, port, timeouts, bootstrap peers
- **Training parameters**: epochs, learning rate, batch size
- **Evaluation settings**: frequency, metrics, thresholds

### Individual Node Execution

Run a single node manually:
```bash
python3 swarm_iot.py \
    --cf config_swarm/swarm_config.yaml \
    --node_id node_1 \
    --rank 1
```

## Data Flow

1. **Initialization**
   - Each node loads its local IoT device data
   - Blockchain is initialized with genesis block
   - Nodes discover peers in the network

2. **Training Round**
   - Nodes train local autoencoder models
   - Model updates are broadcast to network
   - Updates are added as transactions to blockchain

3. **Consensus**
   - Nodes participate in consensus mechanism
   - Best models are selected based on performance and reputation
   - Consensus models are aggregated using weighted averaging

4. **Model Update**
   - Each node updates its local model with consensus result
   - Performance is evaluated on attack detection tasks
   - Results are logged and stored

5. **Blockchain Mining**
   - Nodes compete to mine new blocks
   - Successful miners are rewarded with increased stake
   - Blockchain maintains complete training history

## Consensus Mechanism

The swarm learning consensus combines multiple factors:

- **Performance Metrics**: Model accuracy and convergence rate
- **Node Reputation**: Historical contribution quality
- **Stake Weight**: Node's investment in the network
- **Data Size**: Amount of training data contributed

Formula:
```
Weight = (accuracy × convergence × log(data_size) × reputation × stake)
```

Nodes with higher weights have more influence in the consensus.

## Monitoring and Logging

### Log Files
- `log/swarm/coordinator.log`: Main coordination logs
- `log/swarm/node_X.log`: Individual node logs
- `log/swarm/results/`: Final results and statistics

### Blockchain Statistics
- Total blocks mined
- Transaction count
- Active nodes
- Consensus success rate

### Performance Metrics
- Detection rate for each attack type
- False positive rates
- Model convergence metrics
- Network topology changes

## Comparison with Federated Learning

| Aspect | Federated Learning | Swarm Learning |
|--------|-------------------|----------------|
| Coordination | Central server | Blockchain consensus |
| Trust Model | Trust server | Distributed trust |
| Fault Tolerance | Single point of failure | Byzantine fault tolerant |
| Transparency | Limited | Full transparency |
| Scalability | Server bottleneck | P2P scalable |
| Privacy | Server sees aggregates | Cryptographic privacy |

## Attack Detection

The system detects various IoT cybersecurity attacks:

- **Mirai botnet attacks**: DDoS and scanning
- **Gafgyt attacks**: Telnet brute force and flooding
- **Protocol-specific attacks**: TCP/UDP floods, SYN attacks
- **Reconnaissance**: Port scanning and network probing

### Detection Process

1. **Normal Behavior Learning**: Autoencoders learn normal traffic patterns
2. **Anomaly Scoring**: Reconstruction error indicates anomalies
3. **Threshold Adaptation**: Dynamic thresholds based on network consensus
4. **Collaborative Detection**: Shared knowledge improves detection accuracy

## Configuration Parameters

### Blockchain Settings
```yaml
blockchain_difficulty: 2          # Mining difficulty
consensus_threshold: 0.51         # Minimum consensus percentage
mining_reward: 1.0               # Block mining reward
```

### Network Settings
```yaml
host: "localhost"                # Node host address
port: 8000                      # Base port number
consensus_wait_time: 15         # Consensus timeout
peer_timeout: 300               # Peer connection timeout
```

### Training Settings
```yaml
comm_round: 15                  # Number of training rounds
epochs: 3                       # Local training epochs
batch_size: 32                  # Training batch size
learning_rate: 0.001            # Learning rate
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   - Ensure each node uses a unique port
   - Check for other services using the port range

2. **Consensus Failures**
   - Verify minimum number of nodes are running
   - Check network connectivity between nodes
   - Ensure nodes are using compatible configurations

3. **Blockchain Sync Issues**
   - Allow time for initial blockchain synchronization
   - Check for sufficient disk space
   - Verify all nodes have same genesis block

4. **Performance Issues**
   - Reduce batch size for memory constraints
   - Adjust consensus wait times for slower networks
   - Monitor CPU and memory usage

### Debug Mode

Enable detailed logging by setting log level to DEBUG in the configuration.

## Future Enhancements

- **Encryption**: Add model encryption for enhanced privacy
- **Differential Privacy**: Implement DP for formal privacy guarantees
- **Advanced Consensus**: Implement more sophisticated consensus algorithms
- **Cross-Chain Integration**: Support multiple blockchain networks
- **Real-time Deployment**: Add support for real-time IoT device deployment

## Dependencies

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Pandas
- PyYAML
- Standard library modules (socket, threading, hashlib, etc.)

## License

This implementation is part of the FedML project and follows the same licensing terms.
