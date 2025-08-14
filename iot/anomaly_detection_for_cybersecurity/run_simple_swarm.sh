#!/usr/bin/env bash

# Simple Swarm Learning Simulation Script
# This version uses the simplified swarm implementation without PyTorch dependencies

set -e  # Exit on any error

# Configuration
WORKER_NUM=${1:-3}  # Number of worker nodes (default: 3 for testing)
CONFIG_FILE="config_swarm/swarm_config.yaml"
LOG_DIR="./log/swarm"
RESULTS_DIR="./results/swarm"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   Simple Swarm Learning Test Simulation   ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Validate parameters
if [ "$WORKER_NUM" -lt 2 ]; then
    echo -e "${RED}Error: Minimum 2 nodes required for testing${NC}"
    echo "Usage: $0 [number_of_nodes]"
    echo "Example: $0 3"
    exit 1
fi

# Setup directories
echo -e "${BLUE}Setting up directories...${NC}"
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Check if config file exists, create simple one if not
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}Config file not found, creating simple config...${NC}"
    mkdir -p "$(dirname "$CONFIG_FILE")"
    cat > "$CONFIG_FILE" << EOF
swarm_args:
  blockchain_difficulty: 1
  host: "localhost"
  port: 8000
  consensus_wait_time: 5
  round_wait_time: 2
  bootstrap_peers:
    - host: "localhost"
      port: 8001
    - host: "localhost"
      port: 8002

train_args:
  comm_round: 5

tracking_args:
  log_file_dir: ./log/swarm
EOF
fi

# Clean previous logs
echo -e "${BLUE}Cleaning previous logs...${NC}"
rm -f "$LOG_DIR"/*.log
rm -f "$LOG_DIR"/*.pid

# Function to start a simple swarm node
start_simple_node() {
    local node_id=$1
    local rank=$2
    
    echo -e "${GREEN}Starting Simple Node $node_id (Rank $rank)${NC}"
    
    python3 swarm_simple.py \
        --cf "$CONFIG_FILE" \
        --node_id "$node_id" \
        --rank "$rank" \
        > "$LOG_DIR/simple_${node_id}.log" 2>&1 &
    
    local pid=$!
    echo "$pid" > "$LOG_DIR/simple_${node_id}.pid"
    
    return $pid
}

# Function to stop all nodes
cleanup_nodes() {
    echo -e "${YELLOW}Cleaning up nodes...${NC}"
    
    for pid_file in "$LOG_DIR"/simple_*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo "Stopping process $pid"
                kill "$pid" 2>/dev/null || true
                sleep 1
                kill -9 "$pid" 2>/dev/null || true
            fi
            rm -f "$pid_file"
        fi
    done
}

# Trap to cleanup on exit
trap cleanup_nodes EXIT INT TERM

# Display simulation parameters
echo -e "${BLUE}Simple Simulation Parameters:${NC}"
echo "  - Number of nodes: $WORKER_NUM"
echo "  - Configuration: $CONFIG_FILE"
echo "  - Log directory: $LOG_DIR"
echo ""

# Test basic functionality first
echo -e "${BLUE}Testing basic blockchain functionality...${NC}"
if python3 -c "
import sys
sys.path.append('.')
from blockchain.ledger import SwarmBlockchain
blockchain = SwarmBlockchain(difficulty=1)
print('✅ Blockchain creation successful')
print(f'Genesis block hash: {blockchain.chain[0].hash[:10]}...')
"; then
    echo -e "${GREEN}Basic blockchain test passed${NC}"
else
    echo -e "${RED}Basic blockchain test failed${NC}"
    exit 1
fi

echo ""

# Start coordinator node first
echo -e "${BLUE}Starting Simple Swarm Simulation...${NC}"
start_simple_node "coordinator" 0
COORDINATOR_PID=$!

# Wait for coordinator to initialize
echo "Waiting for coordinator to initialize..."
sleep 3

# Start worker nodes
WORKER_PIDS=()
for i in $(seq 1 "$WORKER_NUM"); do
    start_simple_node "node_$i" "$i"
    WORKER_PIDS+=($!)
    
    # Stagger node startup
    sleep 1
done

echo -e "${GREEN}All nodes started. Running simple swarm learning...${NC}"
echo ""

# Monitor simulation progress
echo -e "${BLUE}Monitoring simulation...${NC}"

# Simple monitoring loop
for round in $(seq 1 5); do
    sleep 10
    
    echo -e "${BLUE}Round $round status:${NC}"
    
    # Check if processes are still running
    running_count=0
    
    if kill -0 "$COORDINATOR_PID" 2>/dev/null; then
        running_count=$((running_count + 1))
    fi
    
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            running_count=$((running_count + 1))
        fi
    done
    
    echo "  - Running nodes: $running_count/$(($WORKER_NUM + 1))"
    
    # Show some log stats
    if [ -f "$LOG_DIR/simple_coordinator.log" ]; then
        local transactions=$(grep -c "Model update broadcasted" "$LOG_DIR"/simple_*.log 2>/dev/null || echo "0")
        local consensus=$(grep -c "Consensus reached" "$LOG_DIR"/simple_*.log 2>/dev/null || echo "0")
        echo "  - Model updates: $transactions"
        echo "  - Consensus attempts: $consensus"
    fi
    
    echo ""
    
    # Break if nodes stopped
    if [ "$running_count" -eq 0 ]; then
        echo -e "${YELLOW}All nodes have stopped${NC}"
        break
    fi
done

# Wait for completion
echo -e "${BLUE}Waiting for simulation to complete...${NC}"
wait $COORDINATOR_PID 2>/dev/null || true

for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo -e "${GREEN}Simple simulation completed!${NC}"

# Show results
echo -e "${BLUE}Results Summary:${NC}"
echo "--- Recent Coordinator Log ---"
tail -n 10 "$LOG_DIR/simple_coordinator.log" 2>/dev/null || echo "No coordinator log"

echo ""
echo "--- Node Logs (last 5 lines each) ---"
for log_file in "$LOG_DIR"/simple_node_*.log; do
    if [ -f "$log_file" ]; then
        echo "$(basename "$log_file"):"
        tail -n 5 "$log_file" | sed 's/^/  /'
        echo ""
    fi
done

# Check for errors
echo "--- Errors (if any) ---"
grep -i "error\|exception\|failed" "$LOG_DIR"/simple_*.log 2>/dev/null | head -n 10 || echo "No critical errors found"

echo ""
echo -e "${GREEN}Simple swarm learning test completed!${NC}"
echo -e "Logs: ${BLUE}$LOG_DIR${NC}"
echo ""
echo "To run the full version (requires PyTorch):"
echo "  ./run_swarm_simulation.sh $WORKER_NUM"
