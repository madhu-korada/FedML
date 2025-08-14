#!/usr/bin/env bash

# Swarm Learning Simulation Script for IoT Anomaly Detection
# This script runs a multi-node swarm learning simulation using blockchain consensus

set -e  # Exit on any error

# Configuration
WORKER_NUM=${1:-9}  # Number of worker nodes (default: 9)
CONFIG_FILE="config_swarm/swarm_config.yaml"
LOG_DIR="./log/swarm"
RESULTS_DIR="./results/swarm"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}   Swarm Learning IoT Simulation     ${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Validate parameters
if [ "$WORKER_NUM" -lt 3 ]; then
    echo -e "${RED}Error: Minimum 3 nodes required for meaningful consensus${NC}"
    echo "Usage: $0 [number_of_nodes]"
    echo "Example: $0 9"
    exit 1
fi

if [ "$WORKER_NUM" -gt 20 ]; then
    echo -e "${YELLOW}Warning: Running with more than 20 nodes may be resource intensive${NC}"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Setup directories
echo -e "${BLUE}Setting up directories...${NC}"
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "./blockchain_data"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file $CONFIG_FILE not found${NC}"
    exit 1
fi

# Check Python dependencies
echo -e "${BLUE}Checking dependencies...${NC}"
python3 -c "import torch, yaml, numpy, pandas" 2>/dev/null || {
    echo -e "${RED}Error: Missing required Python packages${NC}"
    echo "Please install: torch, pyyaml, numpy, pandas"
    exit 1
}

# Clean previous logs
echo -e "${BLUE}Cleaning previous logs...${NC}"
rm -f "$LOG_DIR"/*.log
rm -f "$RESULTS_DIR"/*

# Function to start a swarm node
start_swarm_node() {
    local node_id=$1
    local rank=$2
    local port=$3
    
    echo -e "${GREEN}Starting Node $node_id (Rank $rank) on port $port${NC}"
    
    python3 swarm_iot.py \
        --cf "$CONFIG_FILE" \
        --node_id "$node_id" \
        --rank "$rank" \
        > "$LOG_DIR/node_${node_id}.log" 2>&1 &
    
    local pid=$!
    echo "$pid" > "$LOG_DIR/node_${node_id}.pid"
    
    return $pid
}

# Function to stop all nodes
cleanup_nodes() {
    echo -e "${YELLOW}Cleaning up nodes...${NC}"
    
    for pid_file in "$LOG_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo "Stopping process $pid"
                kill "$pid" 2>/dev/null || true
                # Give process time to cleanup
                sleep 2
                # Force kill if still running
                kill -9 "$pid" 2>/dev/null || true
            fi
            rm -f "$pid_file"
        fi
    done
}

# Trap to cleanup on exit
trap cleanup_nodes EXIT INT TERM

# Display simulation parameters
echo -e "${BLUE}Simulation Parameters:${NC}"
echo "  - Number of nodes: $WORKER_NUM"
echo "  - Configuration: $CONFIG_FILE"
echo "  - Log directory: $LOG_DIR"
echo "  - Results directory: $RESULTS_DIR"
echo ""

# Start coordinator node (rank 0) first
echo -e "${BLUE}Starting Swarm Learning Simulation...${NC}"
start_swarm_node "coordinator" 0 8000
COORDINATOR_PID=$!

# Wait for coordinator to initialize
echo "Waiting for coordinator to initialize..."
sleep 5

# Start worker nodes
WORKER_PIDS=()
for i in $(seq 1 "$WORKER_NUM"); do
    port=$((8000 + i))
    start_swarm_node "node_$i" "$i" "$port"
    WORKER_PIDS+=($!)
    
    # Stagger node startup to avoid network congestion
    sleep 2
done

echo -e "${GREEN}All nodes started. Running swarm learning...${NC}"
echo "Monitor progress with: tail -f $LOG_DIR/*.log"
echo ""

# Monitor simulation progress
echo -e "${BLUE}Monitoring simulation progress...${NC}"

# Function to check if all nodes are still running
check_nodes_running() {
    local all_running=true
    
    # Check coordinator
    if ! kill -0 "$COORDINATOR_PID" 2>/dev/null; then
        all_running=false
    fi
    
    # Check workers
    for pid in "${WORKER_PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            all_running=false
            break
        fi
    done
    
    echo $all_running
}

# Monitor loop
ROUND=0
MAX_ROUNDS=15  # From config
MONITORING_INTERVAL=30

while [ $ROUND -lt $MAX_ROUNDS ]; do
    sleep $MONITORING_INTERVAL
    ROUND=$((ROUND + 1))
    
    # Check if nodes are still running
    if [ "$(check_nodes_running)" = "false" ]; then
        echo -e "${RED}Some nodes have stopped unexpectedly${NC}"
        break
    fi
    
    # Display progress
    echo -e "${BLUE}Progress: Round $ROUND/$MAX_ROUNDS${NC}"
    
    # Show basic stats from logs
    if [ -f "$LOG_DIR/coordinator.log" ]; then
        local consensus_count=$(grep -c "consensus reached" "$LOG_DIR"/*.log 2>/dev/null || echo "0")
        local blocks_mined=$(grep -c "Mined new block" "$LOG_DIR"/*.log 2>/dev/null || echo "0")
        echo "  - Consensus reached: $consensus_count times"
        echo "  - Blocks mined: $blocks_mined"
    fi
    
    echo ""
done

# Wait for all processes to complete naturally
echo -e "${BLUE}Waiting for simulation to complete...${NC}"
wait $COORDINATOR_PID 2>/dev/null || true

for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo -e "${GREEN}Simulation completed!${NC}"

# Analyze results
echo -e "${BLUE}Analyzing results...${NC}"

# Check for result files
RESULT_FILES=$(find "$LOG_DIR/results" -name "*.json" 2>/dev/null | wc -l)
if [ "$RESULT_FILES" -gt 0 ]; then
    echo "Found $RESULT_FILES result files"
    
    # Copy results to results directory
    cp -r "$LOG_DIR/results"/* "$RESULTS_DIR/" 2>/dev/null || true
    
    # Generate summary
    python3 << EOF
import json
import os
import glob

print("\\n" + "="*50)
print("         SWARM LEARNING RESULTS SUMMARY")
print("="*50)

results_dir = "$RESULTS_DIR"
stats_files = glob.glob(os.path.join(results_dir, "final_stats_*.json"))

if stats_files:
    total_nodes = len(stats_files)
    total_blocks = 0
    total_transactions = 0
    avg_detection_rate = 0
    
    print(f"Total participating nodes: {total_nodes}")
    
    for stats_file in stats_files:
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            blockchain_stats = stats.get('blockchain_stats', {})
            total_blocks = max(total_blocks, blockchain_stats.get('total_blocks', 0))
            total_transactions += blockchain_stats.get('total_transactions', 0)
            
            node_stats = stats.get('node_stats', {})
            print(f"Node {node_stats.get('node_id', 'unknown')}: {node_stats.get('model_parameters', 0)} parameters")
            
        except Exception as e:
            print(f"Error reading {stats_file}: {e}")
    
    print(f"\\nBlockchain Statistics:")
    print(f"  - Total blocks mined: {total_blocks}")
    print(f"  - Total transactions: {total_transactions}")
    print(f"  - Average transactions per block: {total_transactions/max(1,total_blocks):.1f}")
    
else:
    print("No result files found")

print("\\nDetailed logs available in: $LOG_DIR")
print("Result files available in: $RESULTS_DIR")
print("="*50)
EOF

else
    echo -e "${YELLOW}No result files found. Check logs for errors.${NC}"
fi

# Show final logs
echo -e "${BLUE}Recent log entries:${NC}"
echo "--- Coordinator Log (last 10 lines) ---"
tail -n 10 "$LOG_DIR/coordinator.log" 2>/dev/null || echo "No coordinator log found"

echo ""
echo "--- Node Logs (errors only) ---"
grep -i "error\|fail\|exception" "$LOG_DIR"/node_*.log 2>/dev/null | tail -n 5 || echo "No errors found in node logs"

echo ""
echo -e "${GREEN}Swarm learning simulation completed successfully!${NC}"
echo -e "Logs: ${BLUE}$LOG_DIR${NC}"
echo -e "Results: ${BLUE}$RESULTS_DIR${NC}"
