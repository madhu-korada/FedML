import asyncio
import json
import logging
import socket
import threading
import time
from typing import Dict, List, Set, Optional, Callable, Any
from dataclasses import dataclass
import pickle
import hashlib

# Handle torch import gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, using basic model handling")

from blockchain.ledger import SwarmBlockchain, ModelUpdate


@dataclass
class PeerInfo:
    """Information about a peer node"""
    node_id: str
    host: str
    port: int
    last_seen: float
    reputation: float = 0.5


class SwarmNode:
    """
    Swarm Learning Node implementation
    Handles peer-to-peer communication, model sharing, and consensus participation
    """
    
    def __init__(self, 
                 node_id: str, 
                 host: str = "localhost", 
                 port: int = 8000,
                 blockchain: SwarmBlockchain = None):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.blockchain = blockchain or SwarmBlockchain()
        
        # Peer management
        self.peers: Dict[str, PeerInfo] = {}
        self.known_nodes: Set[str] = set()
        
        # Communication
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.message_handlers: Dict[str, Callable] = {}
        self.lock = threading.Lock()
        
        # Model state
        self.current_model_state: Optional[Dict] = None
        self.current_round = 0
        self.training_complete = False
        
        # Setup message handlers
        self._setup_message_handlers()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"SwarmNode-{node_id}")
    
    def _setup_message_handlers(self):
        """Setup handlers for different message types"""
        self.message_handlers = {
            'peer_discovery': self._handle_peer_discovery,
            'model_update': self._handle_model_update,
            'consensus_request': self._handle_consensus_request,
            'consensus_vote': self._handle_consensus_vote,
            'blockchain_sync': self._handle_blockchain_sync,
            'ping': self._handle_ping
        }
    
    def start(self):
        """Start the swarm node"""
        self.running = True
        
        # Start server to listen for incoming connections
        self._start_server()
        
        # Start periodic tasks
        threading.Thread(target=self._periodic_tasks, daemon=True).start()
        
        self.logger.info(f"Swarm node {self.node_id} started on {self.host}:{self.port}")
    
    def stop(self):
        """Stop the swarm node"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        self.logger.info(f"Swarm node {self.node_id} stopped")
    
    def _start_server(self):
        """Start server to listen for incoming peer connections"""
        def server_thread():
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            try:
                self.server_socket.bind((self.host, self.port))
                self.server_socket.listen(10)
                
                while self.running:
                    try:
                        client_socket, address = self.server_socket.accept()
                        threading.Thread(
                            target=self._handle_client_connection,
                            args=(client_socket, address),
                            daemon=True
                        ).start()
                    except socket.error:
                        if self.running:
                            self.logger.error("Socket error in server thread")
                        break
                        
            except Exception as e:
                self.logger.error(f"Server error: {e}")
            finally:
                if self.server_socket:
                    self.server_socket.close()
        
        threading.Thread(target=server_thread, daemon=True).start()
    
    def _handle_client_connection(self, client_socket: socket.socket, address):
        """Handle incoming client connection"""
        try:
            # Receive message
            data = client_socket.recv(8192)
            if data:
                message = json.loads(data.decode('utf-8'))
                self._process_message(message, client_socket)
        except Exception as e:
            self.logger.error(f"Error handling client connection: {e}")
        finally:
            client_socket.close()
    
    def _process_message(self, message: Dict, client_socket: socket.socket = None):
        """Process incoming message"""
        msg_type = message.get('type')
        if msg_type in self.message_handlers:
            response = self.message_handlers[msg_type](message)
            
            # Send response if socket is available
            if client_socket and response:
                try:
                    response_data = json.dumps(response).encode('utf-8')
                    client_socket.send(response_data)
                except Exception as e:
                    self.logger.error(f"Error sending response: {e}")
        else:
            self.logger.warning(f"Unknown message type: {msg_type}")
    
    def _send_message_to_peer(self, peer_host: str, peer_port: int, message: Dict) -> Optional[Dict]:
        """Send message to a specific peer"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)  # 10 second timeout
            sock.connect((peer_host, peer_port))
            
            # Send message
            message_data = json.dumps(message).encode('utf-8')
            sock.send(message_data)
            
            # Receive response
            response_data = sock.recv(8192)
            if response_data:
                response = json.loads(response_data.decode('utf-8'))
                sock.close()
                return response
            
            sock.close()
            return None
            
        except Exception as e:
            self.logger.error(f"Error sending message to peer {peer_host}:{peer_port}: {e}")
            return None
    
    def discover_peers(self, bootstrap_peers: List[tuple]):
        """Discover peers in the network"""
        for host, port in bootstrap_peers:
            if (host, port) != (self.host, self.port):  # Don't connect to self
                message = {
                    'type': 'peer_discovery',
                    'node_id': self.node_id,
                    'host': self.host,
                    'port': self.port,
                    'timestamp': time.time()
                }
                
                response = self._send_message_to_peer(host, port, message)
                if response:
                    self._process_peer_discovery_response(response)
    
    def _handle_peer_discovery(self, message: Dict) -> Dict:
        """Handle peer discovery message"""
        peer_id = message['node_id']
        peer_host = message['host']
        peer_port = message['port']
        
        # Add peer to known peers
        with self.lock:
            self.peers[peer_id] = PeerInfo(
                node_id=peer_id,
                host=peer_host,
                port=peer_port,
                last_seen=time.time()
            )
            self.known_nodes.add(peer_id)
        
        self.logger.info(f"Discovered peer: {peer_id}")
        
        # Return current peer list
        return {
            'type': 'peer_discovery_response',
            'node_id': self.node_id,
            'peers': [
                {
                    'node_id': peer.node_id,
                    'host': peer.host,
                    'port': peer.port
                }
                for peer in self.peers.values()
            ]
        }
    
    def _process_peer_discovery_response(self, response: Dict):
        """Process peer discovery response"""
        if response['type'] == 'peer_discovery_response':
            for peer_info in response['peers']:
                peer_id = peer_info['node_id']
                if peer_id != self.node_id:  # Don't add self
                    with self.lock:
                        self.peers[peer_id] = PeerInfo(
                            node_id=peer_id,
                            host=peer_info['host'],
                            port=peer_info['port'],
                            last_seen=time.time()
                        )
                        self.known_nodes.add(peer_id)
    
    def broadcast_model_update(self, model_state: Dict, performance_metrics: Dict, training_data_size: int):
        """Broadcast model update to all peers"""
        # Serialize model weights
        model_weights = self.blockchain.serialize_model_weights(model_state)
        
        # Create model update transaction
        update = ModelUpdate(
            node_id=self.node_id,
            model_weights=model_weights,
            performance_metrics=performance_metrics,
            training_data_size=training_data_size,
            timestamp=time.time(),
            round_number=self.current_round
        )
        
        # Add to blockchain
        if self.blockchain.add_transaction(update):
            # Broadcast to peers
            message = {
                'type': 'model_update',
                'update': {
                    'node_id': update.node_id,
                    'performance_metrics': update.performance_metrics,
                    'training_data_size': update.training_data_size,
                    'timestamp': update.timestamp,
                    'round_number': update.round_number,
                    'model_hash': hashlib.sha256(model_weights).hexdigest()
                }
            }
            
            self._broadcast_to_peers(message)
            self.logger.info(f"Broadcasted model update for round {self.current_round}")
    
    def _handle_model_update(self, message: Dict) -> Dict:
        """Handle model update message from peer"""
        update_info = message['update']
        peer_id = update_info['node_id']
        
        self.logger.info(f"Received model update from {peer_id}")
        
        return {'status': 'received'}
    
    def request_consensus(self) -> Optional[Dict]:
        """Request consensus for current round"""
        consensus_result = self.blockchain.reach_consensus(self.current_round)
        
        if consensus_result:
            # Mine a new block with consensus updates
            new_block = self.blockchain.mine_block(self.node_id)
            if new_block:
                self.logger.info(f"Mined new block: {new_block.index}")
                
                # Broadcast new block to peers
                self._broadcast_blockchain_sync()
        
        return consensus_result
    
    def _handle_consensus_request(self, message: Dict) -> Dict:
        """Handle consensus request from peer"""
        round_number = message.get('round_number', self.current_round)
        consensus = self.blockchain.reach_consensus(round_number)
        
        return {
            'type': 'consensus_response',
            'consensus': consensus is not None,
            'round_number': round_number
        }
    
    def _handle_consensus_vote(self, message: Dict) -> Dict:
        """Handle consensus vote from peer"""
        # Implementation for voting mechanism
        return {'status': 'vote_received'}
    
    def _handle_blockchain_sync(self, message: Dict) -> Dict:
        """Handle blockchain synchronization"""
        # Simple sync - in production would be more sophisticated
        return {
            'type': 'blockchain_sync_response',
            'chain_length': len(self.blockchain.chain)
        }
    
    def _handle_ping(self, message: Dict) -> Dict:
        """Handle ping message"""
        return {
            'type': 'pong',
            'node_id': self.node_id,
            'timestamp': time.time()
        }
    
    def _broadcast_to_peers(self, message: Dict):
        """Broadcast message to all known peers"""
        failed_peers = []
        
        for peer_id, peer in self.peers.items():
            if peer_id != self.node_id:
                response = self._send_message_to_peer(peer.host, peer.port, message)
                if response is None:
                    failed_peers.append(peer_id)
                else:
                    # Update last seen
                    peer.last_seen = time.time()
        
        # Remove failed peers
        for peer_id in failed_peers:
            with self.lock:
                if peer_id in self.peers:
                    del self.peers[peer_id]
                    self.known_nodes.discard(peer_id)
    
    def _broadcast_blockchain_sync(self):
        """Broadcast blockchain sync message"""
        message = {
            'type': 'blockchain_sync',
            'node_id': self.node_id,
            'chain_length': len(self.blockchain.chain),
            'latest_hash': self.blockchain.chain[-1].hash
        }
        self._broadcast_to_peers(message)
    
    def _periodic_tasks(self):
        """Run periodic maintenance tasks"""
        while self.running:
            try:
                # Clean up old peers
                current_time = time.time()
                timeout = 300  # 5 minutes
                
                with self.lock:
                    expired_peers = [
                        peer_id for peer_id, peer in self.peers.items()
                        if current_time - peer.last_seen > timeout
                    ]
                    
                    for peer_id in expired_peers:
                        del self.peers[peer_id]
                        self.known_nodes.discard(peer_id)
                
                # Ping peers to check connectivity
                if self.peers:
                    ping_message = {
                        'type': 'ping',
                        'node_id': self.node_id,
                        'timestamp': current_time
                    }
                    self._broadcast_to_peers(ping_message)
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Error in periodic tasks: {e}")
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        return {
            'node_id': self.node_id,
            'connected_peers': len(self.peers),
            'known_nodes': len(self.known_nodes),
            'blockchain_stats': self.blockchain.get_chain_stats(),
            'current_round': self.current_round
        }
    
    def set_model_state(self, model_state: Dict):
        """Set current model state"""
        self.current_model_state = model_state
    
    def get_consensus_model(self) -> Optional[Dict]:
        """Get consensus model from blockchain"""
        consensus = self.blockchain.reach_consensus(self.current_round)
        
        if consensus and consensus['consensus_updates']:
            # For simplicity, use the first consensus update
            # In practice, you'd aggregate multiple updates
            first_update = consensus['consensus_updates'][0]
            model_weights = self.blockchain.deserialize_model_weights(first_update.model_weights)
            return model_weights
        
        return None
