#!/usr/bin/env python3
"""
Generate and save the Federated Learning vs Swarm Learning comparison diagram
"""

import os
import subprocess
import tempfile

def create_mermaid_diagram():
    """Create the Mermaid diagram content"""
    diagram_content = """graph TB
    subgraph "Federated Learning (Original)"
        FS[Central Server]
        FC1[IoT Device 1<br/>Doorbell]
        FC2[IoT Device 2<br/>Thermostat] 
        FC3[IoT Device 3<br/>Camera]
        FC4[IoT Device 4<br/>Monitor]
        
        FC1 -->|Model Updates| FS
        FC2 -->|Model Updates| FS
        FC3 -->|Model Updates| FS
        FC4 -->|Model Updates| FS
        
        FS -->|Global Model| FC1
        FS -->|Global Model| FC2
        FS -->|Global Model| FC3
        FS -->|Global Model| FC4
    end
    
    subgraph "Swarm Learning (New Implementation)"
        BC[Blockchain Ledger<br/>🔗]
        SC1[Swarm Node 1<br/>Doorbell + Blockchain]
        SC2[Swarm Node 2<br/>Thermostat + Blockchain]
        SC3[Swarm Node 3<br/>Camera + Blockchain]
        SC4[Swarm Node 4<br/>Monitor + Blockchain]
        
        SC1 <-->|P2P Communication| SC2
        SC2 <-->|P2P Communication| SC3
        SC3 <-->|P2P Communication| SC4
        SC4 <-->|P2P Communication| SC1
        SC1 <-->|P2P Communication| SC3
        SC2 <-->|P2P Communication| SC4
        
        SC1 -->|Transactions| BC
        SC2 -->|Transactions| BC
        SC3 -->|Transactions| BC
        SC4 -->|Transactions| BC
        
        BC -->|Consensus Model| SC1
        BC -->|Consensus Model| SC2
        BC -->|Consensus Model| SC3
        BC -->|Consensus Model| SC4
    end
    
    subgraph "Key Differences"
        D1["❌ Single Point of Failure<br/>✅ Fully Decentralized"]
        D2["❌ Central Trust Required<br/>✅ Distributed Trust"]
        D3["❌ Limited Transparency<br/>✅ Full Transparency"]
        D4["❌ Simple Aggregation<br/>✅ Consensus-based Aggregation"]
    end"""
    
    return diagram_content

def save_as_mermaid_file():
    """Save the diagram as a .mmd file"""
    diagram_content = create_mermaid_diagram()
    
    output_file = "federated_vs_swarm_comparison.mmd"
    
    with open(output_file, 'w') as f:
        f.write(diagram_content)
    
    print(f"✅ Mermaid diagram saved as: {output_file}")
    return output_file

def try_mermaid_cli_conversion(mermaid_file):
    """Try to convert to PNG using Mermaid CLI if available"""
    try:
        # Check if mermaid CLI is available
        result = subprocess.run(['mmdc', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            output_png = mermaid_file.replace('.mmd', '.png')
            cmd = ['mmdc', '-i', mermaid_file, '-o', output_png, '-t', 'neutral']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ PNG diagram saved as: {output_png}")
                return True
            else:
                print(f"❌ Error converting to PNG: {result.stderr}")
                return False
        else:
            print("ℹ️  Mermaid CLI not available for PNG conversion")
            return False
    except FileNotFoundError:
        print("ℹ️  Mermaid CLI not found. Install with: npm install -g @mermaid-js/mermaid-cli")
        return False

def create_html_preview():
    """Create an HTML file that can render the Mermaid diagram"""
    diagram_content = create_mermaid_diagram()
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Federated Learning vs Swarm Learning Comparison</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .diagram-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .description {{
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>Federated Learning vs Swarm Learning Architecture Comparison</h1>
    
    <div class="description">
        <h3>Architecture Comparison</h3>
        <p>This diagram illustrates the key architectural differences between traditional Federated Learning 
        and the new Swarm Learning implementation for IoT anomaly detection.</p>
        
        <h4>Key Advantages of Swarm Learning:</h4>
        <ul>
            <li><strong>Decentralized:</strong> No single point of failure</li>
            <li><strong>Transparent:</strong> All decisions recorded on blockchain</li>
            <li><strong>Trustless:</strong> No need to trust a central authority</li>
            <li><strong>Consensus-based:</strong> Democratic model selection</li>
        </ul>
    </div>
    
    <div class="diagram-container">
        <div class="mermaid">
{diagram_content}
        </div>
    </div>
    
    <div class="description">
        <h4>Implementation Details:</h4>
        <ul>
            <li><strong>Blockchain Ledger:</strong> Records all model updates and transactions</li>
            <li><strong>P2P Communication:</strong> Direct node-to-node communication</li>
            <li><strong>Consensus Mechanism:</strong> Weighted voting based on performance and reputation</li>
            <li><strong>Reputation System:</strong> Tracks node contribution quality over time</li>
        </ul>
    </div>

    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
    </script>
</body>
</html>"""
    
    html_file = "federated_vs_swarm_comparison.html"
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    print(f"✅ HTML preview saved as: {html_file}")
    print(f"   Open in browser to view: file://{os.path.abspath(html_file)}")
    return html_file

def create_svg_version():
    """Create an SVG version using Python libraries if available"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import FancyBboxPatch, ConnectionPatch
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        fig.suptitle('Federated Learning vs Swarm Learning Architecture', fontsize=16, fontweight='bold')
        
        # Federated Learning (Left side)
        ax1.set_title('Federated Learning (Original)', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.axis('off')
        
        # Central Server
        server = FancyBboxPatch((4, 8), 2, 1, boxstyle="round,pad=0.1", 
                               facecolor='lightblue', edgecolor='blue', linewidth=2)
        ax1.add_patch(server)
        ax1.text(5, 8.5, 'Central\nServer', ha='center', va='center', fontweight='bold')
        
        # IoT Devices
        devices = [
            (2, 6, 'Doorbell'), (8, 6, 'Thermostat'),
            (2, 3, 'Camera'), (8, 3, 'Monitor')
        ]
        
        for x, y, name in devices:
            device = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, boxstyle="round,pad=0.1",
                                   facecolor='lightgreen', edgecolor='green')
            ax1.add_patch(device)
            ax1.text(x, y, name, ha='center', va='center', fontsize=10)
            
            # Arrows to/from server
            ax1.annotate('', xy=(5, 8), xytext=(x, y+0.4),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
            ax1.annotate('', xy=(x, y-0.4), xytext=(5, 8),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        
        ax1.text(1, 1, '❌ Single Point of Failure\n❌ Central Trust Required', 
                fontsize=10, color='red')
        
        # Swarm Learning (Right side)
        ax2.set_title('Swarm Learning (New Implementation)', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.axis('off')
        
        # Blockchain
        blockchain = FancyBboxPatch((4, 8.5), 2, 1, boxstyle="round,pad=0.1",
                                   facecolor='gold', edgecolor='orange', linewidth=2)
        ax2.add_patch(blockchain)
        ax2.text(5, 9, 'Blockchain\nLedger 🔗', ha='center', va='center', fontweight='bold')
        
        # Swarm Nodes
        nodes = [
            (2, 6, 'Node 1\nDoorbell'), (8, 6, 'Node 2\nThermostat'),
            (2, 3, 'Node 3\nCamera'), (8, 3, 'Node 4\nMonitor')
        ]
        
        for x, y, name in nodes:
            node = FancyBboxPatch((x-0.8, y-0.5), 1.6, 1, boxstyle="round,pad=0.1",
                                 facecolor='lightcoral', edgecolor='darkred')
            ax2.add_patch(node)
            ax2.text(x, y, name, ha='center', va='center', fontsize=9)
            
            # Arrows to blockchain
            ax2.annotate('', xy=(5, 8.5), xytext=(x, y+0.5),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))
        
        # P2P connections
        connections = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
        for i, j in connections:
            x1, y1 = nodes[i][0], nodes[i][1]
            x2, y2 = nodes[j][0], nodes[j][1]
            ax2.plot([x1, x2], [y1, y2], 'purple', linestyle='--', alpha=0.6, linewidth=1)
        
        ax2.text(1, 1, '✅ Fully Decentralized\n✅ Distributed Trust\n✅ Full Transparency', 
                fontsize=10, color='green')
        
        plt.tight_layout()
        
        svg_file = "federated_vs_swarm_comparison.svg"
        plt.savefig(svg_file, format='svg', dpi=300, bbox_inches='tight')
        plt.savefig("federated_vs_swarm_comparison.png", format='png', dpi=300, bbox_inches='tight')
        
        print(f"✅ SVG diagram saved as: {svg_file}")
        print(f"✅ PNG diagram saved as: federated_vs_swarm_comparison.png")
        
        plt.close()
        return True
        
    except ImportError:
        print("ℹ️  Matplotlib not available for SVG/PNG generation")
        return False

def main():
    """Main function to generate all diagram formats"""
    print("📊 Generating Federated Learning vs Swarm Learning Comparison Diagrams")
    print("=" * 70)
    
    # Always create Mermaid file
    mermaid_file = save_as_mermaid_file()
    
    # Try different output formats
    html_file = create_html_preview()
    
    # Try Mermaid CLI conversion
    try_mermaid_cli_conversion(mermaid_file)
    
    # Try matplotlib
    create_svg_version()
    
    print("\n📁 Generated Files:")
    for file in [mermaid_file, html_file, "federated_vs_swarm_comparison.png", "federated_vs_swarm_comparison.svg"]:
        if os.path.exists(file):
            print(f"   ✅ {file}")
    
    print(f"\n🌐 To view the interactive diagram, open: {html_file}")
    print("💡 Tip: The HTML file will render the diagram in any modern web browser")

if __name__ == "__main__":
    main()

