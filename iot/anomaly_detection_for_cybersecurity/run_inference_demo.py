#!/usr/bin/env python3
"""
Simple script to run the model inference demonstration with different configurations.
"""

import sys
import os
from model_inference_demo import ModelInferenceDemo

def run_basic_demo():
    """Run basic demo with synthetic data."""
    print("🚀 Running Basic Inference Demo (Synthetic Data)")
    print("="*60)
    
    demo = ModelInferenceDemo()
    results = demo.run_complete_demo(
        use_real_data=False,
        visualize=False,
        num_samples=5
    )
    
    return results

def run_real_data_demo():
    """Run demo with real IoT data."""
    print("🚀 Running Real Data Inference Demo")
    print("="*60)
    
    demo = ModelInferenceDemo()
    results = demo.run_complete_demo(
        use_real_data=True,
        visualize=False,
        num_samples=5
    )
    
    return results

def run_visual_demo():
    """Run demo with visualizations."""
    print("🚀 Running Visual Inference Demo")
    print("="*60)
    
    demo = ModelInferenceDemo()
    results = demo.run_complete_demo(
        use_real_data=True,
        visualize=True,
        num_samples=10
    )
    
    return results

def demonstrate_step_by_step():
    """Demonstrate each step of the inference process separately."""
    print("🔍 Step-by-Step Inference Demonstration")
    print("="*60)
    
    demo = ModelInferenceDemo()
    
    # Step 1: Create data
    print("\n📊 Step 1: Creating test data...")
    benign_data, attack_data = demo.create_synthetic_data(num_samples=3)
    print(f"Created {len(benign_data)} benign samples and {len(attack_data)} attack samples")
    print(f"Feature dimensions: {benign_data.shape[1]}")
    
    # Step 2: Forward pass on benign data
    print("\n🧠 Step 2: Forward pass on benign data...")
    benign_results = demo.demonstrate_forward_pass(benign_data, "benign")
    
    # Step 3: Forward pass on attack data
    print("\n🧠 Step 3: Forward pass on attack data...")
    attack_results = demo.demonstrate_forward_pass(attack_data, "attack")
    
    # Step 4: Calculate threshold
    print("\n📏 Step 4: Calculating anomaly threshold...")
    threshold = demo.calculate_anomaly_threshold(benign_results)
    
    # Step 5: Perform detection
    print("\n🎯 Step 5: Performing anomaly detection...")
    metrics = demo.perform_anomaly_detection(benign_results, attack_results, threshold)
    
    return {
        'benign_results': benign_results,
        'attack_results': attack_results,
        'threshold': threshold,
        'metrics': metrics
    }

def compare_architectures():
    """Compare different aspects of the model architecture."""
    print("🏗️ Model Architecture Analysis")
    print("="*60)
    
    demo = ModelInferenceDemo()
    
    # Analyze model structure
    print("\n📐 Model Structure:")
    print(f"Total parameters: {sum(p.numel() for p in demo.model.parameters()):,}")
    
    # Encoder analysis
    print(f"\n🔽 Encoder layers:")
    for i, layer in enumerate(demo.model.enc):
        if hasattr(layer, 'weight'):
            print(f"  Layer {i}: {layer.weight.shape[1]} → {layer.weight.shape[0]}")
    
    # Decoder analysis  
    print(f"\n🔼 Decoder layers:")
    for i, layer in enumerate(demo.model.dec):
        if hasattr(layer, 'weight'):
            print(f"  Layer {i}: {layer.weight.shape[1]} → {layer.weight.shape[0]}")
    
    # Test with different input sizes
    print(f"\n🧪 Testing forward pass...")
    test_input = demo.create_synthetic_data(num_samples=1)[0]
    results = demo.demonstrate_forward_pass(test_input, "test")
    
    print(f"\n✅ Architecture analysis complete!")
    return results

def main():
    """Main function with interactive menu."""
    print("🤖 IoT Autoencoder Model Inference Demonstration")
    print("="*60)
    print("Choose a demonstration mode:")
    print("1. Basic Demo (Synthetic Data)")
    print("2. Real Data Demo") 
    print("3. Visual Demo (with plots)")
    print("4. Step-by-Step Demo")
    print("5. Architecture Analysis")
    print("6. Run All Demos")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-6): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                run_basic_demo()
            elif choice == '2':
                run_real_data_demo()
            elif choice == '3':
                run_visual_demo()
            elif choice == '4':
                demonstrate_step_by_step()
            elif choice == '5':
                compare_architectures()
            elif choice == '6':
                print("🚀 Running all demonstrations...")
                run_basic_demo()
                print("\n" + "="*60 + "\n")
                run_real_data_demo()
                print("\n" + "="*60 + "\n")
                demonstrate_step_by_step()
                print("\n" + "="*60 + "\n")
                compare_architectures()
                print("\n✅ All demonstrations completed!")
            else:
                print("❌ Invalid choice. Please enter 0-6.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "basic":
            run_basic_demo()
        elif sys.argv[1] == "real":
            run_real_data_demo()
        elif sys.argv[1] == "visual":
            run_visual_demo()
        elif sys.argv[1] == "step":
            demonstrate_step_by_step()
        elif sys.argv[1] == "arch":
            compare_architectures()
        else:
            print("Usage: python run_inference_demo.py [basic|real|visual|step|arch]")
    else:
        main()
