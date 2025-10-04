#!/usr/bin/env python3
"""
Quick Training Runner for IoT Autoencoder

Simple script to run different training configurations without command line arguments.
"""

from simple_train import SimpleIoTTrainer
import torch

def quick_train():
    """Quick training with synthetic data."""
    print("🚀 Quick Training (Synthetic Data)")
    print("="*50)
    
    trainer = SimpleIoTTrainer()
    results = trainer.train_federated(
        communication_rounds=5,
        local_epochs=1,
        learning_rate=0.03,
        use_real_data=False
    )
    
    print(f"✅ Training completed! Final loss: {results['final_loss']:.6f}")
    return trainer, results

def full_train():
    """Full training with real data."""
    print("🚀 Full Training (Real Data)")
    print("="*50)
    
    trainer = SimpleIoTTrainer()
    results = trainer.train_federated(
        communication_rounds=10,
        local_epochs=2,
        learning_rate=0.03,
        use_real_data=True
    )
    
    # Evaluate
    eval_results = trainer.evaluate_model(use_real_data=True)
    print(f"📊 Attack detection rate: {eval_results['attack_results']['detection_rate']:.2%}")
    
    # Save model
    trainer.save_model('./trained_models/iot_autoencoder_full.pt')
    
    return trainer, results

def demo_train():
    """Demo training with visualization."""
    print("🚀 Demo Training (With Visualization)")
    print("="*50)
    
    trainer = SimpleIoTTrainer()
    
    # Train on subset of devices for speed
    demo_devices = ["Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell"]
    
    results = trainer.train_federated(
        communication_rounds=8,
        local_epochs=1,
        learning_rate=0.03,
        use_real_data=True,
        participating_devices=demo_devices
    )
    
    # Visualize
    trainer.visualize_training(save_path='./demo_training_results.png')
    
    # Evaluate
    eval_results = trainer.evaluate_model(use_real_data=True)
    
    print(f"✅ Demo completed!")
    print(f"Final loss: {results['final_loss']:.6f}")
    print(f"Detection rate: {eval_results['attack_results']['detection_rate']:.2%}")
    
    return trainer, results

def compare_configurations():
    """Compare different training configurations."""
    print("🚀 Configuration Comparison")
    print("="*50)
    
    configs = [
        {"name": "Low LR", "lr": 0.01, "rounds": 5},
        {"name": "High LR", "lr": 0.05, "rounds": 5},
        {"name": "More Rounds", "lr": 0.03, "rounds": 15},
    ]
    
    results_comparison = []
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        trainer = SimpleIoTTrainer()
        
        results = trainer.train_federated(
            communication_rounds=config['rounds'],
            local_epochs=1,
            learning_rate=config['lr'],
            use_real_data=False  # Use synthetic for speed
        )
        
        results_comparison.append({
            'config': config,
            'final_loss': results['final_loss'],
            'trainer': trainer
        })
        
        print(f"  Final loss: {results['final_loss']:.6f}")
    
    # Find best configuration
    best = min(results_comparison, key=lambda x: x['final_loss'])
    print(f"\n🏆 Best configuration: {best['config']['name']}")
    print(f"   Final loss: {best['final_loss']:.6f}")
    
    return results_comparison

def main():
    """Interactive menu for different training modes."""
    print("🤖 IoT Autoencoder Training Runner")
    print("="*50)
    print("Choose a training mode:")
    print("1. Quick Train (5 rounds, synthetic data)")
    print("2. Full Train (10 rounds, real data)")
    print("3. Demo Train (with visualization)")
    print("4. Compare Configurations")
    print("5. Custom Training")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-5): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                quick_train()
            elif choice == '2':
                full_train()
            elif choice == '3':
                demo_train()
            elif choice == '4':
                compare_configurations()
            elif choice == '5':
                custom_training()
            else:
                print("❌ Invalid choice. Please enter 0-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def custom_training():
    """Custom training with user input."""
    print("\n🔧 Custom Training Configuration")
    print("-" * 30)
    
    try:
        rounds = int(input("Communication rounds (default 10): ") or "10")
        epochs = int(input("Local epochs (default 1): ") or "1")
        lr = float(input("Learning rate (default 0.03): ") or "0.03")
        use_real = input("Use real data? (y/n, default n): ").lower().startswith('y')
        save_model = input("Save model? (y/n, default n): ").lower().startswith('y')
        visualize = input("Show plots? (y/n, default n): ").lower().startswith('y')
        
        print(f"\n🚀 Starting custom training...")
        print(f"Rounds: {rounds}, Epochs: {epochs}, LR: {lr}")
        print(f"Real data: {use_real}, Save: {save_model}, Visualize: {visualize}")
        
        trainer = SimpleIoTTrainer()
        results = trainer.train_federated(
            communication_rounds=rounds,
            local_epochs=epochs,
            learning_rate=lr,
            use_real_data=use_real
        )
        
        print(f"✅ Custom training completed! Final loss: {results['final_loss']:.6f}")
        
        if save_model:
            trainer.save_model('./trained_models/custom_model.pt')
            print("💾 Model saved!")
        
        if visualize:
            trainer.visualize_training()
            print("📊 Plots displayed!")
            
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
    except Exception as e:
        print(f"❌ Training error: {e}")

if __name__ == "__main__":
    # Check if running with command line arguments
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "quick":
            quick_train()
        elif mode == "full":
            full_train()
        elif mode == "demo":
            demo_train()
        elif mode == "compare":
            compare_configurations()
        else:
            print(f"Usage: python {sys.argv[0]} [quick|full|demo|compare]")
    else:
        main()
