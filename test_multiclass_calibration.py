#!/usr/bin/env python3
"""
Test script for comparing single-class vs multi-class influence function calibration.

This script demonstrates the fixed implementation and compares:
1. Single-class calibration (optimizing for one target class)
2. Multi-class calibration (optimizing for all classes)
"""

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np

from model import CNN
from influence_function import calibrate_model_comparative, get_critical_slice

def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy overall and per-class."""
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            # Per-class accuracy
            for i in range(y.size(0)):
                label = y[i].item()
                class_correct[label] += (predicted[i] == y[i]).item()
                class_total[label] += 1
    
    overall_acc = 100 * correct / total
    class_accs = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                  for i in range(10)]
    
    return overall_acc, class_accs

def main():
    print("🔬 INFLUENCE FUNCTION CALIBRATION EXPERIMENT")
    print("=" * 60)
    
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Set reproducible random seeds
    from config import set_random_seeds
    set_random_seeds()
    
    # Data
    trans = T.Compose([T.ToTensor(), T.Normalize((.5,.5,.5),(.5,.5,.5))])
    testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=trans)
    trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=trans)
    
    # Smaller subsets for quick experiment
    pub_subset = Subset(testset, range(1000))  # Public data for calibration
    train_subset = Subset(trainset, range(1000))  # Training data for influence vectors
    eval_subset = Subset(testset, range(1000, 2000))  # Evaluation data
    
    pub_loader = DataLoader(pub_subset, batch_size=64, shuffle=False)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=False)
    eval_loader = DataLoader(eval_subset, batch_size=64, shuffle=False)
    
    # Train baseline model
    print("\n🔧 Training baseline model...")
    model = CNN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    
    for epoch in range(5):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    
    # Evaluate baseline
    print("\n📊 Baseline Performance:")
    baseline_overall, baseline_per_class = evaluate_model(model, eval_loader, device)
    print(f"   • Overall accuracy: {baseline_overall:.2f}%")
    print(f"   • Per-class accuracy: {[f'{acc:.1f}' for acc in baseline_per_class]}")
    print(f"   • Class 3 (target) accuracy: {baseline_per_class[3]:.2f}%")
    
    # Run comparative calibration
    print("\n🧪 Running Comparative Calibration Experiment:")
    results = calibrate_model_comparative(
        model, pub_loader, train_loader, device,
        method='linear', eta=100, target_class=3, comparison_mode=True
    )
    
    # Evaluate single-class calibration
    print("\n📊 Single-Class Calibration Results:")
    single_overall, single_per_class = evaluate_model(results['single_class'], eval_loader, device)
    print(f"   • Overall accuracy: {single_overall:.2f}% ({single_overall - baseline_overall:+.2f}%)")
    print(f"   • Per-class accuracy: {[f'{acc:.1f}' for acc in single_per_class]}")
    print(f"   • Class 3 (target) accuracy: {single_per_class[3]:.2f}% ({single_per_class[3] - baseline_per_class[3]:+.2f}%)")
    
    # Evaluate multi-class calibration
    print("\n📊 Multi-Class Calibration Results:")
    multi_overall, multi_per_class = evaluate_model(results['multi_class'], eval_loader, device)
    print(f"   • Overall accuracy: {multi_overall:.2f}% ({multi_overall - baseline_overall:+.2f}%)")
    print(f"   • Per-class accuracy: {[f'{acc:.1f}' for acc in multi_per_class]}")
    print(f"   • Class 3 (target) accuracy: {multi_per_class[3]:.2f}% ({multi_per_class[3] - baseline_per_class[3]:+.2f}%)")
    
    # Analysis
    print("\n📈 COMPARATIVE ANALYSIS:")
    print("=" * 50)
    
    print(f"🎯 Target Class (3) Performance:")
    print(f"   • Baseline:     {baseline_per_class[3]:.2f}%")
    print(f"   • Single-class: {single_per_class[3]:.2f}% ({single_per_class[3] - baseline_per_class[3]:+.2f}%)")
    print(f"   • Multi-class:  {multi_per_class[3]:.2f}% ({multi_per_class[3] - baseline_per_class[3]:+.2f}%)")
    
    print(f"\n🌐 Overall Performance:")
    print(f"   • Baseline:     {baseline_overall:.2f}%")
    print(f"   • Single-class: {single_overall:.2f}% ({single_overall - baseline_overall:+.2f}%)")
    print(f"   • Multi-class:  {multi_overall:.2f}% ({multi_overall - baseline_overall:+.2f}%)")
    
    # Check for overfitting
    single_class_degradation = sum(1 for i in range(10) if i != 3 and single_per_class[i] < baseline_per_class[i] - 2.0)
    multi_class_degradation = sum(1 for i in range(10) if i != 3 and multi_per_class[i] < baseline_per_class[i] - 2.0)
    
    print(f"\n⚠️  Overfitting Analysis:")
    print(f"   • Single-class: {single_class_degradation}/9 non-target classes degraded significantly")
    print(f"   • Multi-class:  {multi_class_degradation}/9 non-target classes degraded significantly")
    
    if single_class_degradation > multi_class_degradation:
        print(f"   ✅ Multi-class approach shows less overfitting!")
    elif multi_class_degradation > single_class_degradation:
        print(f"   ⚠️  Single-class approach shows less overfitting")
    else:
        print(f"   📊 Both approaches show similar overfitting patterns")
    
    print(f"\n🏆 CONCLUSION:")
    if multi_overall > single_overall and multi_per_class[3] >= single_per_class[3] * 0.95:
        print(f"   🥇 Multi-class calibration is BETTER: preserves overall performance while maintaining target class improvement")
    elif single_per_class[3] > multi_per_class[3] and single_overall >= multi_overall * 0.95:
        print(f"   🥈 Single-class calibration is more TARGETED: better target class improvement")
    else:
        print(f"   🤝 Both approaches have trade-offs - choose based on priorities")
    
    print(f"\n✅ Experiment completed!")

if __name__ == "__main__":
    main() 