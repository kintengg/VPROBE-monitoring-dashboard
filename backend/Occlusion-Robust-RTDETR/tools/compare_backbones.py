#!/usr/bin/env python3
"""
Backbone Comparison Script: ResNet50 vs FasterViT-0 for RT-DETR
Compare performance metrics between different backbone architectures
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def load_training_logs(log_file_path):
    """Load training logs and extract metrics"""
    epochs = []
    metrics = {
        'precision@IoU=0.5': [],
        'recall@100': [],
        'f1_score': [],
        'mAP@0.5:0.95': [],
        'mAP@0.5': [],
        'mAP@0.75': []
    }
    
    with open(log_file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'epoch' in data:
                epochs.append(data['epoch'])
                
                # Extract test metrics
                for key in metrics.keys():
                    test_key = f'test_{key}'
                    if test_key in data:
                        metrics[key].append(data[test_key])
                    else:
                        metrics[key].append(0.0)
    
    return epochs, metrics

def create_comparison_plots(resnet_data, fastervit_data, output_dir):
    """Create comparison plots for different metrics"""
    resnet_epochs, resnet_metrics = resnet_data
    fastervit_epochs, fastervit_metrics = fastervit_data
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    
    # Colors for the two models
    resnet_color = '#1f77b4'  # Blue
    fastervit_color = '#ff7f0e'  # Orange
    
    # 1. Precision Comparison
    plt.figure(figsize=(12, 8))
    plt.plot(resnet_epochs, resnet_metrics['precision@IoU=0.5'], 
             marker='o', color=resnet_color, label='ResNet50', linewidth=2)
    plt.plot(fastervit_epochs, fastervit_metrics['precision@IoU=0.5'], 
             marker='s', color=fastervit_color, label='FasterViT-0', linewidth=2)
    plt.title('Precision@IoU=0.5 Comparison: ResNet50 vs FasterViT-0', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Precision@IoU=0.5', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Recall Comparison
    plt.figure(figsize=(12, 8))
    plt.plot(resnet_epochs, resnet_metrics['recall@100'], 
             marker='o', color=resnet_color, label='ResNet50', linewidth=2)
    plt.plot(fastervit_epochs, fastervit_metrics['recall@100'], 
             marker='s', color=fastervit_color, label='FasterViT-0', linewidth=2)
    plt.title('Recall@100 Comparison: ResNet50 vs FasterViT-0', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Recall@100', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'recall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. F1-Score Comparison
    plt.figure(figsize=(12, 8))
    plt.plot(resnet_epochs, resnet_metrics['f1_score'], 
             marker='o', color=resnet_color, label='ResNet50', linewidth=2)
    plt.plot(fastervit_epochs, fastervit_metrics['f1_score'], 
             marker='s', color=fastervit_color, label='FasterViT-0', linewidth=2)
    plt.title('F1-Score Comparison: ResNet50 vs FasterViT-0', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('F1-Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_score_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. mAP@0.5:0.95 Comparison
    plt.figure(figsize=(12, 8))
    plt.plot(resnet_epochs, resnet_metrics['mAP@0.5:0.95'], 
             marker='o', color=resnet_color, label='ResNet50', linewidth=2)
    plt.plot(fastervit_epochs, fastervit_metrics['mAP@0.5:0.95'], 
             marker='s', color=fastervit_color, label='FasterViT-0', linewidth=2)
    plt.title('mAP@0.5:0.95 Comparison: ResNet50 vs FasterViT-0', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('mAP@0.5:0.95', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'mAP_0.5_0.95_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. mAP@0.5 Comparison
    plt.figure(figsize=(12, 8))
    plt.plot(resnet_epochs, resnet_metrics['mAP@0.5'], 
             marker='o', color=resnet_color, label='ResNet50', linewidth=2)
    plt.plot(fastervit_epochs, fastervit_metrics['mAP@0.5'], 
             marker='s', color=fastervit_color, label='FasterViT-0', linewidth=2)
    plt.title('mAP@0.5 Comparison: ResNet50 vs FasterViT-0', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('mAP@0.5', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'mAP_0.5_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Combined 2x2 subplot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Backbone Performance Comparison: ResNet50 vs FasterViT-0', fontsize=18, fontweight='bold')
    
    # Precision
    axes[0, 0].plot(resnet_epochs, resnet_metrics['precision@IoU=0.5'], 
                    marker='o', color=resnet_color, label='ResNet50', linewidth=2)
    axes[0, 0].plot(fastervit_epochs, fastervit_metrics['precision@IoU=0.5'], 
                    marker='s', color=fastervit_color, label='FasterViT-0', linewidth=2)
    axes[0, 0].set_title('Precision@IoU=0.5', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Recall
    axes[0, 1].plot(resnet_epochs, resnet_metrics['recall@100'], 
                    marker='o', color=resnet_color, label='ResNet50', linewidth=2)
    axes[0, 1].plot(fastervit_epochs, fastervit_metrics['recall@100'], 
                    marker='s', color=fastervit_color, label='FasterViT-0', linewidth=2)
    axes[0, 1].set_title('Recall@100', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # F1-Score
    axes[1, 0].plot(resnet_epochs, resnet_metrics['f1_score'], 
                    marker='o', color=resnet_color, label='ResNet50', linewidth=2)
    axes[1, 0].plot(fastervit_epochs, fastervit_metrics['f1_score'], 
                    marker='s', color=fastervit_color, label='FasterViT-0', linewidth=2)
    axes[1, 0].set_title('F1-Score', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # mAP@0.5:0.95
    axes[1, 1].plot(resnet_epochs, resnet_metrics['mAP@0.5:0.95'], 
                    marker='o', color=resnet_color, label='ResNet50', linewidth=2)
    axes[1, 1].plot(fastervit_epochs, fastervit_metrics['mAP@0.5:0.95'], 
                    marker='s', color=fastervit_color, label='FasterViT-0', linewidth=2)
    axes[1, 1].set_title('mAP@0.5:0.95', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('mAP')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'backbone_comparison_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Performance improvement analysis
    create_performance_analysis(resnet_data, fastervit_data, output_dir)
    
def create_performance_analysis(resnet_data, fastervit_data, output_dir):
    """Create performance improvement analysis"""
    resnet_epochs, resnet_metrics = resnet_data
    fastervit_epochs, fastervit_metrics = fastervit_data
    
    # Calculate final performance metrics
    final_metrics = {}
    for metric in ['precision@IoU=0.5', 'recall@100', 'f1_score', 'mAP@0.5:0.95']:
        resnet_final = resnet_metrics[metric][-1] if resnet_metrics[metric] else 0
        fastervit_final = fastervit_metrics[metric][-1] if fastervit_metrics[metric] else 0
        improvement = ((fastervit_final - resnet_final) / resnet_final * 100) if resnet_final > 0 else 0
        
        final_metrics[metric] = {
            'resnet': resnet_final,
            'fastervit': fastervit_final,
            'improvement': improvement
        }
    
    # Create performance comparison bar chart
    plt.figure(figsize=(14, 8))
    metrics_names = ['Precision@0.5', 'Recall@100', 'F1-Score', 'mAP@0.5:0.95']
    resnet_values = [final_metrics[metric]['resnet'] for metric in ['precision@IoU=0.5', 'recall@100', 'f1_score', 'mAP@0.5:0.95']]
    fastervit_values = [final_metrics[metric]['fastervit'] for metric in ['precision@IoU=0.5', 'recall@100', 'f1_score', 'mAP@0.5:0.95']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, resnet_values, width, label='ResNet50', color='#1f77b4', alpha=0.8)
    bars2 = plt.bar(x + width/2, fastervit_values, width, label='FasterViT-0', color='#ff7f0e', alpha=0.8)
    
    plt.title('Final Performance Comparison: ResNet50 vs FasterViT-0', fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(x, metrics_names)
    plt.legend(fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary report
    with open(output_dir / 'comparison_summary.txt', 'w') as f:
        f.write("Backbone Comparison Summary: ResNet50 vs FasterViT-0\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Training Epochs: ResNet50 ({len(resnet_epochs)}), FasterViT-0 ({len(fastervit_epochs)})\n\n")
        
        f.write("Final Performance Metrics:\n")
        f.write("-" * 25 + "\n")
        for metric_key, display_name in [
            ('precision@IoU=0.5', 'Precision@IoU=0.5'),
            ('recall@100', 'Recall@100'),
            ('f1_score', 'F1-Score'),
            ('mAP@0.5:0.95', 'mAP@0.5:0.95')
        ]:
            metrics = final_metrics[metric_key]
            f.write(f"{display_name}:\n")
            f.write(f"  ResNet50:    {metrics['resnet']:.4f}\n")
            f.write(f"  FasterViT-0: {metrics['fastervit']:.4f}\n")
            f.write(f"  Improvement: {metrics['improvement']:+.2f}%\n\n")
        
        # Calculate parameter comparison
        # Note: These values are extracted from the logs
        resnet_params = 42710790  # From ResNet50 logs
        fastervit_params = 49981998  # From FasterViT logs
        param_increase = ((fastervit_params - resnet_params) / resnet_params * 100)
        
        f.write("Model Complexity:\n")
        f.write("-" * 15 + "\n")
        f.write(f"ResNet50 Parameters:    {resnet_params:,}\n")
        f.write(f"FasterViT-0 Parameters: {fastervit_params:,}\n")
        f.write(f"Parameter Increase:     {param_increase:+.2f}%\n\n")
        
        f.write("Key Observations:\n")
        f.write("-" * 15 + "\n")
        
        best_metric = max(final_metrics.keys(), key=lambda x: final_metrics[x]['improvement'])
        worst_metric = min(final_metrics.keys(), key=lambda x: final_metrics[x]['improvement'])
        
        f.write(f"• Best improvement: {best_metric} ({final_metrics[best_metric]['improvement']:+.2f}%)\n")
        f.write(f"• Worst improvement: {worst_metric} ({final_metrics[worst_metric]['improvement']:+.2f}%)\n")
        
        avg_improvement = np.mean([final_metrics[m]['improvement'] for m in final_metrics.keys()])
        f.write(f"• Average improvement: {avg_improvement:+.2f}%\n")
        
        if avg_improvement > 0:
            f.write("• FasterViT-0 shows overall better performance than ResNet50\n")
        else:
            f.write("• ResNet50 shows overall better performance than FasterViT-0\n")
    
    print(f"\nComparison analysis saved to: {output_dir}")
    print(f"Generated files:")
    print(f"  - Individual metric comparisons (precision, recall, f1_score, mAP)")
    print(f"  - Combined overview plot")
    print(f"  - Final performance comparison")
    print(f"  - Summary report")

def main():
    parser = argparse.ArgumentParser(description='Compare RT-DETR backbone performance')
    parser.add_argument('--resnet-logs', type=str, 
                       default='graph/logs_resnet50.txt',
                       help='Path to ResNet50 training logs')
    parser.add_argument('--fastervit-logs', type=str, 
                       default='graph/logs_fastervit.txt',
                       help='Path to FasterViT-0 training logs')
    parser.add_argument('--output-dir', type=str, 
                       default='backbone_comparison',
                       help='Output directory for plots and analysis')
    
    args = parser.parse_args()
    
    print("Loading training logs...")
    try:
        resnet_data = load_training_logs(args.resnet_logs)
        print(f"✓ ResNet50 logs loaded: {len(resnet_data[0])} epochs")
    except Exception as e:
        print(f"✗ Error loading ResNet50 logs: {e}")
        return
    
    try:
        fastervit_data = load_training_logs(args.fastervit_logs)
        print(f"✓ FasterViT-0 logs loaded: {len(fastervit_data[0])} epochs")
    except Exception as e:
        print(f"✗ Error loading FasterViT-0 logs: {e}")
        return
    
    print("Creating comparison plots...")
    create_comparison_plots(resnet_data, fastervit_data, args.output_dir)
    print("✓ Comparison analysis complete!")

if __name__ == "__main__":
    main()