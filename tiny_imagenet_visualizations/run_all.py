"""
Tiny ImageNet Visualization Suite
==================================

Run all visualizations with a single command.

Usage: python run_all.py

This will generate:
- ./outputs/confusion_matrix.png
- ./outputs/per_class_accuracy.png  
- ./outputs/grad_cam_gallery.png
- ./outputs/embedding_tsne.png
- ./outputs/embedding_umap.png (if umap-learn installed)
- ./outputs/top_errors_gallery.png
"""

import os
import sys
import subprocess


def run_script(script_name: str):
    """Run a visualization script."""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(os.path.dirname(__file__))  # Project root
    )
    
    if result.returncode != 0:
        print(f"⚠️  {script_name} failed with code {result.returncode}")
        return False
    return True


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║     TINY IMAGENET VISUALIZATION SUITE                   ║
╠══════════════════════════════════════════════════════════╣
║  This will generate comprehensive model analysis plots.  ║
║  Estimated time: 5-10 minutes (depending on GPU)         ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    scripts = [
        ("confusion_matrix.py", "Confusion Matrix"),
        ("per_class_accuracy.py", "Per-Class Accuracy"),
        ("top_errors.py", "Top Errors Gallery"),
        ("grad_cam.py", "Grad-CAM Attention Maps"),
        ("tsne_embeddings.py", "t-SNE/UMAP Embeddings"),
    ]
    
    results = []
    for script, name in scripts:
        success = run_script(script)
        results.append((name, success))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    print("\nOutputs saved to: ./outputs/")
    print("\n📊 Open the PNG files to explore your model's behavior!")


if __name__ == "__main__":
    main()
