#!/usr/bin/env python
"""
Setup script to prepare the project for deployment.
Checks for models, validates dependencies, and provides deployment guidance.
"""

import os
import sys
from pathlib import Path

def check_models():
    """Check if trained models exist."""
    root_dir = Path(__file__).resolve().parent
    candidates = [
        root_dir / "outputs_improved" / "best_model.keras",
        root_dir / "outputs_improved" / "final_model.keras",
        root_dir / "outputs_run2" / "best_model.keras",
        root_dir / "outputs_run2" / "final_model.keras",
    ]
    
    found = False
    for path in candidates:
        if path.exists():
            print(f"✓ Found model: {path.relative_to(root_dir)}")
            found = True
    
    if not found:
        print("\n⚠ WARNING: No trained models found!")
        print("Models should be in 'outputs_improved/' or 'outputs_run2/'")
        print("For deployment on Render, you need to:")
        print("  1. Upload models to cloud storage (AWS S3, GCS, etc.)")
        print("  2. Use a setup script to download them during build")
        print("  3. Or manually upload them via Render Shell after deployment")
        return False
    return True

def check_dependencies():
    """Check if major dependencies are installed."""
    required = [
        'tensorflow',
        'gradio',
        'fastapi',
        'opencv',
        'numpy',
        'pandas',
    ]
    
    print("\nChecking dependencies:")
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True

def check_git():
    """Check if git is initialized."""
    root_dir = Path(__file__).resolve().parent
    git_dir = root_dir / ".git"
    
    if git_dir.exists():
        print("\n✓ Git repository already initialized")
        return True
    else:
        print("\n⚠ Git repository not initialized")
        print("Initialize with: git init")
        return False

def print_deployment_steps():
    """Print deployment instructions."""
    print("\n" + "="*60)
    print("DEPLOYMENT TO RENDER")
    print("="*60)
    print("""
1. Initialize Git repository (if needed):
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main

2. Push to GitHub:
   git remote add origin https://github.com/YOUR_USERNAME/retina-tissue.git
   git push -u origin main

3. Create Render Service:
   - Go to https://dashboard.render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect render.yaml configuration

4. Upload Models:
   For the free tier, you can:
   - Use Render Shell to upload models directly
   - Or host models on AWS S3/GCS and download during build

5. Visit Your App:
   https://your-service-name.onrender.com
""")

def main():
    print("🔍 Preparing Diabetic Retinopathy Classifier for Deployment\n")
    
    models_ok = check_models()
    deps_ok = check_dependencies()
    git_ok = check_git()
    
    print_deployment_steps()
    
    if not models_ok:
        print("\n⚠️  Models need to be set up before deployment")
        sys.exit(1)
    
    if not deps_ok:
        print("\n⚠️  Install missing dependencies before deployment")
        sys.exit(1)
    
    print("\n✅ All checks passed! Ready for deployment.")
    print("\nNext steps:")
    print("1. Commit changes: git add -A && git commit -m 'Prepare for deployment'")
    print("2. Push to GitHub: git push")
    print("3. Deploy on Render")

if __name__ == "__main__":
    main()
