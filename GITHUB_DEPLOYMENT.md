# GitHub & Render Deployment - Quick Start

## Step 1: Create a GitHub Repository

1. Go to https://github.com/new
2. Create a new repository named `retina-tissue` (or your choice)
3. Choose **Public** (unless you want private)
4. Skip initializing with README (you already have one)
5. Click **Create repository**

## Step 2: Push Your Local Code to GitHub

### First Time Setup
```bash
# Navigate to your project directory
cd d:\Retina_Tissue

# Initialize git (if not already done)
git init

# Add all files
git add .

# Make first commit
git commit -m "Initial commit: Diabetic Retinopathy Classifier"

# Rename branch to main
git branch -M main

# Add remote GitHub repository
git remote add origin https://github.com/YOUR_USERNAME/retina-tissue.git

# Push to GitHub
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

### Future Updates
```bash
# Make changes to files
# Then:
git add .
git commit -m "Description of changes"
git push origin main
```

## Step 3: Deploy on Render

### Simple Deployment (Recommended)

1. Go to https://render.com
2. Sign up with GitHub account
3. Click **New +** → **Web Service**
4. Click **Connect a repository**
5. Select your `retina-tissue` repository
6. Render will auto-detect `render.yaml` and use those settings
7. Click **Create Web Service**
8. Wait for deployment (2-5 minutes)
9. Visit your app URL: `https://your-service-name.onrender.com`

### Manual Configuration

If auto-detection doesn't work:
- **Name**: `dr-classifier`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python dr_inference_app.py`
- **Plan**: Free (or paid)

## Step 4: Add Your Models

Since the models are too large for GitHub, you need to add them to Render separately.

### Option A: Upload via Render Shell (Easy)
```bash
# After deployment, go to your service dashboard
# Click "Shell" tab
# Run:
cd /opt/render/project/src
mkdir -p outputs_improved
# Then upload your model file via SFTP or download from cloud storage
```

### Option B: Use Cloud Storage (Recommended for Production)

1. **Upload model to AWS S3**:
   ```bash
   # Install AWS CLI
   pip install awscli
   
   # Configure AWS
   aws configure
   
   # Upload model
   aws s3 cp outputs_improved/best_model.keras s3://your-bucket/models/
   ```

2. **Update Render environment variables**:
   - Add `AWS_ACCESS_KEY_ID`
   - Add `AWS_SECRET_ACCESS_KEY`

3. **Create `download_models.sh`**:
   ```bash
   #!/bin/bash
   mkdir -p outputs_improved
   aws s3 cp s3://your-bucket/models/best_model.keras outputs_improved/
   ```

4. **Update `render.yaml`**:
   ```yaml
   buildCommand: "bash download_models.sh && pip install -r requirements.txt"
   ```

## Troubleshooting

### Models Not Found After Deployment
- Check Render logs for errors
- Make sure models are uploaded/downloaded before app starts
- Verify model file paths in `dr_model_utils.py`

### Build Fails
- Check build logs in Render dashboard
- Verify `requirements.txt` contains all dependencies
- Test locally: `pip install -r requirements.txt`

### App Crashes on Render
- Check service logs
- Verify models are accessible
- Check that the app binds to `0.0.0.0` (already configured)

## Useful Links

- **Render Documentation**: https://render.com/docs
- **Gradio Hosting**: https://gradio.app/sharing/
- **FastAPI Deployment**: https://fastapi.tiangolo.com/deployment/

## Environment Variables (Set in Render Dashboard)

```
GRADIO_SERVER_PORT=10000
GRADIO_SERVER_NAME=0.0.0.0
RENDER=true
```

## Next: Verify Before Deployment

Run this locally to check everything is ready:
```bash
python setup_deployment.py
```

This will verify:
- ✓ Models exist
- ✓ Dependencies installed
- ✓ Git is initialized
