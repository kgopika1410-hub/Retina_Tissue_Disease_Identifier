# Diabetic Retinopathy Classifier - Deployment Guide

## Prerequisites
- GitHub account with the repository
- Render account (free tier available at https://render.com)

## Step 1: Push to GitHub

```bash
# Initialize git repo (if not already done)
git init
git add .
git commit -m "Initial commit: DR classifier for Render deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/retina-tissue.git
git push -u origin main
```

## Step 2: Deploy on Render

### Option A: Using render.yaml (Recommended)
1. Go to https://dashboard.render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Select the repository and branch (main)
5. Render will auto-detect `render.yaml` and use those settings
6. Click "Create Web Service"

### Option B: Manual Configuration
1. Go to https://dashboard.render.com
2. Click "New +" → "Web Service"  
3. Connect your GitHub repository
4. Configure:
   - **Name**: dr-classifier
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python dr_inference_app.py`
   - **Plan**: Free (or paid as needed)

## Step 3: Configure Models (IMPORTANT)

The trained models are not included in the GitHub repo (too large). You have two options:

### Option A: Upload via Render Shell (Recommended for small models)
```bash
# Via Render dashboard → Your Service → Shell
cd /opt/render/project/src
# Then upload/download model files
```

### Option B: Use Cloud Storage (Recommended for production)
1. Upload models to AWS S3, Google Cloud Storage, or similar
2. Create a `setup.sh` script to download them during build:

```bash
#!/bin/bash
# setup.sh
if [ ! -d "outputs_improved" ]; then
  mkdir -p outputs_improved
  aws s3 cp s3://your-bucket/best_model.keras outputs_improved/
fi
```

3. Update `render.yaml`:
```yaml
buildCommand: "bash setup.sh && pip install -r requirements.txt"
```

## Step 4: Environment Variables

Set in Render Dashboard → Your Service → Environment:

```
GRADIO_SERVER_PORT=10000
GRADIO_SERVER_NAME=0.0.0.0
RENDER=true
```

## Step 5: Access Your App

Once deployed, Render will provide a URL like:
```
https://dr-classifier-xxxx.onrender.com
```

Visit that URL to use your Diabetic Retinopathy classifier!

## Troubleshooting

### Model Not Found Error
- Ensure the model files are in the `outputs_improved/` folder
- Check Render logs: Dashboard → Your Service → Logs

### Port Binding Error
- Render automatically assigns ports; the app now uses dynamic port assignment

### Build Failures
- Check build logs in Render dashboard
- Verify all dependencies in `requirements.txt` are compatible

## Local Testing Before Deployment

```bash
# Set environment variable to simulate Render
$env:RENDER="true"
$env:GRADIO_SERVER_PORT="7860"

# Run the app
python dr_inference_app.py
```

## Notes

- The free tier on Render has 15-minute idle timeout; services spin down when unused
- For production, upgrade to a paid plan for better performance and uptime
- Models should be version-controlled via cloud storage, not GitHub
- Total repo size should stay under 500MB (excluding large model files)
