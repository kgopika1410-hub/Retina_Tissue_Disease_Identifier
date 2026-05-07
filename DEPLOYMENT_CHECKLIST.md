# Render Deployment Checklist

Use this checklist to ensure everything is ready before deploying to Render.

## Before You Start

- [ ] You have a GitHub account (free at github.com)
- [ ] You have a Render account (free at render.com)
- [ ] All code is working locally (`python dr_inference_app.py`)
- [ ] Models exist in `outputs_improved/` or `outputs_run2/`

## Local Preparation

- [ ] Run `python setup_deployment.py` (shows any issues)
- [ ] All Python files have correct syntax
- [ ] `requirements.txt` is up-to-date
- [ ] `.gitignore` excludes large files
- [ ] No hardcoded secrets or API keys in code

## GitHub Setup

- [ ] Create GitHub repository (retina-tissue)
- [ ] Run `git init` (if first time)
- [ ] Run `git add .`
- [ ] Run `git commit -m "Initial commit"`
- [ ] Run `git branch -M main`
- [ ] Run `git remote add origin https://github.com/USERNAME/retina-tissue.git`
- [ ] Run `git push -u origin main`
- [ ] Verify code appears on GitHub

## Files That Should Exist

These files are needed for Render deployment:
- [ ] `Procfile` - tells Render how to start the app
- [ ] `render.yaml` - Render configuration
- [ ] `requirements.txt` - Python dependencies
- [ ] `.gitignore` - prevent uploading large files
- [ ] `.github/workflows/validate.yml` - CI/CD validation
- [ ] `README.md` - Project documentation
- [ ] `DEPLOYMENT.md` - Deployment guide
- [ ] `GITHUB_DEPLOYMENT.md` - GitHub/Render quick start

## Render Deployment

- [ ] Sign into https://render.com with GitHub
- [ ] Click "New +" → "Web Service"
- [ ] Select your `retina-tissue` repository
- [ ] Render detects `render.yaml` (auto-configured)
- [ ] Click "Create Web Service"
- [ ] Wait for deployment (typically 2-5 minutes)
- [ ] Check build logs for errors
- [ ] Once live, you get a URL like: `https://dr-classifier-xxxx.onrender.com`

## Model Handling (Choose One)

### Quick Option (Free Tier):
- [ ] After deployment, go to Render dashboard
- [ ] Click service → "Shell" tab
- [ ] Upload or download model files manually

### Production Option (Recommended):
- [ ] Upload models to AWS S3 or Google Cloud Storage
- [ ] Create `download_models.sh` script
- [ ] Update `render.yaml` buildCommand
- [ ] Set AWS credentials in Render environment variables

## Testing on Render

- [ ] Visit your Render URL
- [ ] Upload a test retinal image
- [ ] Verify prediction works
- [ ] Check that confidence scores are reasonable
- [ ] Test enhancement options (if needed)

## Monitoring

- [ ] Check Render logs regularly
- [ ] Set up alerts (Render Pro feature)
- [ ] Monitor app performance
- [ ] Verify models load correctly

## Troubleshooting

### If models not found:
- [ ] Check `/opt/render/project/src` directory in Shell
- [ ] Verify models downloaded correctly
- [ ] Check build logs for download errors

### If app won't start:
- [ ] Check render logs for Python errors
- [ ] Verify all dependencies in requirements.txt
- [ ] Ensure binding to 0.0.0.0 (already set in code)

### If predictions fail:
- [ ] Check model file permissions
- [ ] Verify model format (.keras)
- [ ] Check available memory (free tier has limits)

## After Deployment

- [ ] Share URL with stakeholders
- [ ] Document how to use the app
- [ ] Set up monitoring/alerts
- [ ] Plan for paid tier if needed (free tier sleeps after 15 min idle)

## Quick Command Reference

```bash
# Check deployment readiness
python setup_deployment.py

# Push code to GitHub
git add .
git commit -m "message"
git push origin main

# Monitor Render service
# (via dashboard or CLI if you install render-cli)
```

## Support Links

- Render Docs: https://render.com/docs
- Gradio Docs: https://www.gradio.app/docs/
- TensorFlow Serving: https://www.tensorflow.org/serving

---

**Last Updated**: $(date)
**Version**: 1.0
