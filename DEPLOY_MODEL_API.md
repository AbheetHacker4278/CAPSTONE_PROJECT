# Deployment Guide: Model API

To bypass Render's 512MB RAM limit, follow these steps to deploy your Model API separately.

## Option 1: Hugging Face Spaces (Recommended - Free 16GB RAM)

1. **Create a New Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces).
   - Click "Create new Space".
   - Name it (e.g., `breast-cancer-model-api`).
   - Select **Docker**.

2. **Upload Files**:
   - `model_api.py` (Rename it to `app.py` for Hugging Face or specify it in Dockerfile)
   - `model_utils.py`
   - `requirements.txt` (Pinned to `tensorflow==2.15.0`)
   - `weights/` folder (containing `modeldense1.h5`)

3. **Add Dockerfile**:
   - I have created `Dockerfile.model` in your project.
   - When uploading to Hugging Face, rename it to just `Dockerfile`.
   - It is pre-configured with all necessary system dependencies (OpenCV, etc.).

4. **Update Render Environment Variable**:
   - Once your Space is "Running", copy its URL (e.g., `https://user-space.hf.space`).
   - Go to your Render Dashboard -> Environment.
   - Add `REMOTE_MODEL_URL` = `https://user-space.hf.space`.

## Option 2: Run Locally (For Testing)

1. **Terminal 1** (Model):
   ```bash
   python model_api.py
   ```
2. **Terminal 2** (App):
   - Add `REMOTE_MODEL_URL=http://localhost:5001` to `.env`
   - Run:
     ```bash
     python app.py
     ```

## Troubleshooting
- If the Model API crashes, check the logs for "Out of Memory" (OOM).
- Ensure `weights/modeldense1.h5` is correctly uploaded.
- Large files (>50MB) on Hugging Face require **Git LFS**.
