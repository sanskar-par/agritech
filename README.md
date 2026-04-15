# Hugging Face Image Classification Demo

This project is ready for Hugging Face Spaces with Gradio.

## Files

- `app.py`: Gradio app that loads a `.pt` model and runs image classification.
- `requirements.txt`: Python dependencies for Spaces.
- `labels.txt` (optional): One class name per line.

## Notes

- By default, the app picks the first `.pt` file in the project root.
- You can override this with `MODEL_PATH` environment variable.
- If your model uses custom normalization/image size, set:
  - `IMG_SIZE` (default `224`)
  - `NORM_MEAN` (default `0.485,0.456,0.406`)
  - `NORM_STD` (default `0.229,0.224,0.225`)

## Deploy on Hugging Face Spaces

1. Create a new Space with **Gradio** SDK.
2. Upload this folder contents (`app.py`, model `.pt`, `requirements.txt`, optional `labels.txt`).
3. Space will install dependencies and automatically run `app.py`.

If the model file is only a `state_dict` checkpoint, re-export it as a full model or TorchScript for inference.
