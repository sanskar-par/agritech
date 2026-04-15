import os
from pathlib import Path
from typing import Any

import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image


WORKDIR = Path(__file__).resolve().parent


def _parse_floats(env_name: str, default: list[float]) -> list[float]:
    raw = os.getenv(env_name)
    if not raw:
        return default
    try:
        values = [float(x.strip()) for x in raw.split(",")]
        if len(values) != 3:
            return default
        return values
    except Exception:
        return default


def find_model_path() -> Path:
    env_model = os.getenv("MODEL_PATH")
    if env_model:
        candidate = WORKDIR / env_model
        if candidate.exists():
            return candidate
        absolute_candidate = Path(env_model)
        if absolute_candidate.exists():
            return absolute_candidate

    models = sorted(WORKDIR.glob("*.pt"))
    if not models:
        raise FileNotFoundError("No .pt model file found in project root.")
    return models[0]


def load_class_names() -> list[str] | None:
    labels_file = WORKDIR / "labels.txt"
    if not labels_file.exists():
        return None

    labels = [line.strip() for line in labels_file.read_text(encoding="utf-8").splitlines()]
    labels = [x for x in labels if x]
    return labels if labels else None


def resolve_output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)) and output:
        for item in output:
            if isinstance(item, torch.Tensor):
                return item

    if isinstance(output, dict):
        for key in ("logits", "pred", "output", "outputs"):
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value

    raise ValueError("Model output format is unsupported for classification.")


def load_model(model_path: Path, device: torch.device) -> tuple[torch.nn.Module, str]:
    # Try TorchScript first for maximum compatibility with .pt exports.
    try:
        scripted = torch.jit.load(str(model_path), map_location=device)
        scripted.eval()
        return scripted, "torchscript"
    except Exception:
        pass

    checkpoint = torch.load(str(model_path), map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        checkpoint.eval()
        return checkpoint, "nn.Module"

    if isinstance(checkpoint, dict):
        model_obj = checkpoint.get("model")
        if isinstance(model_obj, torch.nn.Module):
            model_obj.eval()
            return model_obj, "checkpoint:model"

        if "state_dict" in checkpoint:
            raise RuntimeError(
                "This .pt file contains only a state_dict. "
                "Please export/save a full model or TorchScript model for inference."
            )

    raise RuntimeError("Could not load a runnable model from this .pt file.")


def build_transform() -> T.Compose:
    image_size = int(os.getenv("IMG_SIZE", "224"))
    mean = _parse_floats("NORM_MEAN", [0.485, 0.456, 0.406])
    std = _parse_floats("NORM_STD", [0.229, 0.224, 0.225])

    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def make_predict_fn(model: torch.nn.Module, device: torch.device, class_names: list[str] | None):
    transform = build_transform()

    def predict(image: Image.Image):
        if image is None:
            raise gr.Error("Please upload an image.")

        image = image.convert("RGB")
        x = transform(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            raw_output = model(x)
            logits = resolve_output_tensor(raw_output)

            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            if logits.ndim > 2:
                logits = logits.flatten(start_dim=1)

            probs = torch.softmax(logits, dim=1)[0].detach().cpu()

        num_classes = probs.shape[0]
        labels = class_names if class_names and len(class_names) == num_classes else [f"class_{i}" for i in range(num_classes)]

        top_k = min(5, num_classes)
        values, indices = torch.topk(probs, k=top_k)

        confidence_map = {labels[int(idx)]: float(val) for idx, val in zip(indices, values)}
        best_idx = int(indices[0])
        best_label = labels[best_idx]
        best_confidence = float(values[0])

        result_text = f"Prediction: {best_label} (confidence: {best_confidence:.4f})"
        return result_text, confidence_map

    return predict


def build_demo() -> gr.Blocks:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_class_names()

    load_error = None
    model_path = None
    model_kind = "unknown"
    predict = None

    try:
        model_path = find_model_path()
        model, model_kind = load_model(model_path, device)
        predict = make_predict_fn(model, device, class_names)
    except Exception as exc:
        load_error = str(exc)

    if predict is None:
        def predict(_image: Image.Image):
            raise gr.Error(f"Model is not ready: {load_error}")

    with gr.Blocks(title="Image Classification Demo") as demo:
        model_name = model_path.name if model_path else "not found"
        gr.Markdown(
            "# Image Classification Demo\n"
            f"Loaded model: `{model_name}`  \n"
            f"Model type: `{model_kind}`  \n"
            f"Device: `{device}`"
        )

        if load_error:
            gr.Markdown(f"**Startup error:** `{load_error}`")

        with gr.Row():
            image_input = gr.Image(type="pil", label="Input Image")
            label_output = gr.Label(num_top_classes=5, label="Top Classes")

        text_output = gr.Textbox(label="Predicted Class", interactive=False)
        run_btn = gr.Button("Classify")

        run_btn.click(fn=predict, inputs=image_input, outputs=[text_output, label_output])
        image_input.change(fn=predict, inputs=image_input, outputs=[text_output, label_output])

        gr.Markdown(
            "Optional files/settings:\n"
            "- Add `labels.txt` (one class name per line) to show real class names.\n"
            "- Set `IMG_SIZE`, `NORM_MEAN`, `NORM_STD`, or `MODEL_PATH` as environment variables if needed."
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
