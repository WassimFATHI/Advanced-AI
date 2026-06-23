from __future__ import annotations
import logging, os
import torch

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/current")

_model = None
_tokenizer = None
_image_processor = None
_image_string = None
_device = "cpu"


def load_model() -> None:
    global _model, _tokenizer, _image_processor, _image_string, _device
    if _model is not None:
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading checkpoint from %s on %s", CHECKPOINT_PATH, _device)

    _model = VisionLanguageModel.from_pretrained(CHECKPOINT_PATH).to(_device)
    _model.eval()

    cfg = _model.cfg
    _tokenizer = get_tokenizer(cfg.lm.tokenizer, cfg.image_token)
    _image_processor = get_image_processor(cfg.vit.img_size)
    _image_string = get_image_string(cfg.projector.image_token_length, cfg.image_token)

    n_params = sum(p.numel() for p in _model.parameters())
    logger.info("Model loaded — %s parameters", f"{n_params:,}")


def generate(messages, image=None, max_new_tokens=128, temperature=0.7) -> str:
    if _model is None:
        load_model()

    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        "Describe the image.",
    )

    if image is None:
        return "Merci d'envoyer une image à décrire."

    img = image.convert("RGB")
    pixel_values = _image_processor(img).unsqueeze(0).to(_device)

    prompt_text = _image_string + last_user
    chat = [{"role": "user", "content": prompt_text}]
    encoded = _tokenizer.apply_chat_template(
        [chat], tokenize=True, add_generation_prompt=True
    )
    input_ids = torch.tensor(encoded).to(_device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        gen = _model.generate(
            input_ids, pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            greedy=(temperature <= 0.0),
            temperature=temperature if temperature > 0.0 else 1.0,
        )

    text = _tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return text.strip()
