
"""Evaluate test loss on Flickr30k test set.

Usage:
    python eval_test.py --checkpoint_path checkpoints/current/best_step5000 \
                       --dataset_local_path /work/formation/tpirtlmrdk/datasets/flickr30k

Reports the test loss on the true test set (raw["test"]).
"""

import argparse
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from models.config import VLMConfig, TrainConfig
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor
from data.dataset import FlickrDataset
from data.collator import VQACollator


def evaluate_test(checkpoint_path: str, dataset_local_path: str, batch_size: int = 32):
    """Evaluate test loss on the true test set."""
    
    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # ── Load configs ──────────────────────────────────────────────────────────
    vlm_cfg = VLMConfig()
    train_cfg = TrainConfig(batch_size=batch_size)
    
    # ── Load tokenizer & image processor ──────────────────────────────────────
    tokenizer = get_tokenizer(vlm_cfg.lm.tokenizer, vlm_cfg.image_token)
    image_processor = get_image_processor(vlm_cfg.vit.img_size)
    
    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {checkpoint_path}")
    model = VisionLanguageModel.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    
    # ── Load test dataset ─────────────────────────────────────────────────────
    print(f"Loading dataset from {dataset_local_path}")
    raw = load_from_disk(dataset_local_path)
    
    if "test" not in raw:
        raise ValueError(
            f"No 'test' split found in dataset. "
            f"Available splits: {list(raw.keys())}"
        )
    
    test_ds = raw["test"]
    print(f"Test set size: {len(test_ds)}")
    
    # ── Create test loader ────────────────────────────────────────────────────
    test_dataset = FlickrDataset(test_ds, tokenizer, image_processor, vlm_cfg)
    collator = VQACollator(tokenizer, max_length=train_cfg.max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=1,
        pin_memory=True,
    )
    
    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("Evaluating on test set...")
    test_losses = []
    
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    autocast_ctx = torch.autocast(device_type=device.type, dtype=autocast_dtype)
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            with autocast_ctx:
                _, loss = model(
                    batch["input_ids"].to(device),
                    batch["pixel_values"].to(device),
                    batch["attention_mask"].to(device),
                    batch["labels"].to(device),
                )
            
            if loss is not None:
                test_losses.append(loss.item())
            
            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1} batches...")
    
    avg_test_loss = sum(test_losses) / len(test_losses) if test_losses else float("nan")
    
    print(f"\n{'='*60}")
    print(f"TEST LOSS: {avg_test_loss:.4f}")
    print(f"  (computed over {len(test_losses)} batches, {len(test_losses) * batch_size} images)")
    print(f"{'='*60}")
    
    return avg_test_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate test loss on Flickr30k")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/current/best_step5000",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--dataset_local_path",
        type=str,
        default="/work/formation/tpirtlmrdk/datasets/flickr30k",
        help="Path to the Flickr30k dataset directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    
    args = parser.parse_args()
    
    evaluate_test(
        checkpoint_path=args.checkpoint_path,
        dataset_local_path=args.dataset_local_path,
        batch_size=args.batch_size,
    )
