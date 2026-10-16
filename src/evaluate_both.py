import csv
from pathlib import Path
from typing import cast

import open_clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset import RSICD


def evaluate_model(model, text_features, classes, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    n = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    matrix = [[0] * n for _ in range(n)]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch[0]
            labels = batch[2]

            images = images.to(device)

            image_feats = model.encode_image(images)
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

            # similarity (batch x num_classes)
            sims = image_feats @ text_features.t()
            preds = sims.argmax(dim=1).cpu().tolist()

            # labels may be list of strings
            for pred_idx, true_label in zip(preds, labels):
                if true_label is None:
                    continue
                true_idx = class_to_idx.get(true_label)
                if true_idx is None:
                    # try case-insensitive match
                    true_idx = class_to_idx.get(str(true_label))
                if true_idx is None:
                    continue
                total += 1
                if true_idx == pred_idx:
                    correct += 1
                matrix[true_idx][pred_idx] += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, matrix


def build_text_features(model, tokenizer, classes, device):
    # Build prompts and encode text features for zero-shot classification
    prompts = [f"A satellite image of {c}" for c in classes]
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats


def save_matrix_csv(path: Path, classes, matrix):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + classes)
        for cname, row in zip(classes, matrix):
            writer.writerow([cname] + row)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = cast(open_clip.model.CLIP, model)  # For better type checking
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    dataset = RSICD(
        img_dir=utils.IMG_DIR,
        txtclasses_dir=utils.TXTCLASSES_DIR,
        json_path=utils.JSON_PATH,
        preprocess=preprocess,
        tokenizer=None,
        split="C",
        caption_mode="first",
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    classes = utils.classes
    utils.RESULTS_DIR.mkdir(exist_ok=True)

    text_feats = build_text_features(model, tokenizer, classes, device)

    # Evaluate pretrained (zero-shot)
    print("Evaluating pretrained laion2b_s34b_b79k zero-shot...")
    acc_pre, matrix_pre = evaluate_model(model, text_feats, classes, dataloader, device)
    print(f"Pretrained zero-shot accuracy: {acc_pre:.4f}")
    save_matrix_csv(utils.RESULTS_DIR / "pretrained_crosstab.csv", classes, matrix_pre)

    # If finetuned checkpoint exists, load and evaluate
    ckpt = utils.CHECKPOINT_DIR / "finetuned.pt"
    if ckpt.exists():
        print(f"\nLoading finetuned checkpoint from {ckpt} and evaluating...")
        finetuned = utils.load_model_checkpoint(model, ckpt, map_location=device)
        finetuned = finetuned.to(device)
        # Text encoder was finetuned too, so class prompts must be re-encoded
        text_feats_f = build_text_features(finetuned, tokenizer, classes, device)
        acc_f, matrix_f = evaluate_model(
            finetuned, text_feats_f, classes, dataloader, device
        )
        print(f"Finetuned accuracy: {acc_f:.4f}")
        save_matrix_csv(utils.RESULTS_DIR / "finetuned_crosstab.csv", classes, matrix_f)


if __name__ == "__main__":
    main()
