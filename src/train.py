import csv
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import open_clip
import torch
from open_clip.loss import ClipLoss
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import RSICD
from utils import CHECKPOINT_DIR, IMG_DIR, JSON_PATH, RESULTS_DIR, TXTCLASSES_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model = cast(open_clip.model.CLIP, model)  # For better type checking
model.train()
model = model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Two captioning strategies exist:
# - training: random, val: first (to train on provided captions)
# - training: label, val: label (to train on class labels as captions)
train_base_dataset = RSICD(
    img_dir=IMG_DIR,
    txtclasses_dir=TXTCLASSES_DIR,
    json_path=JSON_PATH,
    preprocess=preprocess,
    tokenizer=tokenizer,
    split="B",
    caption_mode="random",
)

val_base_dataset = RSICD(
    img_dir=IMG_DIR,
    txtclasses_dir=TXTCLASSES_DIR,
    json_path=JSON_PATH,
    preprocess=preprocess,
    tokenizer=tokenizer,
    split="B",
    caption_mode="all",
)

generator = torch.Generator().manual_seed(12345)
train_subset, val_subset = random_split(
    train_base_dataset, [0.8, 0.2], generator=generator
)

train = torch.utils.data.Subset(train_base_dataset, train_subset.indices)
val = torch.utils.data.Subset(val_base_dataset, val_subset.indices)

train_loader = DataLoader(train, batch_size=32, shuffle=True)
val_loader = DataLoader(val, batch_size=32, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = ClipLoss()

num_epochs = 10
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
checkpoint_path = CHECKPOINT_DIR / "finetuned.pt"
best_val_loss = float("inf")

avg_train_losses = []
avg_val_losses = []

for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    train_loss = 0.0
    for image, caption, _ in pbar:
        optimizer.zero_grad()

        image = image.to(device)
        caption = caption.to(device)

        image_feat = model.encode_image(image)
        text_feat = model.encode_text(caption)

        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        loss = loss_fn(image_feat, text_feat, model.logit_scale.exp())

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for image, captions, _ in val_loader:
            image = image.to(device)
            captions = captions.to(device)

            image_feat = model.encode_image(image)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            # evaluating over all 5 captions for given image
            batch_loss = 0.0
            for caption_idx in range(5):
                curr_caption = captions[:, caption_idx, :]

                text_feat = model.encode_text(curr_caption)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

                loss = loss_fn(image_feat, text_feat, model.logit_scale.exp())
                batch_loss += loss

            batch_loss = batch_loss / 5.0
            val_loss += batch_loss.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_losses.append(avg_val_loss)
    print(
        f"Epoch {epoch + 1}, training_loss: {avg_train_loss:.4f}, validation loss: {avg_val_loss:.4f}",
        end="",
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(
            {
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print(" (New best, saving checkpoint)")
    else:
        print("")

print("Losses per epoch:")
for epoch in range(num_epochs):
    print(
        f"Epoch {epoch + 1}: Train Loss = {avg_train_losses[epoch]:.4f}, Val Loss = {avg_val_losses[epoch]:.4f}"
    )

# Save losses to CSV
losses_csv_path = RESULTS_DIR / "finetuned_losses.csv"
with losses_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["epoch", "train_loss", "val_loss"])
    for epoch in range(num_epochs):
        writer.writerow([epoch + 1, avg_train_losses[epoch], avg_val_losses[epoch]])

print(f"Saved losses to {losses_csv_path}")

# Plot losses and save as PDF
plt.figure()
epochs = list(range(1, num_epochs + 1))
plt.plot(epochs, avg_train_losses, marker="o", label="Train Loss")
plt.plot(epochs, avg_val_losses, marker="o", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
losses_pdf_path = RESULTS_DIR / "finetuned_losses.pdf"
plt.savefig(losses_pdf_path, format="pdf", bbox_inches="tight")
print(f"Saved plot to {losses_pdf_path}")
