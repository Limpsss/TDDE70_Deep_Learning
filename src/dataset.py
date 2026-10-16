import json
import os
import random

from PIL import Image
from torch.utils.data import Dataset


class RSICD(Dataset):
    """A PyTorch Dataset class for the TDDE70 use of the RSICD dataset.

    Args:
        img_dir (str): Path to the directory containing all images in the dataset.
        txtclasses_dir (str): Path to the directory containing the text classes.
        json_path (str): Path to the JSON file describing each image, including captions and split information (train/val/test).
        preprocess (callable): A function to preprocess the images (e.g., from OpenCLIP).
        tokenizer (callable): A function to tokenize the text captions.
        split (str): The dataset split to use ('A' for self-supervised dataset, 'B' for labeled dataset, 'C' for labeled test dataset).
        caption_mode (str): How to select captions for samples with multiple captions. Options are 'first' (use the first caption) or 'random' (select a random caption).
    """

    def __init__(
        self,
        img_dir,
        txtclasses_dir,
        json_path,
        preprocess,
        tokenizer,
        split,
        caption_mode="first",
    ):
        self.img_dir = img_dir
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.split = split
        self.caption_mode = caption_mode

        split_map = {"A": "train", "B": "val", "C": "test"}
        if split not in split_map:
            raise ValueError("Invalid split. Must be 'A', 'B', or 'C'.")

        target_split = split_map[split]

        with open(json_path, "r") as f:
            data = json.load(f)
            samples = data["images"]
            self.samples = [s for s in samples if s["split"] == target_split]

        # Read class text files: each file name (without extension) is the class name
        # and contains image filenames (one per line) that belong to that class.
        class_map = {}
        if txtclasses_dir and os.path.isdir(txtclasses_dir):
            for fname in os.listdir(txtclasses_dir):
                path = os.path.join(txtclasses_dir, fname)
                if not os.path.isfile(path):
                    continue
                class_name, _ = os.path.splitext(fname)
                with open(path, "r") as cf:
                    for line in cf:
                        line = line.strip()
                        if not line:
                            continue
                        class_map[line] = class_name
                        base = os.path.basename(line)
                        name_no_ext, _ = os.path.splitext(base)
                        class_map[name_no_ext] = class_name

        for s in self.samples:
            fname = s.get("filename")
            s["target"] = class_map.get(fname) if fname is not None else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(f"{self.img_dir}/{sample['filename']}")
        caption = None

        if self.preprocess:
            image = self.preprocess(image)

        if self.split == "A":
            return image

        label = sample.get("target", "")

        if self.caption_mode == "first":
            caption = sample["sentences"][0]["raw"]

        if self.caption_mode == "random":
            caption = random.choice([c["raw"] for c in sample["sentences"]])

        if self.caption_mode == "label":
            caption = f"A satellite image of {label}"

        if self.caption_mode == "all":
            caption = [c["raw"] for c in sample["sentences"]]

        if self.tokenizer:
            caption = self.tokenizer(caption).squeeze()

        return image, caption, label
