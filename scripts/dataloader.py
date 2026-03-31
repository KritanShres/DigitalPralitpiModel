# data_utils.py
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'][idx]
        text = str(self.df.loc[idx, "text"]).strip()
        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(text, padding="max_length", max_length=self.max_target_length).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

def dataset_generator(data_path):
    with open(data_path, encoding="utf-8") as f:
        dataset = f.readlines()
    dataset_list = []
    for i in range(len(dataset)):
        line = dataset[i].strip()
        if not line:
            continue
        parts = line.split(' ', 1)
        image_id = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ""

        # Normalize path separators
        image_id = image_id.replace("/", os.sep).replace("\\", os.sep)

        # Replace .jpg extension with .png
        image_id = os.path.splitext(image_id)[0] + ".png"

        dataset_list.append([image_id, text])

    return pd.DataFrame(dataset_list, columns=['file_name', 'text'])