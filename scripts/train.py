import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import RobertaTokenizer, TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import ViTImageProcessor
from transformers import GenerationConfig
from transformers import default_data_collator
# from datasets import load_metric
import evaluate
os.environ["WANDB_DISABLED"] = "true"
os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs"
os.makedirs("logs/")
# torch.cuda.empty_cache()

# directory and file paths
train_text_file = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\train.txt"
test_text_file  = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\test.txt"
val_text_file   = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\val.txt"
root_dir        = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\HindiSeg"

def dataset_generator(data_path):
    with open(data_path, encoding="utf-8") as f:
        dataset = f.readlines()
    # counter = 0

    dataset_list = []
    for i in range(len(dataset)):
        # if counter > 30000:
        #     break
        image_id = dataset[i].split("\n")[0].split(' ')[0].strip()
        # vocab_id = int(dataset[i].split(",")[1].strip())
        text = dataset[i].split("\n")[0].split(' ')[1].strip()
        row = [image_id, text]
        dataset_list.append(row)
        # counter += 1

    dataset_df = pd.DataFrame(dataset_list, columns=['file_name', 'text'])
    # dataset_df.head()
    return dataset_df
    
train_df = dataset_generator(train_text_file)
test_df = dataset_generator(test_text_file)
val_df = dataset_generator(val_text_file)

print(f"Train, Test & Val shape: {train_df.shape, test_df.shape, val_df.shape}")

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = str(self.df.loc[idx, "text"]).strip()
        # prepare image (i.e. resize + normalize)
        # image = Image.open(self.root_dir / file_name).convert("RGB")
        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB");
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        # print(encoding)
        return encoding


encode = 'google/vit-base-patch16-224-in21k'
decode = 'flax-community/roberta-hindi'

feature_extractor= ViTImageProcessor.from_pretrained(encode)
tokenizer = RobertaTokenizer.from_pretrained(decode)
processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

train_dataset = IAMDataset(root_dir=root_dir,
                           df=train_df,
                           processor=processor)
eval_dataset = IAMDataset(root_dir=root_dir,
                           df=test_df,
                           processor=processor)

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encode, decode)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
print(f"processor.tokenizer.pad_token_id: {processor.tokenizer.pad_token_id}")
model.config.vocab_size = model.config.decoder.vocab_size
# config_decoder.is_decoder = True
# config_decoder.add_cross_attention = True

# set beam search parameters
generation_config = GenerationConfig(
    bos_token_id=processor.tokenizer.cls_token_id,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.sep_token_id,
    max_length=64,
    early_stopping=True,
    no_repeat_ngram_size=3,
    length_penalty=2.0,
    num_beams=4
)

model.generation_config = generation_config


training_args = Seq2SeqTrainingArguments(
    num_train_epochs=50,
    predict_with_generate=True,
    eval_strategy="steps",
    # per_device_train_batch_size=2,
    # per_device_eval_batch_size=4,
    output_dir="./checkpoints/",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    # output_dir="./",
    logging_steps=2,
    save_steps=2000,
    eval_steps=100,
    report_to= ['tensorboard']
)

# LOGGIN STATS
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

grad_accum = training_args.gradient_accumulation_steps if training_args.gradient_accumulation_steps else 1
eff_batch_size = training_args.per_device_train_batch_size * grad_accum 

total_steps = (len(train_dataset) // eff_batch_size) * training_args.num_train_epochs

print(f"\n{'='*30}")
print(f"DATASET STATISTICS: ")
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))
print(f"\n{'='*30}")
print(f"MODEL STATISTICS:")
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Effective Batch Size: {eff_batch_size}")
print(f"Total Training Steps: {total_steps}")
print(f"{'='*30}\n")
# END LOGGIN STATS

cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

trainer = Seq2SeqTrainer(
    model=model,
    processing_class=processor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)

trainer.train()

os.makedirs("model/")
model.save_pretrained("model/")