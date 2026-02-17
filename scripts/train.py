import os
import time
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
from transformers.trainer_utils import get_last_checkpoint
import evaluate
from dotenv import load_dotenv


# Environment configs
load_dotenv()
os.environ["WANDB_DISABLED"] = "true"
os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs"
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
if not os.path.exists("logs"):
    os.makedirs("logs/")

# Directory and file paths
train_text_file = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\train.txt"
test_text_file  = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\test.txt"
val_text_file   = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\val.txt"
root_dir        = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\HindiSeg"

def dataset_generator(data_path):
    with open(data_path, encoding="utf-8") as f:
        dataset = f.readlines()

    dataset_list = []
    for i in range(len(dataset)):

        image_id = dataset[i].split("\n")[0].split(' ')[0].strip()

        text = dataset[i].split("\n")[0].split(' ')[1].strip()
        row = [image_id, text]
        dataset_list.append(row)

    dataset_df = pd.DataFrame(dataset_list, columns=['file_name', 'text'])

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
        file_name = self.df['file_name'][idx]
        text = str(self.df.loc[idx, "text"]).strip()

        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        labels = self.processor.tokenizer(text, padding="max_length", max_length=self.max_target_length).input_ids

        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
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

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encode, decode, tie_word_embeddings = False)

bos_id = processor.tokenizer.cls_token_id
eos_id = processor.tokenizer.sep_token_id
pad_id = processor.tokenizer.pad_token_id

model.config.decoder_start_token_id = bos_id
model.config.eos_token_id = eos_id
model.config.pad_token_id = pad_id
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
generation_config = GenerationConfig(
    bos_token_id = bos_id,
    pad_token_id = pad_id,
    eos_token_id = eos_id,
    decoder_start_token_id = bos_id,

    max_length = 64,
    early_stopping = True,
    no_repeat_ngram_size = 3,
    length_penalty = 2.0,
    num_beams = 4,
)

model.generation_config = generation_config


training_args = Seq2SeqTrainingArguments(
    num_train_epochs = 20,
    predict_with_generate = True,
    eval_strategy =  "steps",
    output_dir = "./checkpoints/",

    per_device_train_batch_size = 8, #16 for 5070Ti
    fp16 = torch.cuda.is_available(), 
    weight_decay = 0.01,
    # gradient_accumulation_steps = 1, 
    # gradient_checkpointing = True

    learning_rate = 5e-4,
    optim = "adamw_torch_fused",

    per_device_eval_batch_size = 8, #16 for 5070Ti
    logging_steps = 50,
    save_steps = 1000,
    eval_steps = 1000,
    report_to = ['tensorboard'],
    load_best_model_at_end = True
)

# LOGGING STATS
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

grad_accum = training_args.gradient_accumulation_steps if training_args.gradient_accumulation_steps else 1
eff_batch_size = training_args.per_device_train_batch_size * grad_accum 
total_steps = (len(train_dataset) // eff_batch_size) * training_args.num_train_epochs
estimated_seconds = total_steps * 0.5
steps_per_epoch = len(train_dataset) // eff_batch_size
estimate_hours = estimated_seconds / 3600 

encoder_params = sum(p.numel() for p in model.encoder.parameters())
decoder_params = sum(p.numel() for p in model.decoder.parameters())
encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
decoder_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)

frozen_params = total_params - trainable_params
enc_dec_ratio = encoder_params / decoder_params
sample_lengths = [len(processor.tokenizer(str(t)).input_ids) for t in train_df['text'].sample(200)]

print(f"{'='*30}")
if torch.cuda.is_available(): 
    print("GPU STATS:")
    print("Current device: ", torch.cuda.current_device())
    print("Cuda device count: ", torch.cuda.device_count())
    print("Device Name: ", torch.cuda.get_device_name())

print(f"{'='*30}")
print(f"DATASET STATISTICS: ")
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))

print(f"{'='*30}")
print(f"MODEL STATISTICS:")
print(f"Total Parameters: {total_params:,}")
print(f"Estimated Training Time: {estimate_hours: .1f} hours")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Effective Batch Size: {eff_batch_size}")
print(f"Total Training Steps: {total_steps}")
print("Steps per epoch: ", steps_per_epoch)

print(f"{'='*30}")
print("DATASET TOKEN COVERAGE: ")
print(f"Avg Token Length (sample): {np.mean(sample_lengths):.1f}")
print(f"Max Token Length (sample): {np.max(sample_lengths)}")
print(f"Sequences exceeding max_length (128): {sum(l > 128 for l in sample_lengths)}")

print(f"{'='*30}")
print("\nPARAMETER BREAKDOWN:")
print(f"Encoder Total Params: {encoder_params:,}")
print(f"Decoder Total Params: {decoder_params:,}")
print(f"Encoder Trainable: {encoder_trainable:,}")
print(f"Decoder Trainable: {decoder_trainable:,}")

print(f"{'='*30}")
print("CROSS-ARCHITECTURE PARAMTER RATIO")
print(f"Encoder/Decoder Param Ratio: {enc_dec_ratio:.2f}x")
print(f"Frozen Parameters: {frozen_params:,}")
print(f"Trainable %: {100 * trainable_params / total_params:.1f}%")
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

last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)

trainer.train(resume_from_checkpoint = last_checkpoint)

os.makedirs("model/", exist_ok = True)
model.save_pretrained("model/")