import os
import re
import json
import torch
import evaluate
import numpy as np
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from torch.utils.data import Dataset

from transformers import (
    RobertaTokenizer,
    TrOCRProcessor,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    GenerationConfig,
    default_data_collator
)
from transformers.trainer_utils import get_last_checkpoint, PREFIX_CHECKPOINT_DIR

from logs import log_training_stats
from dataloader import IAMDataset, dataset_generator

# ===========================
# ENVIRONMENT CONFIGURATION
# ===========================
load_dotenv()
os.environ["WANDB_DISABLED"] = "true"
os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs"
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
if not os.path.exists("logs"):
    os.makedirs("logs/")

# ===========================
# DATA FILE PATHS
# ===========================
train_text_file = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\train.txt"
test_text_file  = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\test.txt"
val_text_file   = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\val.txt"
root_dir        = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\HindiSeg"

# ===========================
# LOAD DATAFRAMES
# ===========================
train_df = dataset_generator(train_text_file)
test_df  = dataset_generator(test_text_file)
val_df   = dataset_generator(val_text_file)

print(f"Train, Test & Val shape: {train_df.shape, test_df.shape, val_df.shape}")

# ===========================
# PREPROCESSORS & TOKENIZERS
# ===========================
encode = 'google/vit-base-patch16-224-in21k'
decode = 'flax-community/roberta-hindi'

feature_extractor = ViTImageProcessor.from_pretrained(encode)
tokenizer         = RobertaTokenizer.from_pretrained(decode)
processor         = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

# ===========================
# CREATE DATASETS
# ===========================
train_dataset = IAMDataset(root_dir=root_dir, df=train_df, processor=processor)
eval_dataset  = IAMDataset(root_dir=root_dir, df=test_df, processor=processor)

# ===========================
# TOKEN IDS
# ===========================
bos_id = processor.tokenizer.cls_token_id
eos_id = processor.tokenizer.sep_token_id
pad_id = processor.tokenizer.pad_token_id

# ===========================
# TRAINING ARGUMENTS
# ===========================
training_args = Seq2SeqTrainingArguments(
    eval_strategy =  "steps",
    output_dir = "./checkpoints/",

    per_device_train_batch_size = 8, #16 
    per_device_eval_batch_size = 16, #32 
    fp16 = torch.cuda.is_available(),
    weight_decay = 0.01,
    gradient_accumulation_steps = 2,
    gradient_checkpointing = True,

    num_train_epochs = 20,
    predict_with_generate = True,
    learning_rate = 5e-4,
    optim = "adamw_torch_fused",

    logging_steps = 50,
    save_steps = 1000, # 2000
    eval_steps = 1000, # 2000
    report_to = ['tensorboard'],
    load_best_model_at_end = False,
    save_total_limit = 3,
)

# ===========================
# METRICS
# ===========================
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

# ===========================
# CHECKPOINT MANAGEMENT
# ===========================
def get_last_complete_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None

    pattern = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    checkpoints = [
        p for p in os.listdir(output_dir)
        if pattern.search(p) and os.path.isdir(os.path.join(output_dir, p))
    ]
    if not checkpoints:
        return None

    # Sort descending by step number
    checkpoints = sorted(checkpoints, key=lambda x: int(pattern.search(x).groups()[0]), reverse=True)

    required_files = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
    model_ok = lambda d: os.path.isfile(os.path.join(d, "model.safetensors")) or \
                         os.path.isfile(os.path.join(d, "pytorch_model.bin"))

    for name in checkpoints:
        folder = os.path.join(output_dir, name)
        if not model_ok(folder):
            continue
        if all(os.path.isfile(os.path.join(folder, f)) for f in required_files):
            return folder
    return None

# ===========================
# LOAD LAST CHECKPOINT IF AVAILABLE
# ===========================
last_checkpoint = None
best_checkpoint = None

if os.path.isdir(training_args.output_dir):
    last_checkpoint = get_last_complete_checkpoint(training_args.output_dir)
    if last_checkpoint is None:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    if last_checkpoint:
        trainer_state_path = os.path.join(last_checkpoint, "trainer_state.json")
        if os.path.isfile(trainer_state_path):
            with open(trainer_state_path, "r") as f:
                state = json.load(f)

            best_checkpoint = state.get('best_model_checkpoint', last_checkpoint)
        else:
            last_checkpoint = None

# ===========================
# MODEL INITIALIZATION
# ===========================
if last_checkpoint:
    print(f"Resuming from checkpoint: {last_checkpoint}")
    model = VisionEncoderDecoderModel.from_pretrained(last_checkpoint)
else:
    print("No checkpoint found, training from scratch")
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encode, decode, tie_word_embeddings=False)

# Token configuration
model.config.decoder_start_token_id = bos_id
model.config.eos_token_id = eos_id
model.config.pad_token_id = pad_id
model.config.vocab_size = model.config.decoder.vocab_size
model.decoder.config.tie_word_embeddings = False
model.config.use_cache = False
# ===========================
# GENERATION CONFIG
# ===========================
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
model.generation_config.use_cache = False

# ===========================
# LOGGING MODEL & DATA STATS
# ===========================
log_training_stats(
    model=model,
    training_args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processor=processor,
    train_df=train_df
)

# ===========================
# TRAINER INITIALIZATION
# ===========================
trainer = Seq2SeqTrainer(
    model=model,
    processing_class=processor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)

# ===========================
# TRAINING
# ===========================
trainer.train(resume_from_checkpoint=last_checkpoint)

# ===========================
# SAVE FINAL MODEL
# ===========================
os.makedirs("model/", exist_ok=True)
model.save_pretrained("model/")
