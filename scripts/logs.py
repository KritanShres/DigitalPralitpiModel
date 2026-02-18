import torch
import numpy as np
import time

def log_training_stats(
    model,
    training_args,
    train_dataset,
    eval_dataset,
    processor=None,
    train_df=None,
    text_column="text",
    max_length=128,
    seconds_per_step_estimate=0.5,
    token_sample_size=200
):

    print("=" * 40)
    print("SYSTEM / GPU INFORMATION")
    print("=" * 40)

    if torch.cuda.is_available():
        print("Current device:      ", torch.cuda.current_device())
        print("CUDA device count:   ", torch.cuda.device_count())
        print("Device name:         ", torch.cuda.get_device_name())
    else:
        print("CUDA not available")

    print("\n" + "=" * 40)
    print("DATASET STATISTICS")
    print("=" * 40)

    print(f"Training examples:   {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")

    print("\n" + "=" * 40)
    print("MODEL PARAMETER STATISTICS")
    print("=" * 40)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    decoder_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)

    enc_dec_ratio = encoder_params / decoder_params if decoder_params != 0 else float("inf")

    grad_accum = training_args.gradient_accumulation_steps or 1
    eff_batch_size = training_args.per_device_train_batch_size * grad_accum
    steps_per_epoch = len(train_dataset) // eff_batch_size
    total_steps = steps_per_epoch * training_args.num_train_epochs
    estimated_seconds = total_steps * seconds_per_step_estimate
    estimate_hours = estimated_seconds / 3600

    print(f"Total parameters:        {total_params:,}")
    print(f"Trainable parameters:    {trainable_params:,}")
    print(f"Frozen parameters:       {frozen_params:,}")
    print(f"Trainable percentage:    {100 * trainable_params / total_params:.2f}%")
    print(f"Encoder total params:    {encoder_params:,}")
    print(f"Decoder total params:    {decoder_params:,}")
    print(f"Encoder trainable:       {encoder_trainable:,}")
    print(f"Decoder trainable:       {decoder_trainable:,}")
    print(f"Encoder/Decoder ratio:   {enc_dec_ratio:.2f}x")

    print("\n" + "=" * 40)
    print("TRAINING DYNAMICS")
    print("=" * 40)

    print(f"Effective batch size:    {eff_batch_size}")
    print(f"Steps per epoch:         {steps_per_epoch}")
    print(f"Total training steps:    {total_steps}")
    print(f"Estimated training time: {estimate_hours:.2f} hours "
          f"(assuming {seconds_per_step_estimate}s per step)")

    if processor is not None and train_df is not None and text_column in train_df.columns:
        print("\n" + "=" * 40)
        print("TOKEN LENGTH STATISTICS (Sample-Based)")
        print("=" * 40)

        sample_texts = train_df[text_column].sample(
            min(token_sample_size, len(train_df))
        )

        sample_lengths = [
            len(processor.tokenizer(str(t)).input_ids)
            for t in sample_texts
        ]

        print(f"Sample size:                   {len(sample_lengths)}")
        print(f"Average token length:          {np.mean(sample_lengths):.2f}")
        print(f"Maximum token length:          {np.max(sample_lengths)}")
        print(f"Sequences > max_length({max_length}): "
              f"{sum(l > max_length for l in sample_lengths)}")

    print("\n" + "=" * 40)
    print("END OF LOG")
    print("=" * 40)
