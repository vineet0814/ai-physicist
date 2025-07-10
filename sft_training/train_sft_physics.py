import os
import torch
from transformers import Trainer, TrainingArguments
from utils.dataset_utils import load_and_split_dataset
from utils.model_utils import load_model_and_tokenizer, apply_lora
from config import MODEL_NAME, DEVICE

def main():
    # Paths
    data_file = "data/sft_physics_data.json"
    output_dir = "sft_training/best_model"

    # Load tokenizer and model
    model, tokenizer = load_model_and_tokenizer(model_name=MODEL_NAME, device=DEVICE)

    # Load and split dataset
    train_dataset, test_dataset, eval_dataset = load_and_split_dataset(
        json_file=data_file,
        tokenizer=tokenizer,
        test_fraction=0.1,
        eval_fraction=0.1,
        max_length=512
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # Keep only the best model
        load_best_model_at_end=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="sft_training/logs",
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        report_to="none",  # Disable reporting to external services
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the best model
    print(f"Saving the best model to {output_dir}...")
    trainer.save_model(output_dir)

    print("Training complete.")

if __name__ == "__main__":
    main()