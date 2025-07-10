import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from tools.tool_manager import retrieve_arxiv_documents  # Import the arXiv retrieval utility
import config
from utils.model_utils import load_model_and_tokenizer

def load_hypothesis_preference_pairs_with_grounding(data_file, tokenizer, max_length=512, grounding_enabled=True):
    """
    Load hypothesis preference pairs from a JSON file, retrieve grounding documents, and tokenize them.

    Args:
        data_file (str): Path to the JSON file containing hypothesis preference pairs.
        tokenizer: The tokenizer to tokenize the data.
        max_length (int): Maximum token length for each input.
        grounding_enabled (bool): Whether to retrieve grounding documents.

    Returns:
        dataset: A tokenized dataset ready for training.
    """
    # Load the dataset
    dataset = load_dataset("json", data_files=data_file)["train"]

    # Tokenize the dataset
    def tokenize_function(example):
        input_text = example["input"]
        preferred_hypothesis = example["preferred_hypothesis"]
        dispreferred_hypothesis = example["dispreferred_hypothesis"]

        # Retrieve grounding documents if enabled
        grounding_text = ""
        if config.GROUNDING_ENABLED_STAGE_1:
            grounding_docs = retrieve_arxiv_documents(input_text, max_results=3)  # Retrieve top 3 documents
            grounding_text = "\n".join(grounding_docs)

        # Concatenate input with grounding and hypotheses
        input_with_grounding = f"{input_text}\nGrounding:\n{grounding_text}"
        preferred = f"{input_with_grounding}\nPreferred: {preferred_hypothesis}"
        dispreferred = f"{input_with_grounding}\nDispreferred: {dispreferred_hypothesis}"

        return {
            "preferred_input_ids": tokenizer(preferred, max_length=max_length, truncation=True, padding="max_length")["input_ids"],
            "dispreferred_input_ids": tokenizer(dispreferred, max_length=max_length, truncation=True, padding="max_length")["input_ids"]
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def main():
    # Paths
    data_file = "data/stage1_hypothesis_pairs.json"
    output_dir = "dpo_training/best_model"
    model_path = "sft_training/best_model"  # Path to the SFT model
    # Load tokenizer and model
    _, tokenizer = load_model_and_tokenizer(model_name=config.MODEL_NAME, device=config.DEVICE)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    print(f"Using model from {model_path}")

    # Apply LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Load and tokenize dataset with grounding
    tokenized_dataset = load_hypothesis_preference_pairs_with_grounding(data_file, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="dpo_training/logs",
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        report_to="none",
        fp16=torch.cuda.is_available(),
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    print("Starting DPO Stage 1 training with grounding...")
    trainer.train()

    # Save the best model
    print(f"Saving the best model to {output_dir}...")
    trainer.save_model(output_dir)

    print("DPO Stage 1 training complete.")

if __name__ == "__main__":
    main()