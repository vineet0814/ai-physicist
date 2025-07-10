from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import config

def load_model_and_tokenizer(model_name=config.MODEL_NAME, device=config.DEVICE):
    """
    Load the model and tokenizer.

    Args:
        model_name (str): The name of the model to load from Hugging Face.
        device (str): The device to load the model onto ("cuda" or "cpu").

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    return model, tokenizer

def apply_lora(model, lora_r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"]):
    """
    Apply LoRA to the model.

    Args:
        model: The base model to which LoRA will be applied.
        lora_r (int): The rank of the LoRA matrices.
        lora_alpha (int): The scaling factor for LoRA.
        lora_dropout (float): Dropout probability for LoRA layers.
        target_modules (list): List of module names to apply LoRA to.

    Returns:
        model: The model with LoRA applied.
    """
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print("LoRA applied successfully.")
    return model

def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """
    Generate a response from the model given a prompt using chat template.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        prompt (str): The input prompt for the model.
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        str: The generated response.
    """
    # Define the chat messages
    messages = [
        {"role": "system", "content": "You are a physics researcher."},
        {"role": "user", "content": prompt}
    ]

    # Apply the chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Prepare model inputs
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )

    # Remove input tokens from the output
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response