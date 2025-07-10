import torch 

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LoRA configuration
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj"]

# Generation configuration
MAX_LENGTH = 512
TEMPERATURE = 0.7
TOP_P = 0.9


# Random seed for reproducibility 
SEED = 42  # For reproducibility

# DPO training configuration
GROUNDING_ENABLED_STAGE_1 = True  # Enable grounding for DPO training stage 1

RL_ALGO = "ppo"
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
NUM_EPOCHS = 5
MAX_NEW_TOKENS = 512
TURN_LIMIT = 4
REDUNDANT_TOOL_PENALTY = 0.2