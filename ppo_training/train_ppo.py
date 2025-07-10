from agent import CentralLLM
from tools import TOOLS
from reward import heuristic_reward
from config import NUM_EPOCHS, LEARNING_RATE, MODEL_NAME
from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

HYPOTHESES = [
    "If we increase the temperature of an ideal gas while keeping volume constant, the pressure increases linearly.",
    "Gravitational waves move faster than the speed of light in strong magnetic fields."
]

def run_rollout(agent, hypothesis):
    messages = [
        {"role": "system", "content": "You are a physics assistant that evaluates scientific hypotheses using reasoning and tools."},
        {"role": "user", "content": f"Evaluate this hypothesis: '{hypothesis}'"}
    ]
    messages = agent.run_dialogue(messages)
    response = messages[-1]["content"]
    reward = heuristic_reward(messages)
    return messages, response, reward

def train():
    agent = CentralLLM(TOOLS)
    ppo_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ppo_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")

    config = PPOConfig(
        model_name=MODEL_NAME,
        learning_rate=LEARNING_RATE,
        batch_size=1,
        mini_batch_size=1,
    )
    trainer = PPOTrainer(config, ppo_model, ppo_tokenizer)

    for epoch in range(NUM_EPOCHS):
        for hyp in HYPOTHESES:
            messages, response, reward = run_rollout(agent, hyp)
            query = ppo_tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)["input_ids"]
            response_ids = ppo_tokenizer.encode(response, return_tensors="pt").squeeze().tolist()
            trainer.step([query], [response_ids], [reward])
            print("Epoch", epoch, "Hypothesis:", hyp)
            print("Response:", response)
            print("Reward:", reward)

if __name__ == "__main__":
    train()
