# AI Physicist: Supervised Fine-Tuning + Dual-Stage RL with Tool Calling for Evaluation

This repository implements a three-phase training framework to build a specialized large language model for scientific reasoning in physics. The system consists of:

1. **Phase 1:** Supervised Fine-Tuning (SFT) on physics data to domain-adapt the base model.
2. **Phase 2:** Hypothesis Generation via Direct Preference Optimization (DPO).
3. **Phase 3:** Evaluation Strategy Formulation via PPO with tool-assisted preference supervision.

---

## Directory Structure

```
ai_physicist_dpo/
├── data/
│   ├── sft_physics_data.json                # Instruction-format data for supervised 
│   ├── stage1_hypothesis_pairs.json         # Dataset of hypothesis preference pairs
│   ├── stage2_evals.json    # Evaluation strategy pairs + tool outputs
│
├── tools/
│   ├── tool_manager.py
│
├── sft_training/
│   ├── train_sft_physics.py                 # Script for supervised fine-tuning
│
├── dpo_training/
│   ├── train_stage1_hypothesis_dpo.py       # DPO trainer for hypothesis generation
│   ├── model_utils.py                       # Utilities for loading model/tokenizer
│   ├── dataset_utils.py                     # Loaders and formatting for DPO datasets
├── ppo_training/
│   ├── train_ppo.py       # PPO trainer for evaluation strategy synthesis
│   ├── agent.py                       # Utilities for loading CentralLLM/tokenizer
│   ├── reward.py                       # Reward model
│
├── readme.md                                # Documentation and setup
├── requirements.txt                         # Python dependencies
└── .gitignore
```

---

## 🚀 Training Phases

### Phase 1: Supervised Fine-Tuning (SFT)

- Input: Instruction-tuning dataset of physics tasks (QA, problem-solving, derivation)
- Goal: Align model toward physics reasoning using `train_sft_physics.py`
- Example data entry:
```json
{
  "instruction": "Explain the difference between general relativity and special relativity.",
  "output": "General relativity accounts for gravity as curvature of spacetime, whereas special relativity does not include gravitational effects."
}
```

```bash
python sft_training/train_sft_physics.py \
  --model_name Qwen \
  --dataset_path data/sft_physics_data.json \
  --output_dir best_sft_model
```

---

### Phase 2: Hypothesis Generation via DPO



---

### Phase 3: Evaluation Strategy via Tool-Assisted PPO


---