# Qwen3-0.6B Math Fine-tuning

This repository contains code for fine-tuning the Qwen3-0.6B model on mathematical reasoning tasks using QLoRA. The fine-tuned model shows significant improvement in mathematical reasoning capabilities, achieving 43.06% accuracy on GSM8K (compared to 20.17% for the base model).

## Setup and Installation

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/tyfeng1997/qwen3-finetune.git
cd qwen3-finetune
pip install -r requirements.txt
```

2. Log in to Hugging Face and Weights & Biases:

```bash
huggingface-cli login
wandb login
```

## Data Preparation

The training script expects data in a specific format with system, user, and assistant messages. Convert your dataset using the provided script:

```bash
python scripts/convert_dataset.py
```

The script transforms data to the following format:

```json
{
  "messages": [
    {"role": "system", "content": "system_message"},
    {"role": "user", "content": "sample['question']"},
    {"role": "assistant", "content": "sample['answer']"}
  ]
}
```

## Training

To fine-tune the model using QLoRA, run:

```bash
python scripts/sft.py --config recipes/Qwen3-0.6B-qlora.yaml
```

The YAML configuration file contains all necessary hyperparameters for the training process. The training progress can be monitored through Weights & Biases.

## Merging and Exporting the Model

### Merge and Push to Hugging Face

After training is complete, merge the adapter weights with the base model and push to Hugging Face:

```bash
python scripts/merge_adapter_weights.py \
  --peft_model_id {huggingface_name}/Qwen3-0.6B-math-orca-qlora-10k-ep1 \
  --output_dir merged_weights \
  --save_tokenizer True \
  --push_to_hub True
```

### Save Merged Model Locally

To save the merged model locally:

```bash
python scripts/merge_local.py
```

This will create a local copy of the fully merged model that can be used for inference.

## Serving the Model

You can serve the model using VLLM for efficient inference:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen3-0.6B-math-merged \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --served-model-name qwen3-0.6B-math \
  --trust-remote-code
```

This will start an OpenAI-compatible API server on port 8000.

## Evaluation

Evaluate the model's performance on mathematical reasoning tasks using the lm-evaluation-harness:

```bash
lm_eval --model local-chat-completions \
  --tasks gsm8k_cot \
  --model_args model=qwen3-0.6B-math,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=8,max_retries=3,tokenized_requests=False \
  --apply_chat_template \
  --fewshot_as_multiturn
```

## Performance

Our fine-tuned Qwen3-0.6B model achieves:

| Model | GSM8K Accuracy | Improvement |
|-------|----------------|-------------|
| Base Qwen3-0.6B | 20.17% | - |
| Fine-tuned Qwen3-0.6B | 43.06% | +113% |

This represents a substantial improvement in mathematical reasoning capabilities while maintaining the small model size.

## Client Usage Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"
)

response = client.chat.completions.create(
    model="qwen3-0.6B-math",
    messages=[
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": "If 8x + 5 = 3x - 15, what is the value of x?"}
    ],
    temperature=0.2,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

## License

This project is released under [LICENSE NAME] license. The base Qwen3-0.6B model is subject to Alibaba's license terms.

## Acknowledgements

- Thanks to the creators of the Qwen3 model
- Hugging Face for the transformers library
- VLLM team for the inference server
- EleutherAI for the evaluation harness