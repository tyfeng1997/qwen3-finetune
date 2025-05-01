from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# 加载PEFT模型
adapter_model = PeftModel.from_pretrained(
    base_model,
    "runs/Qwen3-0.6B-math-orca-qlora-10k-ep1",  # 你的模型输出路径
    torch_dtype=torch.bfloat16
)

# 合并模型
merged_model = adapter_model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained("Qwen3-0.6B-math-merged")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
tokenizer.save_pretrained("Qwen3-0.6B-math-merged")