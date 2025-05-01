from datasets import load_dataset
 
# Create system prompt
system_message = """Solve the given high school math problem by providing a clear explanation of each step leading to the final solution.
 
Provide a detailed breakdown of your calculations, beginning with an explanation of the problem and describing how you derive each formula, value, or conclusion. Use logical steps that build upon one another, to arrive at the final answer in a systematic manner.
 
# Steps
 
1. **Understand the Problem**: Restate the given math problem and clearly identify the main question and any important given values.
2. **Set Up**: Identify the key formulas or concepts that could help solve the problem (e.g., algebraic manipulation, geometry formulas, trigonometric identities).
3. **Solve Step-by-Step**: Iteratively progress through each step of the math problem, justifying why each consecutive operation brings you closer to the solution.
4. **Double Check**: If applicable, double check the work for accuracy and sense, and mention potential alternative approaches if any.
5. **Final Answer**: Provide the numerical or algebraic solution clearly, accompanied by appropriate units if relevant.
 
# Notes
 
- Always clearly define any variable or term used.
- Wherever applicable, include unit conversions or context to explain why each formula or step has been chosen.
- Assume the level of mathematics is suitable for high school, and avoid overly advanced math techniques unless they are common at that level.
"""
 
# convert to messages 
def create_conversation(sample):
  return {
    "messages": [
      {"role": "system", "content": system_message},
      {"role": "user", "content": sample["question"]},
      {"role": "assistant", "content": sample["answer"]}
    ]
  }  
 
# Load dataset from the hub
dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
 
# Convert dataset to OAI messages
dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)
 
print(dataset[345]["messages"])
 
# save datasets to disk 
dataset.to_json("train_dataset.json", orient="records")