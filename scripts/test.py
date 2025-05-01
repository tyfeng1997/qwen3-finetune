from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # VLLM不验证API密钥，但需要提供一个值
)

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

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"},
]
expected_answer = "72"

try:
    response = client.chat.completions.create(
        model="qwen3-0.6B-math",  
        messages=messages,
        stream=False,
        max_tokens=1024,
        temperature=0.2,  
    )
    
    response_content = response.choices[0].message.content
    
    # 打印结果
    print(f"\n===== query =====\n{messages[1]['content']}")
    print(f"\n===== true =====\n{expected_answer}")
    print(f"\n===== answer =====\n{response_content}")
  
    
except Exception as e:
    print(f"got error: {str(e)}")
