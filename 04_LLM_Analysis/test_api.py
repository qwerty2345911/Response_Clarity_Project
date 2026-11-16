import os
from openai import OpenAI

# 注意：改用 DeepSeek 的 base_url 和密钥
client = OpenAI(
    api_key="sk-07e5c38567bd41cb9e4cc4b8ab2ce066",  # 或者用 os.environ.get("DEEPSEEK_API_KEY")
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-flash",  # 改用 DeepSeek 的模型名
    messages=[
        {"role": "user", "content": "你好，请用一句话介绍你自己。"}
    ]
)
print(response.choices[0].message.content)