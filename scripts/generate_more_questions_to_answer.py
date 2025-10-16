import pandas as pd
from openai import OpenAI
import json
import time

df = pd.read_csv("./data.csv", sep=";")
with open("added_data.json", 'r', encoding="utf-8") as file:
    loaded_data = json.load(file)

client = OpenAI(
    api_key="AIzaSyBRCSqzf5osQEeC2cS9vyEZKxEaJQAkg34",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


system_prompt = "Ты являешься новым клиентом банка. Тебе дано утверждение. Придумай 3-7 вопросов, на которые отвечает данное утверждение. За каждый корректный и уникальный вопрос ты получишь 100$. Вопрос не является корректным если в утверждении нет однозначного ответа, или вопрос слишком конкретный, или если вопрос слишком неестественный. В случае неккорекного вопроса с твоего счета будет вычтено 1000$.В ответе дожны быть ТОЛЬКО ВОПРОСЫ, разделенные переходом на новую линию, ничего более. Вопрос должен звучать естественно, как обычно реально задают вопрос клиенты, со всеми неточностями. Вопрос не должен быть слишком конкретный, потому что реально так не обащются."
for index, value in df.iterrows():
    if value["answer_pattern"] in loaded_data:
        continue
    print(value["answer_pattern"])

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": f"{system_prompt}"},
            {
                "role": "user",
                "content": f"Категория утверждения - {value["category"]}, подкатегория - {value['subcategory']}, утверждение - {value['answer_pattern']}, пример вопроса к утверждению - {value['text']}",
            }
        ]
    )

    print(response.choices[0].message)
    loaded_data[value["answer_pattern"]] = response.choices[0].message.content.split("\n")

    with open("added_data.json", 'w', encoding="utf-8") as file:
        json.dump(loaded_data, file, indent = 4, ensure_ascii=False)
    
    time.sleep(1)