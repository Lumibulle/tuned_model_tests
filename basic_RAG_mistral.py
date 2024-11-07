from mistralai import Mistral
import requests
import numpy as np
import faiss
import os
from getpass import getpass
import time

#api_key : UwnFADpfdr8VmFURB27OeMcYopXaOIkL

client = Mistral(api_key="UwnFADpfdr8VmFURB27OeMcYopXaOIkL")

response = requests.get('https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt')
text = response.text

chunk_size = 2048
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

print("chunks : ", len(chunks))

print("longeure de texte : ", len(text))

def get_text_embedding(input):
  embeddings_batch_response = client.embeddings.create(
    model="mistral-embed",
    inputs=input
  )
  return embeddings_batch_response.data[0].embedding

text_embeddings = []
for chunk in chunks :
  text_embeddings.append(get_text_embedding(chunk))
  time.sleep(2)
  print(".")
text_embeddings = np.array(text_embeddings)

d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)

question = "What were the two main things the Paul Graham worked on before college?"
question_embeddings = np.array([get_text_embedding(question)])
time.sleep(2)
D, I = index.search(question_embeddings, k=2)
print(I)

retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
print(retrieved_chunk)

prompt = f"""
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and prior knowledge, answer the query.
Query: {question}
Answer:
"""

def run_mistral(user_message, model="mistral-large-latest"):
  messages = [
    {
      "role": "user", "content": user_message
    }
  ]
  chat_response = client.chat.complete(
    model=model,
    messages=messages
  )
  return (chat_response.choices[0].message.content)

print(run_mistral(prompt))
