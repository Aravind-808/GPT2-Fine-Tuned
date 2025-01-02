from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import streamlit as st
import re

model = GPT2LMHeadModel.from_pretrained("./trained_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./trained_gpt2")

tokenizer.pad_token = "<PAD>" 
model.config.pad_token_id = tokenizer.pad_token_id 

tokenizer.add_special_tokens({'pad_token': '<PAD>'})

def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    outputs = model.generate(inputs["input_ids"], 
                             attention_mask=inputs["attention_mask"], 
                             do_sample = True, 
                             temperature = 0.2,
                             max_length=max_length, 
                             num_return_sequences=1, 
                             no_repeat_ngram_size=2, 
                             pad_token_id=tokenizer.pad_token_id
                             )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

def correct_name(answer):
    answer = answer.replace("Aravinda", "Aravind") if "Aravinda" in answer else answer.replace("Arav.", "Aravind") if "Arav." in answer else answer.replace("Arapah","Aravind")
    answer = answer.replace("3","5")
    return answer

st.title("Fine Tuned LLM with my info")

st.chat_message("assistant").markdown("Hello there!! I am Aravind (kind of). Ask me about myself and ill try my best to answer.")
if 'history' not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You: ", "")

if user_input:
    st.session_state.history.append({"role": "user", "message": user_input})

    answer = generate_text(user_input, max_length=150)

    answer = answer.split("?")[-1].strip()
    answer = answer.replace(user_input, "")
    answer = re.sub(r'[^a-zA-Z0-9\s,-.!()]', '', answer)
    answer = correct_name(answer)

    st.session_state.history.append({"role": "bot", "message": answer})

for message in st.session_state.history:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["message"])
    else:
        st.chat_message("assistant").markdown(message["message"])