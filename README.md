# Fine-Tuned GPT-2 Personal Chatbot

This project fine-tunes a GPT-2 model to answer questions about me. It uses the `transformers` library from Hugging Face, fine-tunes the model on the custom dataset given in the repo, and deploys it using Streamlit.

## Features
- Fine-tuned GPT-2 model trained on personalized Q&A data.
- Custom preprocessing for handling questions and answers.
- Streamlit-based chatbot UI for easy interaction.

## Installation
Ensure you have Python installed, then install the required dependencies:

```sh
pip install torch transformers datasets streamlit
```

## Dataset Format
The dataset is in the JSONL format (`dataset.jsonl`), where each line is a JSON object with `question` and `answer` fields:

```json
{"question": "What is your name?", "answer": "My name is Aravind."}
```

## Training the Model
Run the following script to train the GPT-2 model:

```python
python train_model.py
```

This script:
- Loads and tokenizes the dataset.
- Fine-tunes GPT-2 using the Hugging Face `Trainer`.
- Saves the trained model and tokenizer to `./trained_gpt2`.
> Note: You can change the training arguments as you wish. For example: `save_total_limit = 2` will save only the last 2 checkpoints, saving you space

## Running the Chatbot
To start the chatbot, run:

```sh
streamlit run question_model.py
```

This will launch a web-based chat interface where users can ask questions.

I made this to learn the basics of LLMs, tokenization and fine tuning. Hence, the model might not be accurate.
