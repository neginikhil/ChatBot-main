from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load tokenizer and model once at the start
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Store chat history and reset it every few turns
chat_history_ids = None
turn_count = 0  # Count the number of interactions

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    global chat_history_ids, turn_count

    user_input = request.form["msg"]

    # Get bot response
    response = get_chat_response(user_input)

    # Return JSON response
    return jsonify({"response": response})

def get_chat_response(user_input):
    global chat_history_ids, turn_count

    # Encode the new user input and add the EOS token
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # If chat history exists, append the new user input to it; otherwise, start a new conversation
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    # Generate a response while limiting the total chat history to 1000 tokens
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True
    )

    # Decode the bot's response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Increment turn count
    turn_count += 1

    # Reset chat history after every 5 turns to avoid repetition
    if turn_count >= 5:
        chat_history_ids = None
        turn_count = 0

    return response

if __name__ == '__main__':
    app.run(debug=True)
