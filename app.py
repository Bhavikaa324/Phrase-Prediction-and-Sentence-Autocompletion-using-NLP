from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = load_model("text_generation_model.keras")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Define the maximum sequence length (same as used in training)
max_sequence_len = 100

def sample_with_temperature(predictions, top_k=15, temperature=0.7):
    """Selects a word from the top-k predictions using temperature scaling for diversity."""
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions + 1e-8) / temperature  # Apply temperature scaling
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)  # Normalize probabilities

    top_indices = np.argsort(predictions)[-top_k:]  # Get the top K words
    chosen_index = np.random.choice(top_indices, p=predictions[top_indices] / np.sum(predictions[top_indices]))

    return chosen_index #return word

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    seed_text = request.form.get('prompt')
    next_words = int(request.form.get('next_words', 10))  # Default: 10 words
    num_results = int(request.form.get('num_results', 3))  # Default: 3 results
    top_k = 15  # Increased for better word selection
    temperature = 0.7  # Balances randomness and structure

    results = []

    for _ in range(num_results): #for multiple results
        current_text = seed_text
        for _ in range(next_words):  # for number of words
            token_list = tokenizer.texts_to_sequences([current_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

            predictions = model.predict(token_list, verbose=0)[0]  #predict

            # Use sampling instead of always picking the highest probability word
            chosen_index = sample_with_temperature(predictions, top_k=top_k, temperature=temperature)

            output_word = tokenizer.index_word.get(chosen_index, None)  #token to word

            if output_word is None:
                output_word = tokenizer.index_word.get(np.argmax(predictions), "")  # Use most probable word

            while output_word in ["", ".", ",", "?", "!"]:  # Avoid empty/punctuation words
                chosen_index = np.argmax(predictions)
                output_word = tokenizer.index_word.get(chosen_index, "")

            current_text += " " + output_word #add word to sentence

        results.append(current_text)

    return jsonify({"generated_texts": results})

if __name__ == '__main__':
    app.run(debug=True)
