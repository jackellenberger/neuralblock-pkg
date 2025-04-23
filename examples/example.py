import csv
import os

# Paths are relative to the location of this example script (pkg/examples)
# Read paths from environment variables with defaults
MODEL_PATH = os.environ.get("NEURALBLOCK_MODEL_PATH", "../data/nb_stream_fasttext_10k.h5")
TOKENIZER_PATH = os.environ.get("NEURALBLOCK_TOKENIZER_PATH", "../data/tokenizer_stream_10k.json")
TRANSCRIPT_PATH = os.environ.get("NEURALBLOCK_TRANSCRIPT_PATH", "episode_43f6d97d_segments.csv")

# Corrected imports
from neuralblock import NeuralBlockStream
from transcript_parser import parse_transcript

def load_transcript_from_csv(csv_path):
    """Loads transcript data from a CSV file."""
    transcript_list = []
    # Construct the absolute path to the CSV file
    abs_csv_path = os.path.join(os.path.dirname(__file__), csv_path)
    with open(abs_csv_path, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader) # Skip header row
        for row in reader:
            # Assuming format: start, duration, text
            if len(row) == 3:
                try:
                    start = float(row[0])
                    duration = float(row[1])
                    text = row[2]
                    transcript_list.append({'start': start, 'duration': duration, 'text': text})
                except ValueError:
                    print(f"Skipping invalid row: {row}")
            else:
                 print(f"Skipping row with unexpected format: {row}")
    return transcript_list

def main():
    # Load transcript data
    transcript_data = load_transcript_from_csv(TRANSCRIPT_PATH)
    if not transcript_data:
        print(f"Could not load transcript data from {TRANSCRIPT_PATH}. Please ensure the file exists and is correctly formatted.")
        # If using environment variables, print the value that was used
        env_var_value = os.environ.get("NEURALBLOCK_TRANSCRIPT_PATH")
        if env_var_value:
             print(f"The script attempted to use the path from NEURALBLOCK_TRANSCRIPT_PATH: {env_var_value}")
        else:
             print(f"The script used the default path: {TRANSCRIPT_PATH}")

        return

    # Parse the transcript data
    original_transcript, full_text, caption_count = parse_transcript(transcript_data)

    # Construct absolute paths for model and tokenizer
    abs_model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
    abs_tokenizer_path = os.path.join(os.path.dirname(__file__), TOKENIZER_PATH)


    # Check if model and tokenizer files exist
    if not os.path.exists(abs_model_path):
        print(f"Model file not found at {abs_model_path}. Please ensure the model is in the correct location.")
        env_var_value = os.environ.get("NEURALBLOCK_MODEL_PATH")
        if env_var_value:
             print(f"The script attempted to use the path from NEURALBLOCK_MODEL_PATH: {env_var_value}")
        else:
             print(f"The script used the default path: {MODEL_PATH}")
        return
    if not os.path.exists(abs_tokenizer_path):
        print(f"Tokenizer file not found at {abs_tokenizer_path}. Please ensure the tokenizer is in the correct location.")
        env_var_value = os.environ.get("NEURALBLOCK_TOKENIZER_PATH")
        if env_var_value:
             print(f"The script attempted to use the path from NEURALBLOCK_TOKENIZER_PATH: {env_var_value}")
        else:
             print(f"The script used the default path: {TOKENIZER_PATH}")
        return

    # Instantiate the NeuralBlockStream class
    try:
        nb_model = NeuralBlockStream(abs_model_path, abs_tokenizer_path)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print("Please ensure TensorFlow, Keras, and other dependencies are installed and the model/tokenizer files are valid.")
        return

    # Get predictions
    print("Making predictions...")
    predictions = nb_model.predict_stream(full_text)
    print("Predictions made.")

    # Get sponsor timestamps
    sponsor_timestamps = nb_model.get_sponsor_timestamps(original_transcript, caption_count, predictions)

    # Print the results
    print("Sponsor Timestamps (seconds):")
    for start, end in sponsor_timestamps:
        print(f"{start:.3f} - {end:.3f}")

if __name__ == "__main__":
    main()
