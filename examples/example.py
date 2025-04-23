import csv
import os

# Import the package
from neuralblock import NeuralBlockStream

# Paths are relative to the location of this example script (examples)
# Read transcript path from environment variables with defaults
TRANSCRIPT_PATH = os.environ.get(
    "NEURALBLOCK_TRANSCRIPT_PATH", "episode_43f6d97d_segments.csv"
)


def load_transcript_from_csv(csv_path):
    """Loads transcript data from a CSV file."""
    transcript_list = []
    # Construct the absolute path to the CSV file
    abs_csv_path = os.path.join(os.path.dirname(__file__), csv_path)
    try:
        with open(abs_csv_path, mode="r", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Skip header row
            for row in reader:
                # Assuming format: start, duration, text
                if len(row) == 3:
                    try:
                        start = float(row[0])
                        duration = float(row[1])
                        text = row[2]
                        transcript_list.append(
                            {"start": start, "duration": duration, "text": text}
                        )
                    except ValueError:
                        print(f"Skipping invalid row (non-numeric start/duration?): {row}")
                else:
                    print(f"Skipping row with unexpected format (not 3 columns): {row}")
    except FileNotFoundError:
        print(f"Error: Transcript file not found at {abs_csv_path}")
        return None
    except Exception as e:
        print(f"Error reading transcript file {abs_csv_path}: {e}")
        return None

    return transcript_list


def main():
    # Load transcript data
    transcript_data = load_transcript_from_csv(TRANSCRIPT_PATH)
    if transcript_data is None:
        print(
            f"Could not load transcript data from {TRANSCRIPT_PATH}. Please ensure the file exists and is correctly formatted."
        )
        # If using environment variables, print the value that was used
        env_var_value = os.environ.get("NEURALBLOCK_TRANSCRIPT_PATH")
        if env_var_value:
            print(
                f"The script attempted to use the path from NEURALBLOCK_TRANSCRIPT_PATH: {env_var_value}"
            )
        else:
            print(f"The script used the default path: {TRANSCRIPT_PATH}")
        return

    # Instantiate the NeuralBlockStream class without arguments
    try:
        nb_model = NeuralBlockStream()  # Instantiating without arguments
    except FileNotFoundError as e:  # Catch FileNotFoundError specifically for missing model/tokenizer
        print(f"Error loading model or tokenizer: {e}")
        print(
            "Please ensure the model and tokenizer files exist in the expected 'data/' directory within the package or provide custom paths."
        )
        return
    except Exception as e:
        print(f"An unexpected error occurred during model instantiation: {e}")
        print(
            "Please ensure TensorFlow, Keras, and other dependencies are installed and the model/tokenizer files are valid."
        )
        return

    # Process the transcript list to identify sponsored segments
    print("Processing transcript and making predictions...")
    sponsor_timestamps = nb_model.process_transcript_list(transcript_data)
    print("Processing complete.")

    # Print the results
    print("Sponsor Timestamps (seconds):")
    for start, end in sponsor_timestamps:
        print(f"{start:.3f} - {end:.3f}")


if __name__ == "__main__":
    main()
