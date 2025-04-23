# NeuralBlock Package

This repository now contains the `neuralblock` Python package directly at its root, designed to identify sponsored segments within video transcripts. It is a refactoring of the core AI model logic from the original [`neuralblock`](https://github.com/andrewzlee/NeuralBlock) project into a reusable library that is *not* dependent on the YouTube API.

## Relationship to the Original Project

The original `neuralblock` project provided a Django web application that accepted a YouTube video ID, downloaded the transcript using the YouTube API, ran the sponsored segment detection model, and displayed the results.

This package focuses solely on the core model prediction and transcript processing logic. It **removes** the dependencies on the Django web server and direct YouTube API interactions. Instead, it provides a Python package that can be imported and used programmatically with transcript data provided by the user in a specific format.

## Functionality

The main components of this package, now located at the repository root, are:

-   `neuralblock.py`: Contains the `NeuralBlockStream` class. Its `__init__` method can now be called without arguments, and it will automatically look for the trained AI model (`nb_stream_fasttext_10k.h5`) and tokenizer (`tokenizer_stream_10k.json`) in the `data/` directory within the installed package. You can still provide `model_path` and `tokenizer_path` arguments to the constructor if you need to load models from a different location. The class includes methods for making predictions on a stream of text and identifying sponsor timestamps (`process_transcript_list`).
-   `transcript_parser.py`: Provides the `parse_transcript` function to process raw transcript data (expected to be a list of dictionaries with 'start', 'duration', 'text' keys) into the format required internally. This function is used by `NeuralBlockStream.process_transcript_list` internally, so users typically won't need to interact with it directly.

The package expects the model and tokenizer files to be available locally. By default, `NeuralBlockStream` looks for these in the `data/` directory at the repository root when installed.

## Tuning Parameters

The behavior of the sponsored segment identification can be tuned using the following environment variables when running the application (e.g., via Docker):

-   `NB_THRESH`: **Minimum confidence threshold** to *start* and *end* a potential sponsor segment. A higher value makes segments shorter and requires higher confidence to start/end. Default: `0.60`
-   `NB_CONFIDENCE`: **Higher confidence threshold** that must be reached at some point within a potential segment for it to be considered valid. Default: `0.80`
-   `NB_MIN_WORD_COUNT`: **Minimum number of words** required for a predicted segment to be considered valid and not tossed. Default: `6`
-   `NB_WPS`: **Estimated words per second** used for calculating timestamps within clauses when the original transcript data is not per-word. Default: `3.0`

These can be set when running the Docker container using the `-e` flag (e.g., `docker run -e NB_THRESH=0.50 neuralblock_package`).

## Example Usage

An example script demonstrating how to use the `neuralblock` package is located in the `examples/` directory at the repository root.

The example script (`examples/example.py`):
1.  Loads a sample transcript from a CSV file (`examples/episode_43f6d97d_segments.csv`), which follows a format similar to the original project's transcript output.
2.  Instantiates the `NeuralBlockStream` class **without arguments**, relying on the default model and tokenizer paths within the package.
3.  Uses the `process_transcript_list` method on the `NeuralBlockStream` instance to process the transcript and get sponsor timestamps.
4.  Prints the resulting timestamps.

The example script is configured to load the transcript path from an environment variable (`NEURALBLOCK_TRANSCRIPT_PATH`), with a default path pointing to the example file in the `examples/` directory.

```python
import csv
import os

# Read transcript path from environment variable with default
TRANSCRIPT_PATH = os.environ.get("NEURALBLOCK_TRANSCRIPT_PATH", "examples/episode_43f6d97d_segments.csv")

# Import NeuralBlockStream
from neuralblock import NeuralBlockStream

# ... (load_transcript_from_csv function) ...

def main():
    # Load transcript data
    transcript_data = load_transcript_from_csv(TRANSCRIPT_PATH)
    if not transcript_data:
        print(f"Could not load transcript data from {TRANSCRIPT_PATH}. Please ensure the file exists and is correctly formatted.")
        # ... (environment variable check) ...
        return

    # Instantiate the NeuralBlockStream class without arguments
    try:
        nb_model = NeuralBlockStream() # Instantiating without arguments
    except FileNotFoundError as e:
        print(f"Error loading model or tokenizer: {e}")
        print("Please ensure the model and tokenizer files exist in the expected 'data/' directory within the package or provide custom paths.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during model instantiation: {e}")
        print("Please ensure TensorFlow, Keras, and other dependencies are installed and the model/tokenizer files are valid.")
        return

    # Process the transcript list to identify sponsored segments
    print("Processing transcript and making predictions...")
    sponsor_timestamps = nb_model.process_transcript_list(transcript_data)
    print("Processing complete.")

    # Print the results
    print("
Sponsor Timestamps (seconds):
")
    for start, end in sponsor_timestamps:
        print(f"{start:.3f} - {end:.3f}")

if __name__ == "__main__":
    main()
```

## Running the Example with Docker

A `Dockerfile` is included at the repository root to build a containerized environment for running the example script. To build and run the Docker image from the project's root directory:

```bash
# Build the image
docker build -t neuralblock_package .

# Run the container
docker run neuralblock_package
```

This will build an image based on the `Dockerfile` at the root, using the entire repository as the build context. It copies the package code and data into the container, installs the necessary dependencies, and then runs the example script.

This package provides the core sponsored segment detection functionality as a reusable Python library, allowing integration into different applications or workflows without the overhead of the original web server.
