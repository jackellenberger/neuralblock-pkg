# NeuralBlock (Package)

This repository now contains the `neuralblock` Python package directly at its root, designed to identify sponsored segments within video transcripts. It is a refactoring of the core AI model logic from the original [`neuralblock`](https://github.com/andrewzlee/NeuralBlock) project into a reusable library that is *not* dependent on the YouTube API.

## Relationship to the Original Project

The original `neuralblock` project provided a Django web application that accepted a YouTube video ID, downloaded the transcript using the YouTube API, ran the sponsored segment detection model, and displayed the results.

This package focuses solely on the core model prediction and transcript processing logic. It **removes** the dependencies on the Django web server and direct YouTube API interactions. Instead, it provides a Python package that can be imported and used programmatically with transcript data provided by the user in a specific format.

## Functionality

The main components of this package, now located at the repository root, are:

-   `neuralblock.py`: Contains the `NeuralBlockStream` class, which handles loading the trained AI model and tokenizer and making predictions on a stream of text derived from a transcript.
-   `transcript_parser.py`: Provides the `parse_transcript` function to process raw transcript data (expected to be a list of dictionaries with 'start', 'duration', 'text' keys) into the format required for the `NeuralBlockStream` class.

The package expects the model and tokenizer files to be available locally. These are now located in the `data/` directory at the repository root.

## Tuning Parameters

The behavior of the sponsored segment identification can be tuned using the following environment variables when running the application (e.g., via Docker):

-   `NB_THRESH`: **Minimum confidence threshold** to *start* and *end* a potential sponsor segment. A higher value makes segments shorter and requires higher confidence to start/end. Default: `0.60`
-   `NB_CONFIDENCE`: **Higher confidence threshold** that must be reached at some point within a potential segment for it to be considered valid. Default: `0.80`
-   `NB_MIN_WORD_COUNT`: **Minimum number of words** required for a predicted segment to be considered valid and not tossed. Default: `6`
-   `NB_WPM`: **Estimated words per minute** used for calculating timestamps within clauses when the original transcript data is not per-word. Default: `3`

These can be set when running the Docker container using the `-e` flag (e.g., `docker run -e NB_THRESH=0.50 neuralblock_package`).

## Example Usage

An example script demonstrating how to use the `neuralblock` package is located in the `examples/` directory at the repository root.

The example script (`examples/example.py`):
1.  Loads a sample transcript from a CSV file (`examples/episode_43f6d97d_segments.csv`), which follows a format similar to the original project's transcript output.
2.  Uses the `parse_transcript` function to process this data.
3.  Instantiates the `NeuralBlockStream` class, loading the model and tokenizer from `data/`.
4.  Uses the `predict_stream` method to get predictions.
5.  Uses the `get_sponsor_timestamps` method to identify the timestamps of potential sponsored segments.
6.  Prints the resulting timestamps.

The example script is configured to load the model, tokenizer, and transcript paths from environment variables (`NEURALBLOCK_MODEL_PATH`, `NEURALBLOCK_TOKENIZER_PATH`, `NEURALBLOCK_TRANSCRIPT_PATH`), with default paths pointing to the files within the `data/` and `examples/` directories at the repository root.

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
