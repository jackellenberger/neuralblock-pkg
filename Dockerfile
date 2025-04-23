# Use a slim Python 3.8 image as the base
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the working directory /app
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the contents of the package directory (build context) into the working directory /app
# This copies __init__.py, neuralblock.py, setup.py, data/, examples/, etc. into /app
COPY . /app/

# Install the package locally in editable mode from the /app directory
RUN pip install -e /app

# Define environment variables for tuning parameters with default values
ENV NB_THRESH=0.60
ENV NB_CONFIDENCE=0.80
ENV NB_MIN_WORD_COUNT=6
ENV NB_WPS=2.8

# Set the command to run the example script relative to the WORKDIR (/app)
CMD ["python", "examples/example.py"]
