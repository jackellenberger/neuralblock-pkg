# prompt: # download model from https://github.com/andrewzlee/NeuralBlock/raw/refs/heads/master/data/models/nb_stream_fasttext_10k.h5
import tensorflow as tf
from tensorflow import lite
from tensorflow.keras.models import load_model

h5_model_path = 'models/nb_stream_fasttext_10k.h5'
tflite_model_path = 'nb_stream_fasttext_10k.tflite'

try:
    print("Loading Keras model...")
    # Ensure you are using tf.keras.models.load_model
    model = tf.keras.models.load_model(h5_model_path)
    print("Model loaded successfully.")

    print("Initializing TFLiteConverter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    print("Converter initialized.")

    # Apply the settings suggested by the error message
    print("Applying TFLite conversion settings...")
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TFLite builtin ops.
        tf.lite.OpsSet.SELECT_TF_OPS     # Enable Select TensorFlow ops.
    ]
    converter._experimental_lower_tensor_list_ops = False
    converter.inference_input_type = tf.int32
    print("Conversion settings applied.")

    print("Starting TFLite model conversion...")
    tflite_model = converter.convert()
    print("Model converted successfully to TFLite format.")

    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_model_path}")

except Exception as e:
    print(f"An error occurred during the conversion process: {e}")
    print("Traceback:")
    import traceback
    traceback.print_exc()
