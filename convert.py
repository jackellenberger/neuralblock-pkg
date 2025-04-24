# Convert original H5, allow type inference, BUILTINS_ONLY
import tensorflow as tf
import os
import traceback

# --- Configuration ---
h5_model_path = 'neuralblock/models/nb_stream_fasttext_10k.h5'
# Overwrite the original TFLite path 
tflite_output_path = 'neuralblock/models/nb_stream_fasttext_10k.tflite' 

# --- Main Conversion Logic ---
try:
    print(f"Loading original Keras model from: {h5_model_path}")
    model = tf.keras.models.load_model(h5_model_path)
    print("Original model loaded successfully.")
    model.summary()

    print("Initializing TFLiteConverter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # --- Removed explicit input type setting --- 
    print("Letting converter infer input/output types.")
    
    # --- Force BUILTINS ONLY --- 
    print("Applying TFLite conversion settings (BUILTINS_ONLY)...")
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    converter._experimental_lower_tensor_list_ops = False
    print("Conversion settings applied.")

    print("Starting TFLite conversion (BUILTINS_ONLY)...")
    tflite_model = converter.convert()
    print("Model converted successfully to TFLite format (BUILTINS_ONLY).")

    print(f"Saving TFLite model to: {tflite_output_path}")
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)
    print("TFLite model saved successfully.")

    print("--- Conversion Complete (BUILTINS_ONLY attempt) ---")
    print(f"Converted {h5_model_path} -> {tflite_output_path}")
    print("Allowed type inference.")
    print("Only TFLITE_BUILTINS were allowed.")

except Exception as e:
    print(f"--- An error occurred during conversion (BUILTINS_ONLY attempt) ---")
    print(f"Error: {e}")
    print("This error is EXPECTED if the model uses operations not in TFLITE_BUILTINS (e.g., certain LSTM ops).")
    traceback.print_exc()
