import numpy as np
import json
import os
import re
import tflite_runtime.interpreter as tflite
# Removed: import tensorflow as tf 

# Use relative import for transcript_parser within the same package
from .transcript_parser import parse_transcript

class NeuralBlockStream:
    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "models/nb_stream_fasttext_10k_with_tokenizer.tflite"
    )

    def __init__(self, model_path=None): 
        effective_model_path = model_path if model_path else self.DEFAULT_MODEL_PATH
        if not os.path.exists(effective_model_path):
            raise FileNotFoundError(f"Model file not found at {effective_model_path}")
        
        try:
            print(f"Loading TFLite model (with tokenizer) from: {effective_model_path}")
            self.interpreter = tflite.Interpreter(model_path=effective_model_path)
            self.interpreter.allocate_tensors()
            print("TFLite model loaded and tensors allocated.")
        except Exception as e:
            print(f"An unexpected error occurred during model loading: {e}")
            raise

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"DEBUG: Model input details: {self.input_details}")
        # Input dtype should now be tf.string


    def predict_stream(self, text):
        if not isinstance(text, str):
             print("Warning: Input to predict_stream must be a string.")
             return np.empty([0, 2], dtype=np.float32)

        print(f"Preparing input numpy array for text (length {len(text)} chars)...")
        try:
            input_shape = self.input_details[0]['shape']
            input_dtype = self.input_details[0]['dtype'] # Check expected type
            print(f"DEBUG: Model expects input dtype: {input_dtype}")
            
            # --- Create NumPy array for string input --- 
            # Check expected shape and create numpy object array accordingly
            if tuple(input_shape) == (1,): # Shape like [1]
                 input_data = np.array([text], dtype=object) 
            elif tuple(input_shape) == (1, 1): # Shape like [1, 1]
                 input_data = np.array([[text]], dtype=object)
            else: # Fallback for other potential shapes (e.g., flexible batch size [-1])
                 print(f"Warning: Unexpected input shape {input_shape}, attempting shape (1,).")
                 input_data = np.array([text], dtype=object)
                 # If shape is [-1], reshape might be needed: input_data = input_data.reshape([1])
                 # Or for [-1, 1]: input_data = input_data.reshape([1, 1])
                 # Let's rely on the explicit shapes first.
                 
            print(f"Input numpy array created with shape: {input_data.shape}, dtype: {input_data.dtype}")

        except Exception as e:
             print(f"Error creating input numpy array: {e}")
             return np.empty([0, 2], dtype=np.float32)

        try:
            print("Setting input tensor and invoking interpreter...")
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            print("Interpreter invoked.")
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
            print(f"Raw prediction output shape: {prediction.shape}") 
            return prediction[0].round(3) 

        except Exception as e:
            print(f"Error during model invocation or output processing: {e}")
            import traceback
            traceback.print_exc()
            return np.empty([0, 2], dtype=np.float32)

    def get_sponsor_timestamps(
        self,
        transcript,
        captionCount,
        predictions,
        numWords, 
        returnText=0,
    ):
        # (This function remains unchanged from the previous version) 
        sponsorSegments = []
        startIdx = -1 
        thresh = float(os.environ.get("NB_THRESH", "0.51"))
        sFlag = 0 
        confidence = float(os.environ.get("NB_CONFIDENCE", "0.80"))
        sFlagConf = 0 
        min_segment_word_count = int(os.environ.get("NB_MIN_WORD_COUNT", "6"))
        wps = float(os.environ.get("NB_WPS", "2.8"))

        if predictions is None or len(predictions) == 0:
             print("Warning: Received empty or None predictions array for timestamp generation.")
             return [] if not returnText else ([], [])
        
        if not isinstance(predictions, np.ndarray):
             predictions = np.array(predictions)

        effective_prediction_length = min(len(predictions), numWords)
        print(f"Processing predictions up to index {effective_prediction_length-1} (numWords={numWords}, prediction_len={len(predictions)})")

        if predictions.ndim != 2 or predictions.shape[1] != 2:
             print(f"Warning: Predictions array has unexpected shape {predictions.shape}. Cannot generate timestamps.")
             return [] if not returnText else ([], [])

        for index in range(effective_prediction_length):
            row = predictions[index]
            if len(row) < 2: 
                print(f"Warning: Skipping prediction row {index} due to len < 2: {row}")
                continue
            try:
                sponsor_prob = float(row[1])
            except (TypeError, ValueError):
                 print(f"Warning: Could not convert sponsor probability in row {index} to float: {row[1]}")
                 continue

            if sponsor_prob >= thresh and not sFlag:
                startIdx = index
                sFlag = 1
                sFlagConf = 1 if sponsor_prob >= confidence else 0
                continue
                
            if sFlag:
                 if sponsor_prob >= confidence:
                      sFlagConf = 1
                 if sponsor_prob < thresh:
                      segment_len = index - startIdx
                      if sFlagConf and segment_len >= min_segment_word_count:
                           print(f"Found valid segment: ({startIdx}, {index}), len={segment_len}, conf met={sFlagConf}")
                           sponsorSegments.append((startIdx, index))
                      else:
                           pass 
                      sFlag = 0
                      sFlagConf = 0
                      startIdx = -1

        if sFlag:
            segment_len = effective_prediction_length - startIdx
            if sFlagConf and segment_len >= min_segment_word_count:
                 print(f"Found valid segment ending at text end: ({startIdx}, {effective_prediction_length}), len={segment_len}, conf met={sFlagConf}")
                 sponsorSegments.append((startIdx, effective_prediction_length))

        sponsorTimestamps = []
        sponsorText = [] 
        for segs in sponsorSegments:
            numWordsAccum = 0
            start_cap_idx = -1
            end_cap_idx = -1
            excessHead = 0
            excessTail_words = 0
            start_word_idx = segs[0]
            end_word_idx = segs[1] 

            numWordsAccum = 0
            for idx, cap_len in enumerate(captionCount):
                cap_start_word = numWordsAccum
                cap_end_word = numWordsAccum + cap_len 
                if cap_start_word <= start_word_idx < cap_end_word:
                    start_cap_idx = idx
                    excessHead = start_word_idx - cap_start_word
                    break
                numWordsAccum += cap_len
            
            numWordsAccum = 0
            for idx, cap_len in enumerate(captionCount):
                 cap_start_word = numWordsAccum
                 cap_end_word = numWordsAccum + cap_len 
                 if cap_start_word <= (end_word_idx - 1) < cap_end_word:
                      end_cap_idx = idx
                      excessTail_words = (end_word_idx - 1) - cap_start_word + 1 
                      break
                 numWordsAccum += cap_len
            
            if start_cap_idx != -1 and end_cap_idx != -1 and start_cap_idx < len(transcript) and end_cap_idx < len(transcript):
                startTime = transcript[start_cap_idx]["start"] + (excessHead / wps if wps > 0 else 0)
                endTime = transcript[end_cap_idx]["start"] + (excessTail_words / wps if wps > 0 else 0)
                endTime = min(endTime, transcript[end_cap_idx]["start"] + transcript[end_cap_idx]["duration"])
                endTime = max(startTime, endTime) 
                sponsorTimestamps.append((round(startTime, 3), round(endTime, 3)))
                if returnText:
                     sponsorText.append(f"Segment: word {start_word_idx} to {end_word_idx}")
            else:
                print(f"Warning: Could not map segment indices {segs} (words {start_word_idx}-{end_word_idx}) to transcript captions. StartCap={start_cap_idx}, EndCap={end_cap_idx}, TranscriptLen={len(transcript)}")
        
        if returnText:
            return sponsorTimestamps, sponsorText
        else:
            return sponsorTimestamps

    def process_transcript_list(self, transcript_list):
        print("Parsing transcript list...")
        original_transcript, full_text, caption_count = parse_transcript(transcript_list)
        
        if not full_text:
             print("Warning: Parsed transcript resulted in empty text.")
             return []
        
        word_list = full_text.split() 
        numWords = len(word_list)
        print(f"Parsed text: {numWords} words.")
        if numWords == 0:
             print("Warning: Parsed text has 0 words.")
             return []
             
        print("Predicting sponsor segments...")
        predictions = self.predict_stream(full_text)
        
        if predictions is None or len(predictions) == 0:
             print("Warning: predict_stream returned no predictions.")
             return []
             
        print("Calculating timestamps...")
        sponsor_timestamps = self.get_sponsor_timestamps(
            original_transcript, 
            caption_count, 
            predictions,
            numWords
        )
        print(f"Timestamp calculation complete. Found {len(sponsor_timestamps)} segments.")
        return sponsor_timestamps

