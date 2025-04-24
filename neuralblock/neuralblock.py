import numpy as np
import json
import os
import re
import tflite_runtime.interpreter as tflite

# Use relative import for transcript_parser within the same package
from .transcript_parser import parse_transcript

# --- Custom Keras Preprocessing Replacements ---

def custom_pad_sequences(sequences, maxlen, padding='post', truncating='post', value=0.0): # Default padding to float
    """
    Pads sequences to the same length. Mimics tf.keras.preprocessing.sequence.pad_sequences.
    """
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            if truncating == 'pre':
                truncated_seq = seq[-maxlen:]
            elif truncating == 'post':
                truncated_seq = seq[:maxlen]
            else:
                raise ValueError(f"Truncating type '{truncating}' not understood. Use 'pre' or 'post'.")
            padded_sequences.append(truncated_seq)
        elif len(seq) < maxlen:
            padding_needed = maxlen - len(seq)
            # Ensure padding value matches the sequence type implicitly or explicitly
            # If sequences are lists of ints, pad with int(value); if floats, pad with float(value)
            # Inferring type from sequence if possible, else use value's type
            if seq:
                 dtype = type(seq[0])
                 pad_val = dtype(value)
            else:
                 dtype = type(value) # Fallback to value's type
                 pad_val = value

            padding_values = [pad_val] * padding_needed

            if padding == 'pre':
                padded_seq = padding_values + seq
            elif padding == 'post':
                padded_seq = seq + padding_values
            else:
                raise ValueError(f"Padding type '{padding}' not understood. Use 'pre' or 'post'.")
            padded_sequences.append(padded_seq)
        else:
            padded_sequences.append(seq) # No padding or truncation needed

    # Determine final dtype based on the first sequence or padding value type
    final_dtype = np.float32 # Default to float32 as per model requirement
    if padded_sequences:
        if padded_sequences[0]: # If first sequence is not empty
             final_dtype = type(padded_sequences[0][0])
        elif isinstance(value, int):
             final_dtype = np.int32
        elif isinstance(value, float):
             final_dtype = np.float32


    if not padded_sequences:
        # Match dtype to expected model input type if empty
        # This case needs careful handling - assume float32 based on model later?
         return np.empty((0, maxlen), dtype=np.float32)


    # Create numpy array, let numpy handle dtype consistency if possible
    # We will enforce model's required dtype later
    return np.array(padded_sequences)


def load_tokenizer_data(tokenizer_path):
    """Loads configuration needed for text_to_sequences from Keras JSON."""
    try:
        with open(tokenizer_path) as f:
            outer_data_str = f.read()
            # First parse: Handle potential outer JSON string format
            data = json.loads(outer_data_str)

            # Check if the first parse resulted in a string (due to double escaping)
            if isinstance(data, str):
                # If it's a string, parse it again to get the dictionary
                data = json.loads(data)

            # Now, data should be a dictionary. Proceed with getting 'config'
            config_data = data.get('config')
            if not config_data:
                raise ValueError("Tokenizer JSON file structure seems incorrect, missing 'config' key.")
            
            # Parse config if it's a string (Keras sometimes saves it this way)
            if isinstance(config_data, str):
                config = json.loads(config_data)
            else:
                config = config_data # Assume it was already a dict

            # Get the word_index (might also be a string)
            word_index_data = config.get('word_index')
            if not word_index_data:
                raise ValueError("Tokenizer config has no 'word_index' key.")
            
            # Parse word_index if it's a string
            if isinstance(word_index_data, str):
                word_index = json.loads(word_index_data)
            else:
                word_index = word_index_data # Assume it was already a dict/list

            oov_token_name = config.get('oov_token')
            filters = config.get('filters', '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
            lower = config.get('lower', True)
            split = config.get('split', ' ')

            # Find the index for the OOV token
            if oov_token_name and oov_token_name in word_index:
                oov_token_index = word_index[oov_token_name]
            elif 'oovword' in word_index: # Fallback to default 'oovword'
                 oov_token_index = word_index['oovword']
                 print(f"Warning: Specified OOV token '{oov_token_name}' not in word_index. Using index for 'oovword': {oov_token_index}")
            else:
                oov_token_index = 1 # Keras default OOV index
                print(f"Warning: OOV token ('{oov_token_name}' or 'oovword') not found in word_index. Using default index {oov_token_index} for OOV words.")

            # Pre-compile a regex for filter characters
            escaped_filters = ''.join(['' + char if char in R'().*+?$^|[]{}' else char for char in filters])
            filter_regex = re.compile(f'([{re.escape(split)}{escaped_filters}])')

            # Return integer index for OOV token
            return word_index, int(oov_token_index), lower, split, filter_regex

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {tokenizer_path}: {e}")
        raise
    except FileNotFoundError:
        print(f"Error: Tokenizer file not found at {tokenizer_path}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading tokenizer data: {e}")
        raise

def custom_texts_to_sequences(texts, word_index, oov_token_index, lower, split, filter_regex):
    """
    Converts texts to sequences of word indices. Mimics tf.keras.preprocessing.text.Tokenizer.texts_to_sequences.
    """
    sequences = []
    for text in texts:
        if lower:
            text = text.lower()
        words = [word for word in filter_regex.sub(split, text).split(split) if word]
        # Ensure indices are integers
        sequence = [int(word_index.get(str(word), oov_token_index)) for word in words]
        sequences.append(sequence)
    return sequences

# --- End Custom Replacements ---

class NeuralBlockStream:
    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "models/nb_stream_fasttext_10k.tflite"
    )
    DEFAULT_TOKENIZER_PATH = os.path.join(
        os.path.dirname(__file__), "models/tokenizer_stream_10k.json"
    )

    def __init__(self, model_path=None, tokenizer_path=None):
        effective_model_path = model_path if model_path else self.DEFAULT_MODEL_PATH
        effective_tokenizer_path = tokenizer_path if tokenizer_path else self.DEFAULT_TOKENIZER_PATH

        if not os.path.exists(effective_model_path):
            raise FileNotFoundError(f"Model file not found at {effective_model_path}")
        if not os.path.exists(effective_tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {effective_tokenizer_path}")

        try:
            # Load tokenizer configuration using updated custom loader
            (
                self.word_index,
                self.oov_token_index, # Should be int
                self.lower,
                self.split,
                self.filter_regex,
            ) = load_tokenizer_data(effective_tokenizer_path)
            # --- DEBUG PRINT ---
            print(f"DEBUG: Loaded tokenizer. OOV index: {self.oov_token_index}")
            print(f"DEBUG: Total word_index size: {len(self.word_index)}")
            # --- END DEBUG PRINT ---
        except Exception as e:
             print(f"An unexpected error occurred during tokenizer loading: {e}")
             raise
        
        try:
            # Load TFLite model and allocate tensors
            self.interpreter = tflite.Interpreter(model_path=effective_model_path)
            self.interpreter.allocate_tensors()
        except Exception as e:
            print(f"An unexpected error occurred during model loading: {e}")
            raise

        # Get input and output tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.max_sequence_length = self.input_details[0]['shape'][-1]
        # Store the *reported* dtype (this should be float32 based on error)
        self.reported_input_dtype = self.input_details[0]['dtype'] 
        # --- DEBUG PRINT ---
        print(f"DEBUG: Model input details: {self.input_details}")
        print(f"DEBUG: Max sequence length: {self.max_sequence_length}")
        print(f"DEBUG: Model *reports* input dtype: {self.reported_input_dtype}")
        # --- END DEBUG PRINT ---


    def _splitSeq(self, seq, numWords, maxNumWords, overlap):
        # This function deals with lists of integers, should be fine
        X_trimmed = []
        if numWords <= maxNumWords:
            X_trimmed.append(seq)
            return X_trimmed
        X_trimmed.append(seq[0:maxNumWords])
        i = 1
        startPos = (maxNumWords - overlap) * i
        endPos = startPos + maxNumWords
        while endPos < numWords:
            X_trimmed.append(seq[startPos:endPos])
            i += 1
            startPos = (maxNumWords - overlap) * i
            endPos = startPos + maxNumWords
        if startPos < numWords:
            X_trimmed.append(seq[startPos:numWords])
        return X_trimmed

    def predict_stream(self, text):
        # Generates list of lists of ints
        full_seq_list = custom_texts_to_sequences(
            [text],
            self.word_index,
            self.oov_token_index,
            self.lower,
            self.split,
            self.filter_regex
        )
        if not full_seq_list:
             print("Warning: Text tokenization resulted in empty sequence.")
             return np.empty([0, 2], dtype=np.float32) # Output format is float

        full_sequence = full_seq_list[0] # This is a list of ints

        # --- DEBUG PRINT ---
        if full_sequence:
             max_index = max(full_sequence)
             min_index = min(full_sequence)
             print(f"DEBUG: Generated sequence (first 50): {full_sequence[:50]}")
             print(f"DEBUG: Max index in sequence: {max_index}, Min index: {min_index}, Length: {len(full_sequence)}")
        else:
             print("DEBUG: Generated sequence is empty.")
        # --- END DEBUG PRINT ---

        numWords = len(full_sequence)
        print(f"Sequence length: {numWords}")
        maxNumWords = self.max_sequence_length

        if numWords == 0:
            print("Warning: Sequence length is 0 after tokenization.")
            return np.empty([0, 2], dtype=np.float32) # Output format is float

        # --- REVERTED: Use model's reported dtype (float32) ---
        MODEL_INPUT_DTYPE = self.reported_input_dtype # Should be float32
        # Padding value should be float if model expects float
        PADDING_VALUE = 0.0 

        if numWords <= maxNumWords:
            # Pad the integer sequence - Resulting type depends on pad_sequences impl.
            # Let's use value=0.0 explicitly for padding float
            padded_seq_batch = custom_pad_sequences(
                [full_sequence], 
                maxlen=maxNumWords, 
                padding="post",
                value=PADDING_VALUE # Use float padding value
            )
            # Cast the padded sequence to the *reported* dtype (float32)
            input_data = padded_seq_batch.astype(MODEL_INPUT_DTYPE)
            
            # --- DEBUG PRINT ---
            print(f"DEBUG: Padded input shape: {input_data.shape}, dtype: {input_data.dtype} (MATCHING MODEL)")
            # --- END DEBUG PRINT ---
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke() # Expect potential RuntimeError here again
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
            # Return only the relevant part (up to original numWords) and ensure float output
            return prediction[0][:numWords].round(3).astype(np.float32)
        else: # Handling for sequences longer than maxNumWords
            overlap = 500
            # splitSeq returns list of lists of ints
            split_sequences = self._splitSeq(full_sequence, numWords, maxNumWords, overlap) 
            
            # Pad the split integer sequences
            padded_sequences = custom_pad_sequences(
                split_sequences, 
                maxlen=maxNumWords, 
                padding="post",
                value=PADDING_VALUE # Use float padding value
            )
            # Cast the batch of padded sequences to the *reported* dtype (float32)
            input_batch = padded_sequences.astype(MODEL_INPUT_DTYPE)
            
            # --- DEBUG PRINT ---
            print(f"DEBUG: Chunked input batch shape: {input_batch.shape}, dtype: {input_batch.dtype} (MATCHING MODEL)")
            # --- END DEBUG PRINT ---

            predictions_list = []
            num_chunks = input_batch.shape[0]

            for i in range(num_chunks):
                # Ensure each chunk is correctly shaped [1, maxNumWords]
                input_chunk = np.expand_dims(input_batch[i], axis=0) 
                self.interpreter.set_tensor(self.input_details[0]['index'], input_chunk)
                self.interpreter.invoke() # Expect potential RuntimeError here again
                output_chunk = self.interpreter.get_tensor(self.output_details[0]['index'])
                 # output_chunk shape is likely (1, maxNumWords, num_classes), we need the middle part
                predictions_list.append(output_chunk[0]) # Get shape (maxNumWords, num_classes)

            # ... rest of the chunk processing ... (Assume original logic is okay)
            prediction_chunks_array = np.array(predictions_list).round(3)
            final_predictions = []
            for idx, chunk_pred in enumerate(prediction_chunks_array): 
                actual_chunk_len = len(split_sequences[idx]) 
                if idx == 0: 
                     non_overlap_end = maxNumWords - overlap
                     effective_end = min(non_overlap_end, actual_chunk_len)
                     final_predictions.append(chunk_pred[:effective_end]) 
                else: 
                     prev_chunk_pred = prediction_chunks_array[idx-1]
                     prev_actual_chunk_len = len(split_sequences[idx-1])
                     overlap_start_in_prev = maxNumWords - overlap
                     len_overlap_in_prev = max(0, prev_actual_chunk_len - overlap_start_in_prev)
                     len_overlap_in_curr = min(overlap, actual_chunk_len)
                     len_effective_overlap = min(len_overlap_in_prev, len_overlap_in_curr)

                     if len_effective_overlap > 0:
                         overlap_prev = prev_chunk_pred[overlap_start_in_prev : overlap_start_in_prev + len_effective_overlap]
                         overlap_curr = chunk_pred[0 : len_effective_overlap]
                         max_prob_overlap = np.maximum(overlap_prev[:, 1], overlap_curr[:, 1])
                         combined_overlap_segment = np.vstack((1 - max_prob_overlap, max_prob_overlap)).T
                         final_predictions.append(combined_overlap_segment)

                     non_overlap_start_curr = len_effective_overlap
                     non_overlap_end_curr = actual_chunk_len
                     if non_overlap_start_curr < non_overlap_end_curr:
                         final_predictions.append(chunk_pred[non_overlap_start_curr:non_overlap_end_curr])

            if final_predictions:
                 valid_parts = [part for part in final_predictions if isinstance(part, np.ndarray) and part.ndim == 2 and part.shape[0] > 0 and part.shape[1] == 2]
                 if valid_parts:
                     full_prediction = np.concatenate(valid_parts, axis=0)
                 else:
                     print("Warning: No valid prediction parts after chunk processing.")
                     full_prediction = np.empty([0, 2], dtype=np.float32)
            else:
                 print("Warning: No final predictions generated after chunk processing.")
                 full_prediction = np.empty([0, 2], dtype=np.float32)

            if full_prediction.shape[0] > numWords:
                 print(f"Warning: Final prediction length ({full_prediction.shape[0]}) > numWords ({numWords}). Trimming.")
                 full_prediction = full_prediction[:numWords]
            elif full_prediction.shape[0] < numWords:
                 print(f"Info: Final prediction length ({full_prediction.shape[0]}) < numWords ({numWords}).")

            return full_prediction.astype(np.float32) # Ensure float output


    def get_sponsor_timestamps(
        self,
        transcript,
        captionCount,
        predictions,
        returnText=0,
    ):
        # This part should be okay as it receives float predictions
        sponsorSegments = []
        startIdx = 0
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

        if predictions.ndim != 2 or predictions.shape[1] != 2:
             print(f"Warning: Predictions array has unexpected shape {predictions.shape}. Cannot generate timestamps.")
             return [] if not returnText else ([], [])

        for index, row in enumerate(predictions):
            if not isinstance(row, (np.ndarray, list, tuple)) or len(row) < 2:
                 print(f"Warning: Skipping prediction row {index} due to unexpected format or length: {row}")
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
                           sponsorSegments.append((startIdx, index))
                      sFlag = 0
                      sFlagConf = 0
        if sFlag:
            segment_len = len(predictions) - startIdx
            if sFlagConf and segment_len >= min_segment_word_count:
                 sponsorSegments.append((startIdx, len(predictions)))

        sponsorTimestamps = []
        sponsorText = []
        # Need the original full_text here if returnText=1
        # Let's assume process_transcript_list provides it or we retrieve it again if needed

        for segs in sponsorSegments:
            # Timestamp calculation logic (seems okay)
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
                     # This text extraction is still a placeholder/approximation
                     # It requires the full text which might not be directly available here
                     sponsorText.append(f"Segment from word {start_word_idx} to {end_word_idx}") # Placeholder
            else:
                print(f"Warning: Could not map segment indices {segs} to transcript captions. StartCap={start_cap_idx}, EndCap={end_cap_idx}, TranscriptLen={len(transcript)}")
        
        if returnText:
            return sponsorTimestamps, sponsorText
        else:
            return sponsorTimestamps

    def process_transcript_list(self, transcript_list):
        """
        Processes a raw transcript list to identify sponsored segments.
        """
        original_transcript, full_text, caption_count = parse_transcript(transcript_list)
        if not full_text:
             print("Warning: Parsed transcript resulted in empty text.")
             return []
        
        predictions = self.predict_stream(full_text)
        
        if predictions is None or len(predictions) == 0:
             print("Warning: predict_stream returned no predictions.")
             return []
             
        sponsor_timestamps = self.get_sponsor_timestamps(
            original_transcript, caption_count, predictions
            # If get_sponsor_timestamps needs full_text, pass it here
        )
        return sponsor_timestamps
