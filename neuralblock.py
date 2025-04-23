import numpy as np
import json
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from transcript_parser import parse_transcript


class NeuralBlockStream:
    # Default paths for model and tokenizer relative to the package directory
    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "data/nb_stream_fasttext_10k.h5"
    )
    DEFAULT_TOKENIZER_PATH = os.path.join(
        os.path.dirname(__file__), "data/tokenizer_stream_10k.json"
    )

    def __init__(self, model_path=None, tokenizer_path=None):
        # Use provided paths if available, otherwise use defaults
        effective_model_path = model_path if model_path else self.DEFAULT_MODEL_PATH
        effective_tokenizer_path = (
            tokenizer_path if tokenizer_path else self.DEFAULT_TOKENIZER_PATH
        )

        # Check if model and tokenizer files exist
        if not os.path.exists(effective_model_path):
            raise FileNotFoundError(f"Model file not found at {effective_model_path}")
        if not os.path.exists(effective_tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {effective_tokenizer_path}")

        self.model = load_model(effective_model_path)
        with open(effective_tokenizer_path) as f:
            json_obj = json.load(f)
            self.tokenizer = tokenizer_from_json(json_obj)

    def _splitSeq(self, seq, numWords, maxNumWords, overlap):
        X_trimmed = []
        X_trimmed.append(seq[0:maxNumWords])  # First case

        i = 1
        startPos = (maxNumWords - overlap) * i
        endPos = startPos + maxNumWords

        # Split until the end
        while endPos < numWords:
            X_trimmed.append(seq[startPos:endPos])
            # Update parameters
            i += 1
            startPos = (maxNumWords - overlap) * i
            endPos = startPos + maxNumWords

        # Last chunk
        X_trimmed.append(seq[startPos:numWords])

        return X_trimmed

    def predict_stream(self, text):
        full_seq = self.tokenizer.texts_to_sequences([text])
        numWords = len(full_seq[0])
        print(f"Sequence length: {numWords}")
        maxNumWords = 3000
        if numWords <= maxNumWords:
            full_seq = pad_sequences(full_seq, maxlen=maxNumWords, padding="post")
            return self.model.predict(full_seq, batch_size=1).round(3)[0]
        else:
            overlap = 500
            full_seq = self._splitSeq(full_seq[0], numWords, maxNumWords, overlap)
            full_seq = pad_sequences(full_seq, maxlen=maxNumWords, padding="post")
            prediction = self.model.predict(full_seq, batch_size=len(full_seq)).round(3)

            full_prediction = np.empty([0, 2], dtype=np.float32)
            overlapTail = np.empty([0, 2], dtype=np.float32)

            for i in prediction:
                overlapRegion = np.empty([0, 2], dtype=np.float32)
                overlapHead = i[0:overlap]

                for j in range(len(overlapTail)):  # First split is skipped because len = 0
                    maxValue = max(
                        overlapHead[j][1], overlapTail[j][1]
                    )  # Should be the same numbers, but grabbing max just in case
                    np.append(overlapRegion, (1 - maxValue, maxValue))

                full_prediction = np.concatenate(
                    (full_prediction, i[:-overlap], overlapRegion)
                )
                overlapTail = i[(maxNumWords - overlap) :]  # Extract tail n words for next iteration

            return full_prediction

    def get_sponsor_timestamps(
        self,
        transcript,
        captionCount,
        predictions,
        returnText=0,
    ):
        sponsorSegments = []
        startIdx = 0
        # Minimum confidence threshold to *start* a potential sponsor segment.
        # Lowering this value can lead to longer identified segments.
        thresh = float(os.environ.get("NB_THRESH", "0.51"))
        sFlag = 0
        # Higher confidence threshold that must be reached at some point within a potential segment.
        confidence = float(os.environ.get("NB_CONFIDENCE", "0.80"))
        sFlagConf = 0

        # Minimum number of words required for a predicted segment to be considered valid.
        # Segments shorter than this will be tossed unless they meet the confidence threshold.
        min_segment_word_count = int(os.environ.get("NB_MIN_WORD_COUNT", "6"))

        # Estimated words per second for calculating timestamps within clauses.
        wps = float(os.environ.get("NB_WPS", "2.8"))

        for index, row in enumerate(predictions):
            if row[1] >= thresh and not sFlag:
                startIdx = index
                sFlag = 1
                continue

            if sFlag and row[1] >= confidence:
                sFlagConf = 1

            if row[1] < thresh and sFlag:
                # Check if the segment meets the minimum word count AND the higher confidence threshold
                if sFlagConf and (index - startIdx) >= min_segment_word_count:
                    sponsorSegments.append((startIdx, index))
                else:
                    # Calculate approximate timestamps for the tossed segment for better logging
                    tossed_start_time = -1.0
                    tossed_end_time = -1.0
                    temp_sFlag = 0
                    numWords = 0
                    tossed_start_cap_idx = -1
                    tossed_end_cap_idx = -1

                    for cap_idx, cap_len in enumerate(captionCount):
                        if (
                            numWords <= startIdx < numWords + cap_len
                            and not temp_sFlag
                        ):
                            excessHead = max(startIdx - numWords, 0)
                            tossed_start_cap_idx = cap_idx
                            tossed_start_time = transcript[tossed_start_cap_idx][
                                "start"
                            ] + (excessHead / wps if wps > 0 else 0)
                            temp_sFlag = 1

                        if numWords <= index <= numWords + cap_len and temp_sFlag:
                            excessTail = index - (
                                numWords - cap_len if cap_len > 0 else numWords
                            )
                            tossed_end_cap_idx = cap_idx
                            tossed_end_time = min(
                                transcript[tossed_end_cap_idx]["start"]
                                + transcript[tossed_end_cap_idx]["duration"],
                                transcript[tossed_end_cap_idx]["start"]
                                + (excessTail / wps if wps > 0 else 0),
                            )
                            break
                        numWords += cap_len

                    # If end time wasn't found within a caption (e.g., index == last word), use last found caption
                    if tossed_end_cap_idx == -1 and tossed_start_cap_idx != -1:
                        tossed_end_cap_idx = len(transcript) - 1  # Assume last caption
                        tossed_end_time = (
                            transcript[tossed_end_cap_idx]["start"]
                            + transcript[tossed_end_cap_idx]["duration"]
                        )  # Use end of last caption

                    # Provide more specific reason for tossing the segment
                    reason = []
                    if not sFlagConf:
                        reason.append("failed confidence threshold")
                    if (index - startIdx) < min_segment_word_count:
                        reason.append(
                            f"segment too short ({index - startIdx} words, min {min_segment_word_count})"
                        )
                    print(
                        f"The segment (approx {tossed_start_time:.3f}s - {tossed_end_time:.3f}s) was tossed ({', '.join(reason)})."
                    )
                sFlag = 0
                sFlagConf = 0

        sponsorTimestamps = []
        sponsorText = []
        sFlag = 0  # Reset sFlag for the next loop

        for segs in sponsorSegments:
            if returnText:
                pass  # Placeholder, as 'words' is removed

            numWords = 0
            start_cap_idx = -1
            end_cap_idx = -1
            temp_sFlag = 0  # Use a temporary flag for finding start/end within this loop
            for idx, e in enumerate(captionCount):
                if numWords <= segs[0] < numWords + e and not temp_sFlag:
                    excessHead = max(segs[0] - numWords, 0)
                    start_cap_idx = idx
                    temp_sFlag = 1

                if numWords <= segs[1] <= numWords + e and temp_sFlag:
                    excessTail = segs[1] - (
                        numWords - e if e > 0 else numWords
                    )  # Corrected edge case for empty caption
                    end_cap_idx = idx
                    temp_sFlag = 0  # Reset flag as we found the end
                    break  # Found both start and end, exit loop

                numWords += e

            # If end index wasn't found in the loop (e.g., segment ends exactly at caption boundary or last word)
            if end_cap_idx == -1 and start_cap_idx != -1:
                # Iterate again or use the last known index if applicable
                current_word_count = 0
                for idx_retry, e_retry in enumerate(captionCount):
                    if idx_retry >= start_cap_idx:
                        if (
                            current_word_count
                            <= segs[1]
                            < current_word_count + e_retry
                        ):
                            end_cap_idx = idx_retry
                            excessTail = segs[1] - current_word_count
                            break
                    current_word_count += e_retry
                if end_cap_idx == -1:  # If still not found, assume end of last caption
                    end_cap_idx = len(transcript) - 1
                    excessTail = captionCount[end_cap_idx]

            if start_cap_idx != -1 and end_cap_idx != -1:  # Ensure indices were found
                startTime = transcript[start_cap_idx]["start"] + (
                    excessHead / wps if wps > 0 else 0
                )
                endTime = min(
                    transcript[end_cap_idx]["start"]
                    + transcript[end_cap_idx]["duration"],
                    transcript[end_cap_idx]["start"]
                    + (excessTail / wps if wps > 0 else 0),
                )
                sponsorTimestamps.append((round(startTime, 3), round(endTime, 3)))
            else:
                print(f"Warning: Could not map segment indices {segs} to transcript captions.")

        if returnText:
            return sponsorTimestamps, sponsorText
        else:
            return sponsorTimestamps

    def process_transcript_list(self, transcript_list):
        """
        Processes a raw transcript list to identify sponsored segments.

        Args:
            transcript_list: A list of dictionaries, where each dictionary
                             represents a transcript segment and should contain
                             'text', 'start', and 'duration' keys.

        Returns:
            A list of tuples, where each tuple represents a sponsored segment
            with (start_time, end_time) in seconds.
        """
        # Parse the raw transcript data
        original_transcript, full_text, caption_count = parse_transcript(transcript_list)

        # Make predictions on the full text
        predictions = self.predict_stream(full_text)

        # Get sponsor timestamps based on predictions
        sponsor_timestamps = self.get_sponsor_timestamps(
            original_transcript, caption_count, predictions
        )

        return sponsor_timestamps
