import numpy as np
import json
import os # Added import for environment variables
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

class NeuralBlockStream:
    def __init__(self, model_path, tokenizer_path):
        self.model = load_model(model_path)
        with open(tokenizer_path) as f:
            json_obj = json.load(f)
            self.tokenizer = tokenizer_from_json(json_obj)

    def _splitSeq(self, seq, numWords, maxNumWords, overlap):
        X_trimmed = []
        X_trimmed.append(seq[0:maxNumWords]) #First case

        i = 1
        startPos = (maxNumWords-overlap)*i
        endPos = startPos + maxNumWords

        #Split until the end
        while endPos < numWords:
            X_trimmed.append(seq[startPos:endPos])
            #Update parameters
            i += 1
            startPos = (maxNumWords-overlap)*i
            endPos = startPos + maxNumWords

        #Last chunk
        X_trimmed.append(seq[startPos:numWords])

        return X_trimmed

    def predict_stream(self, text):
        full_seq = self.tokenizer.texts_to_sequences([text])
        numWords = len(full_seq[0])
        print("Sequence length: {}".format(numWords))
        maxNumWords = 3000
        if numWords <= maxNumWords:
            full_seq = pad_sequences(full_seq, maxlen = maxNumWords, padding = "post")
            return self.model.predict(full_seq, batch_size = 1).round(3)[0]
        else:
            overlap = 500
            full_seq = self._splitSeq(full_seq[0], numWords, maxNumWords, overlap)
            full_seq = pad_sequences(full_seq, maxlen = maxNumWords, padding = "post")
            prediction = model.predict(full_seq, batch_size = len(full_seq)).round(3)

            full_prediction = np.empty([0,2], dtype = np.float32)
            overlapTail = np.empty([0,2], dtype = np.float32)

            for i in prediction:
                overlapRegion = np.empty([0,2], dtype = np.float32)
                overlapHead = i[0:overlap]

                for j in range(len(overlapTail)): #First split is skipped because len = 0
                    maxValue = max(overlapHead[j][1],overlapTail[j][1]) #Should be the same numbers, but grabbing max just in case
                    np.append(overlapRegion,(1-maxValue, maxValue))

                full_prediction = np.concatenate((full_prediction,i[:-overlap],overlapRegion))
                overlapTail = i[(maxNumWords-overlap):] #Extract tail n words for next iteration

            return full_prediction

    def get_sponsor_timestamps(self, transcript, captionCount, predictions, returnText = 0):
        sponsorSegments = []
        startIdx = 0
        # Minimum confidence threshold to *start* a potential sponsor segment.
        # Lowering this value can lead to longer identified segments.
        thresh = float(os.environ.get("NB_THRESH", "0.60"))
        sFlag = 0
        # Higher confidence threshold that must be reached at some point within a potential segment.
        confidence = float(os.environ.get("NB_CONFIDENCE", "0.80"))
        sFlagConf = 0

        # Minimum number of words required for a predicted segment to be considered valid.
        # Segments shorter than this will be tossed unless they meet the confidence threshold.
        min_segment_word_count = int(os.environ.get("NB_MIN_WORD_COUNT", "6"))


        for index,row in enumerate(predictions):
            if row[1] >= thresh and not sFlag:
                startIdx = index
                sFlag = 1
                continue

            if sFlag and row[1] >= confidence:
                sFlagConf = 1

            if row[1] < thresh and sFlag:
                # Check if the segment meets the minimum word count AND the higher confidence threshold
                if sFlagConf and (index - startIdx) >= min_segment_word_count:
                    sponsorSegments.append((startIdx,index))
                else:
                    print(f"The segment ({startIdx},{index}) was tossed (failed confidence or min word count).") # Updated toss message
                sFlag = 0
                sFlagConf = 0

        sponsorTimestamps = []
        sponsorText = []
        sFlag = 0
        # Estimated words per minute for calculating timestamps within clauses.
        wpm = int(os.environ.get("NB_WPM", "3"))

        for segs in sponsorSegments:
            if returnText:
                 pass # Placeholder, as 'words' is removed

            numWords = 0
            for idx, e in enumerate(captionCount):
                if numWords <= segs[0] < numWords+e and not sFlag: # Corrected comparison to correctly find the starting caption
                    excessHead = max(segs[0] - numWords, 0) # every word before the starting word in the caption
                    startIdx = idx
                    sFlag = 1

                if numWords <= segs[1] <= numWords+e and sFlag: # Corrected comparison to correctly find the ending caption
                    excessTail = segs[1]-(numWords-e) # how many words in the current caption to keep
                    endIdx = idx
                    sFlag = 0
                    break

                numWords += e


            startTime = transcript[startIdx]["start"] + (excessHead / wpm if wpm > 0 else 0) # Added check for division by zero
            endTime = min(transcript[endIdx]["start"] + transcript[endIdx]["duration"],
                          transcript[endIdx]["start"] + (excessTail / wpm if wpm > 0 else 0)) # Added check for division by zero

            sponsorTimestamps.append((round(startTime,3),round(endTime,3)))

        if returnText:
            return sponsorTimestamps, sponsorText
        else:
            return sponsorTimestamps
