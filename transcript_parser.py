import re

def parse_transcript(transcript_list):
    """
    Parses a list of transcript segments into full text and caption word counts.

    Args:
        transcript_list: A list of dictionaries, where each dictionary
                         represents a transcript segment and should contain
                         'text', 'start', and 'duration' keys.

    Returns:
        A tuple containing:
        - transcript_list (the original list)
        - fullText (a single string of all cleaned text)
        - captionCount (a list of word counts for each segment)
    """
    chars = "(!|\"|#|\$|%|&|\(|\)|\*|\+|,|-|\.|/|:|;\<|=|>|\?|@|\[|\\\\|\]|\^|_|`|\{|\||\}|~|\t|\n)+"
    captionCount = []
    fullText = ""

    for t in transcript_list:
        # Clean the text: replace newlines with spaces, remove special characters, and multiple spaces
        cleaned_text = re.sub("  +", " ", re.sub(chars, " ", t["text"])).strip()
        # Calculate word count for the cleaned text
        captionCount.append(len(cleaned_text.split(" ")))
        # Concatenate cleaned text to fullText
        fullText = fullText + " " + cleaned_text

    # Remove leading/trailing whitespace from the full text
    fullText = fullText.strip()

    return transcript_list, fullText, captionCount
