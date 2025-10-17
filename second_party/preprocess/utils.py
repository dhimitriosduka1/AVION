import re


def preprocess_captions(captions):
    def lower(text):
        def replacer(match):
            word = match.group()

            # Keep word unchanged if it starts with '#' or is a single character
            if word.startswith("#") or len(word) == 1:
                return word

            return word.lower()

        # Use regex to match words
        return re.sub(r"\b\w+\b", replacer, text)

    results = []

    for c in captions:
        # 1. Strip the caption
        c = c.strip()

        # 2. Replace multiple consecutive spaces with a single space
        c = re.sub(r"\s{2,}", " ", c)

        # 3. Remove punctuation at the end of the line
        c = re.sub(r"[.,!?;:]+$", "", c)

        # 4. Convert to lowercase (only for words, not for hashtags or single characters)
        c = lower(c)

        results.append(c)

    return results
