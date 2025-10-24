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


def preprocess_caption_v2(captions):
    import re
    from wordfreq import zipf_frequency

    WORD_RE = re.compile(r"[A-Za-z]+(?:['][A-Za-z]+)*")

    def first_valid_english_word_with_index(text, *, min_zipf=2.0):
        """
        Returns a tuple (word, start_index, end_index)
        or None if no valid English word found.
        """
        for m in WORD_RE.finditer(text):
            word = m.group(0)
            wl = word.lower()

            # Allow 'a' and 'I'
            if wl in {"a", "i"}:
                return word, m.start(), m.end()

            if len(wl) == 1:
                continue

            if zipf_frequency(wl, "en") >= min_zipf:
                return word, m.start(), m.end()

        return None

    results = []
    for c in captions:
        # 1. Strip the caption
        c = c.strip()

        # 2. Replace multiple consecutive spaces with a single space
        c = re.sub(r"\s{2,}", " ", c)

        # 3. Remove punctuation at the end of the line
        c = re.sub(r"[.,!?;:]+$", "", c)

        # 4. Convert to lowercase
        c = c.lower()

        res = first_valid_english_word_with_index(c)

        if res is not None:
            results.append(c[res[1] :])
        else:
            results.append(c)

    return results
