import re


class TextPreprocessing():
    def __init__(self):
        self.stop_words = [
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
            "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
            "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
            "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
            "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
            "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
        ]

    def convert_to_lowercase(self, text_entry: str):
        return text_entry.lower()

    def replace_regex_patterns(self, text_entry: str):
        urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
        userPattern       = '@[^\s]+'
        alphaPattern      = "[^a-zA-Z0-9]"
        sequencePattern   = r"(.)\1\1+"
        seqReplacePattern = r"\1\1"
        text_entry = re.sub(urlPattern, ' URL', text_entry)
        text_entry = re.sub(userPattern, ' USER', text_entry)
        text_entry = re.sub(alphaPattern, " ", text_entry)
        text_entry = re.sub(sequencePattern, seqReplacePattern, text_entry)
        return text_entry

    def remove_stopwords(self, text_entry: str):
        # Remove stop words
        text_entry = text_entry.split()
        text_entry = " ".join([word for word in text_entry if not word in self.stop_words])
        return text_entry
