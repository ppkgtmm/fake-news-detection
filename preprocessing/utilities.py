from nltk.tokenize import word_tokenize
import re

reserved_words = ["EMAIL", "NUMBER", "URL", "HASHTAG", "MENTION"]

token_dict = {
    "ca": "can",
    "wo": "will",
    "sha": "shall",
    "'ve": "have",
    "'ll": "will",
    "'m": "am",
    "n't": "not",
    "'re": "are",
}

word_dict = {
    "cant": "can not",
    "couldnt": "could not",
    "wont": "will not",
    "pls": "please",
    "plz": "please",
    "youre": "you are",
    "theyre": "they are",
    "ive": "I have",
    "havent": "have not",
    "hasnt": "has not",
    "hadnt": "had not",
    "im": "I am",
    "didnt": "did not",
    "dont": "do not",
    "doesnt": "does not",
    "gotta": "got to",
    "wanna": "want to",
    "gonna": "going to",
    "wannabe": "want to be",
    "cannot": "can not",
}


def substitute(doc: str):

    doc = re.sub(r"<br />", " ", doc)
    doc = re.sub(r"https?:\S+|http?:\S+", " URL ", doc)  # urls can contain @
    doc = re.sub(
        r"#\S+", " HASHTAG ", doc
    )  # hashtags can not contain symbols other than hash
    doc = re.sub(r"\S+@\S+", " EMAIL ", doc)
    doc = re.sub(r"@\S+", " MENTION ", doc)
    doc = re.sub(r"(\d+\-\d+)|\d+", " NUMBER ", doc)
    doc = re.sub(r"[^A-Za-z']", " ", doc)

    return doc


def expand(word: str):

    if word in reserved_words:
        return word

    lower = word.lower()
    if lower.strip() == "let's":
        return "let us"

    if lower.strip() == "'twas":
        return "it was"

    tokens = word_tokenize(lower)

    if len(tokens) == 1:
        return word_dict.get(tokens[0], tokens[0])

    expanded_tokens = tokens
    for i, token in enumerate(tokens):
        expanded_tokens[i] = token_dict.get(token, token)

    return " ".join(expanded_tokens)


def clean_text(doc: str):
    doc = doc.strip()
    doc = substitute(doc)

    tokens = doc.split()
    doc = " ".join([expand(w) for w in tokens])

    tokens = doc.split()
    tokens = [word for word in tokens if word.isalpha()]

    return " ".join(tokens)
