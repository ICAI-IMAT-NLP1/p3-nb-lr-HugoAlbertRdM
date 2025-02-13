from typing import List, Dict
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    examples: List[SentimentExample] = []

    with open(infile, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            pair = line.rstrip().split("\t")
            words = pair[0].split(" ")
            label = int(pair[-1])
            examples.append(SentimentExample(words, label))
    
    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    vocab: Dict[str, int] = {}
    index = 0

    for example in examples:
        for word in example.words:
            if word not in vocab.keys():
                vocab[word] = index
                index += 1

    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    bow: torch.Tensor = torch.zeros(size = (len(vocab),))
    
    for word in text:
        if word in vocab.keys():
            if binary:
                bow[vocab[word]] = 1
            else:
                bow[vocab[word]] += 1

    return bow