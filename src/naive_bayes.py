import torch
from collections import Counter
from typing import Dict
import math

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = len(features[0]) # Shape of the probability tensors, useful for predictions and conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        class_priors: Dict[int, torch.Tensor] = {}

        for label in labels.unique():
            mask = labels == label
            class_priors[int(label.item())] = torch.Tensor([mask.sum().item()/len(labels)])

        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        
        class_word_counts: Dict[int, torch.Tensor] = {0:torch.zeros(size=(len(features[0]),)),1:torch.zeros(size=(len(features[0]),))}

        word_count = []
        total_negative = 0
        total_positive = 0

        # iterar sobre cada palabra
        for i in range(len(features[0])):
            negative_count = 0
            positive_count = 0

            # iterar sobre cada bow
            for j in range(len(features)):
                bow = features[j]
                label = labels[j]
                if label == 0:
                    negative_count += bow[i]
                else:
                    positive_count += bow[i]
            
            total_negative += negative_count
            total_positive += positive_count
            word_count.append((negative_count, positive_count))

        total_vocab = len(features[0])
        for i in range(len(word_count)):
            negative_prob = (word_count[i][0] + delta) / (total_negative + delta*(total_vocab))
            positive_prob = (word_count[i][1] + delta) / (total_positive + delta*(total_vocab))

            class_word_counts[0][i] = negative_prob
            class_word_counts[1][i] = positive_prob

        return class_word_counts

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        
        log_posteriors: torch.Tensor = torch.empty(size=(len(self.class_priors),))

        for i in range(len(self.class_priors)):
            prob = math.log(self.class_priors[i])
            for j in range(len(feature)):
                prob += feature[j]*math.log(self.conditional_probabilities[i][j])
            log_posteriors[i] = prob

        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        class_posteriors = self.estimate_class_posteriors(feature)
        pred: int = class_posteriors.argmax().item()

        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        class_posteriors = self.estimate_class_posteriors(feature)
        probs: torch.Tensor = torch.nn.functional.softmax(class_posteriors)
        return probs
