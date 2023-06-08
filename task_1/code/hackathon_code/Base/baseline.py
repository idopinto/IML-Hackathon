
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import NoReturn
import numpy as np
from sklearn.metrics import precision_score, recall_score


class BaseEstimator(ABC):
    """
    Base class of supervised estimators (classifiers and regressors)
    """

    def __init__(self) -> BaseEstimator:
        """
        Initialize a supervised estimator instance

        Attributes
        ----------
        fitted_ : bool
            Indicates if estimator has been fitted. Set by ``self.fit`` function
        """
        self.fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """
        Fit estimator for given input samples and responses

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        After fitting sets ``self.fitted_`` attribute to `True`
        """


        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Raises
        ------
        ValueError is raised if ``self.predict`` was called before calling ``self.fit``
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling ``predict``")

        # Generate the array using the binomial distribution
        return np.random.binomial(n=1, p=0.5, size=X.shape[0])


    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function specified for estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function specified for estimator

        Raises
        ------
        ValueError is raised if ``self.loss`` was called before calling ``self.fit``
        """
        from sklearn.metrics import f1_score
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling ``loss``")
        return f1_score(y, self.predict(X), zero_division=0)

    def get_recall_precision(self,X, y):
        return precision_score(y, self.predict(X)), recall_score(y, self.predict(X))
