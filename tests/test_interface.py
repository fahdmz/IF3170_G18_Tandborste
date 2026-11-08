import pytest

from src.models.decision_tree import DecisionTree
from src.models.knn import KNN
from src.models.logistic_regression import LogisticRegression
from src.models.naive_bayes import NaiveBayes
from src.models.svm import SVM

MODELS = [KNN, DecisionTree, LogisticRegression, SVM, NaiveBayes]


@pytest.mark.parametrize("model_class", MODELS)
def test_get_params_and_clone(model_class):
    model = model_class()
    params = model.get_params()
    assert "random_state" in params
    clone = model.clone()
    assert clone is not model
    assert clone.get_params() == params


@pytest.mark.parametrize("model_class", MODELS)
def test_set_params_roundtrip(model_class):
    model = model_class()
    name = next(k for k in model.get_params() if k != "random_state")
    value = model.get_params()[name]
    model.set_params(**{name: value})
    assert model.get_params()[name] == value


@pytest.mark.parametrize("model_class", MODELS)
def test_repr_contains_class_name(model_class):
    assert model_class.__name__ in repr(model_class())
