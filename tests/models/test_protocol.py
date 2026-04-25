"""Verify that all three concrete model classes satisfy the CareerAVModel Protocol."""

import pytest

from src.models.knn import KNNTrajectoryModel
from src.models.parametric import ParametricCurveModel
from src.models.protocol import CareerAVModel
from src.models.ridge import RidgeRegressionModel


@pytest.mark.parametrize("cls", [ParametricCurveModel, KNNTrajectoryModel, RidgeRegressionModel])
def test_satisfies_protocol(cls):
    instance = cls()
    assert isinstance(instance, CareerAVModel)
