from __future__ import annotations

from src.models.knn import KNNTrajectoryModel
from src.models.parametric import ParametricCurveModel
from src.models.protocol import CareerAVModel
from src.models.ridge import RidgeRegressionModel

_REGISTRY: dict[str, type] = {
    "parametric": ParametricCurveModel,
    "knn": KNNTrajectoryModel,
    "ridge": RidgeRegressionModel,
}


def make_career_av_model(name: str, **kwargs) -> CareerAVModel:
    """Instantiate a CareerAVModel by registry name.

    Parameters
    ----------
    name:
        One of "parametric", "knn", "ridge".
    **kwargs:
        Forwarded to the model's constructor.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
