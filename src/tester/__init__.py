from src.tester.interfaces import ITester
from src.tester.open3d_ml_randlanet import _TesterOpen3dMLRandlanet


class TesterType:
    OPEN3D_ML_RANDLANET = "open3d-ml-randlanet"


def get_tester(tester_type: TesterType, *args, **kwargs) -> ITester:
    return {
        TesterType.OPEN3D_ML_RANDLANET: _TesterOpen3dMLRandlanet(*args, **kwargs)
    }[tester_type]
