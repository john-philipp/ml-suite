from src.trainer.interfaces import ITrainer
from src.trainer.open3d_ml_randlanet import _TrainerOpen3dMlRandlanet


class TrainerType:
    OPEN3D_ML_RANDLANET = "open3d-ml-randlanet"


def get_trainer(trainer_type: TrainerType, *args, **kwargs) -> ITrainer:
    return {
        TrainerType.OPEN3D_ML_RANDLANET: _TrainerOpen3dMlRandlanet(*args, **kwargs)
    }[trainer_type]
