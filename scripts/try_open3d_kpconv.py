import warnings

from open3d._ml3d.utils import Config

from src.jinja_yaml_loader import JinjaYamlLoader
from src.path_helper import PathHelper

# Ignore all warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.INFO)

import open3d.ml.torch as ml3d
from open3d._ml3d.torch import KPFCNN, SemanticSegmentation


if __name__ == '__main__':
    config_path = "../config/kpconv_semantickitti.yml"

    path_helper = PathHelper().generated().commit()
    dataset_path = path_helper.datasets().index(-1).path(rollback=True)
    training_path = path_helper.trainings().next("ffff").path(rollback=True)

    config_loader = JinjaYamlLoader(config_path, lambda **x: Config(x))
    config = config_loader.load(dataset_path=dataset_path, training_path=training_path)
    dataset = ml3d.datasets.SemanticKITTI(**config.dataset)
    model = KPFCNN(**config.model)

    pipeline = SemanticSegmentation(model=model, dataset=dataset, **config.pipeline)
    pipeline.run_train()
