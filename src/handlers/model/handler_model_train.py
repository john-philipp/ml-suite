from src.parsers.enums import ArchitectureType, DataFormatType, ModelType
from src.parsers.interfaces import _Args
from src.handlers.interfaces import _Handler
from src.this_env import GLOBALS
from src.trainer import get_trainer, TrainerType


log = GLOBALS.log


class HandlerModelTrain(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.train_into = args.train_into
            self.training_index = args.training_index
            self.dataset_index = args.dataset_index
            self.weights_index = args.weights_index

            self.epochs = args.epochs
            self.split = args.split

            self.data_format = args.data_format
            self.model = args.model
            self.arch = args.arch

            self.bindings_kvp = args.bindings_kvp
            self.bindings_json = args.bindings_json

    def handle(self):
        args: HandlerModelTrain.Args = self.args

        if args.arch == ArchitectureType.OPEN3D_ML:
            if args.data_format == DataFormatType.KITTI:
                if args.model == ModelType.RANDLANET:
                    log.info(f"Will try to train using: arch={args.arch} model={args.model}")
                    # NB: This requires the correct venv.
                    # It's encouraged to keep trainer envs
                    # separate to avoid a dependency mess.
                    trainer = get_trainer(
                        trainer_type=TrainerType.OPEN3D_ML_RANDLANET,
                        config_name="config.yml",
                        args=args)
                    trainer.train()
                    return

        raise NotImplementedError()

