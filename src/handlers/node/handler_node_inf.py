import os
import time

import numpy as np
from sensor_msgs.msg import PointCloud2

from src.handlers.node.methods import point_cloud_msg_as_data, points_as_point_cloud_msg
from src.labels import LEARNING_MAP_INV
from src.node.simple_node import SimpleNode, spin
from src.path_helper import PathHelper
from src.file_logger import FileLogger
from src.jinja_yaml_loader import JinjaYamlLoader
from src.methods import read_yaml
from src.path_handler import PathHandler
from src.parsers.interfaces import _Args
from src.handlers.interfaces import _Handler


class HandlerNodeInf(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.training_index = args.training_index
            self.topic_in = args.topic_in
            self.include_labels = args.include_labels
            self.visualise_predictions = args.visualise_predictions
            self.topics_out = args.topics_out

    def handle(self):
        args: HandlerNodeInf.Args = self.args
        bindings = {}

        dataset_index = -1
        training_index = -1

        dataset_path_handler = PathHandler("_generated/02-datasets")
        dataset_path = dataset_path_handler.get_path(dataset_index)

        training_path_handler = PathHandler("_generated/03-trainings")
        training_path = training_path_handler.get_path(training_index)
        bindings_path = f"{training_path}/bindings.yml"

        file_log = FileLogger(training_path)
        file_log.log(f"Handler: {self.__class__.__name__}")
        file_log.log(f"Bindings: {bindings}")
        file_log.log(f"Training: {file_log.link(training_path)}")

        # Empty bindings means, use defaults at top of config.
        if not os.path.exists(bindings_path):
            raise AssertionError(f"No bindings.yml found: {bindings_path}")

        file_log.log("Handling imports...")

        import open3d.ml.torch as ml3d
        from open3d._ml3d.utils import Config
        from open3d._ml3d.torch import RandLANet, SemanticSegmentation

        config = self.load_config(bindings_path, dataset_path, training_path, Config)
        dataset = ml3d.datasets.SemanticKITTI(**config.dataset)
        model = RandLANet(**config.model)
        pipeline = SemanticSegmentation(model=model, dataset=dataset, **config.pipeline)

        checkpoint_path = PathHelper(training_path).checkpoints().latest().path()
        pipeline.load_ckpt(checkpoint_path)
        publishers = {}

        node = SimpleNode("node_inf")
        if args.topics_out:
            topics_out = self.read_topics_out(args.topics_out)
            for topic, labels in topics_out.items():
                _, publisher = node.register_publisher(PointCloud2, topic)
                file_log.log(f"Registering publisher: topic={topic} labels={labels}")
                publishers[topic] = (publisher, labels)

        def msg_receiver(msg):
            start_time_inf = time.time()

            points = point_cloud_msg_as_data(msg)

            # We're looking to predict these labels. Zero them for now.
            labels = np.zeros(np.shape(points)[0], dtype=np.int32)

            data = {
                'point': points[:, 0:3],
                'feat': np.divide(points[:, 3:], 255),
                'label': labels,
            }

            print(f">>> {max(data['feat'])}")

            result = pipeline.run_inference(data)
            end_time_inf = time.time()
            file_log.log(f"Inference done in {end_time_inf - start_time_inf}s.")

            # Open3d-ml returns predicted labels offset by -1 (ignores unlabeled:0).
            # This is unintuitive. We have to increment by 1 to get back to our labels.
            predicted_labels = result["predict_labels"]
            predicted_labels += 1

            # Map to original labels using inv learning map.
            unique_predicted_labels = np.unique(predicted_labels)
            for unique_predicted_label in unique_predicted_labels:
                try:
                    original_label = LEARNING_MAP_INV[unique_predicted_label]
                except KeyError:
                    file_log.log(f"Couldn't invert label with ID: {unique_predicted_label}")
                    file_log.log("Setting to unlabeled.")
                    original_label = 0
                predicted_labels[predicted_labels == unique_predicted_label] = original_label

            file_log.log(f"Predicted labels: {np.unique(predicted_labels)}")

            for topic, (publisher, labels) in publishers.items():
                label_points = points[np.isin(predicted_labels, labels)]
                file_log.log(f"Publishing to topic: {topic} labels={labels} points={len(label_points)}")
                msg2 = points_as_point_cloud_msg(label_points, msg.header)
                publisher.publish(msg2)

            return result

        node.register_subscriber(PointCloud2, args.topic_in, msg_receiver)
        spin(node, file_log.log)

    @staticmethod
    def read_topics_out(topics_outs_s):
        topics_out = {}
        for topics_out_s in topics_outs_s:
            topic, labels_s = topics_out_s.split(":")
            labels = labels_s.split(",")
            topics_out[topic] = [int(x) for x in labels]
        return topics_out

    @staticmethod
    def load_config(bindings_path, dataset_path, training_path, config_cls):
        bindings = read_yaml(bindings_path)
        config_path = f"{dataset_path}/config.yml"
        config_loader = JinjaYamlLoader(config_path, lambda **x: config_cls(x))
        config = config_loader.load(dataset_path=dataset_path, training_path=training_path, **bindings)
        num_points = int(config.model["num_points"])
        num_layers = int(config.model["num_layers"])
        num_neighbors = int(config.model["num_neighbors"])

        # For lower points we need to reduce layers.
        if num_points < 512:
            num_layers = min(num_layers, 3)
        if num_points < 128:
            num_layers = min(num_layers, 2)
        if num_points < 32:
            num_layers = min(num_layers, 1)
        if num_points < 8:
            num_layers = min(num_layers, 0)
        if num_points < 2:
            raise ValueError("This won't work. Use points >= 2, please.")

        # Type preservation via jinja2 doesn't quite work here.
        config.model["num_points"] = int(num_points)
        config.model["num_layers"] = int(num_layers)
        config.model["num_neighbors"] = int(num_neighbors)
        return config
