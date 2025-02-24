from src.parsers.enums import ModelActionType
from src.parsers.interfaces import _Parser


class ParserModelBuildSampleData(_Parser):
    @classmethod
    def add_args(cls, parent_parser):
        
        parser = parent_parser.add_parser(
            description="Build sample data to train on.",
            name=ModelActionType.BUILD_SAMPLE_DATA,
            help="Build sample data.")

        parser.add_argument(
            "--config-path",
            help="Sample data config to use (default=%(default)s).",
            default="config/sample-data-config.yaml")

        parser.add_argument(
            "--show-frames",
            help="Show frames during build.",
            action="store_true")

        parser.add_argument(
            "--show-frames-dt",
            help="Show frames during build for seconds (default=%(default)s).",
            default=0.1,
            type=float)

        parser.add_argument(
            "--bindings-kvp", "-b",
            help="Define bindings as kvp pairs (key:value[:int|float|str|bool]). Takes priority over JSON.",
            nargs="+")

        parser.add_argument(
            "--bindings-json", "-b2",
            help="Define bindings as a JSON string for more complex bindings.")

        return parser
