from ruamel.yaml import YAML
from jinja2 import Template


class JinjaYamlLoader:

    def __init__(self, file_path, obj_factory):
        self.obj_factory = obj_factory
        self.file_path = file_path

    def load(self, yaml_string=None, **bindings):

        if not yaml_string:
            with open(self.file_path) as f:
                yaml_string = "".join(f.readlines())

        pages = yaml_string.split("---")
        page_offset = 0
        yml = YAML()

        def is_bindings_page(page_):
            return any([line.strip().startswith("# BINDINGS:") for line in page_.split("\n")])

        if len(pages) > 1 and is_bindings_page(pages[0]):
            yml_bindings = yml.load(pages[0])
            yml_bindings.update(bindings)
            page_offset = 1
        else:
            yml_bindings = bindings

        for page in pages[page_offset:]:
            template = Template(page)
            rendered_page = template.render(**yml_bindings)
            yaml_page = yml.load(rendered_page) or {}
            yml_bindings.update(yaml_page)

        return self.obj_factory(**yml_bindings)
