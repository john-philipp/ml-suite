import importlib


class Importer:
    def __init__(self):
        self._modules_to_import = []

    def register_import(self, module):
        self._modules_to_import.append(module)
        return self

    def handle_imports(self):
        for module in self._modules_to_import:
            imported_module = importlib.import_module(module)
            setattr(self, module.replace(".", "_"), imported_module)
