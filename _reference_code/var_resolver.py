import re


class Resolver:

    RE_VALUE = "\\${([^$]+)}"

    def __init__(self, values, **special_values):
        self.values = values
        self.values.update(**special_values)

    def resolve_string(self, string):
        value_keys = re.findall(self.RE_VALUE, string)

        # For one, we take existing type.
        if len(value_keys) == 1:
            return self.values[value_keys[0]]

        # If multiple, result is expected as string.
        for value_key in value_keys:
            string = string.replace(value_key, self.values[value_key])

        return string

    def resolve_entry(self, container, x, y):
        if isinstance(y, (dict, list)):
            self.resolve(y)
        elif isinstance(y, str):
            container[x] = self.resolve_string(y)

    def resolve(self, container):
        if isinstance(container, dict):
            for x, y in container.items():
                self.resolve_entry(container, x, y)
        elif isinstance(container, list):
            for x, y in enumerate(container):
                self.resolve_entry(container, x, y)


if __name__ == '__main__':
    values = {
        "string_value": "abc",
        "float_value": 1.2,
        "int_value": 123
    }
    data = {
        "a": "${string_value}",
        "b": {
            "x": "${int_value}"
        },
        "c": [
            "${float_value}"
        ]
    }
    expected = {
        "a": "abc",
        "b": {
            "x": 123
        },
        "c": [
            1.2
        ]
    }

    resolver = Resolver(values)
    resolver.resolve(data)
    assert data == expected

