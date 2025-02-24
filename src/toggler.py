import time


class Toggler:
    class Toggle:
        def __init__(self, sleep_s=0.1):
            self.sleep_s = sleep_s
            self.value = False

        def toggle(self):
            self.value = not self.value

        def wait(self, for_value):
            while self.value is not for_value:
                time.sleep(self.sleep_s)

        def __bool__(self):
            return self.value

        def __repr__(self):
            return f"Toggle({self.value})"

    def __init__(self):
        self.toggles = []

    def make_toggle(self):
        toggle = Toggler.Toggle()
        self.toggles.append(toggle)
        return toggle

    def wait(self, for_value):
        all([toggle.wait(for_value) for toggle in self.toggles])

    def toggle(self):
        all([toggle.toggle() for toggle in self.toggles])

    def __repr__(self):
        return f"Toggler({self.toggles})"
