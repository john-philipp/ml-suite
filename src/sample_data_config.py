class ObjectDef:
    def __init__(self, desc=None, type=None, args=None, label=None, intensity=None):
        self.intensity = intensity
        self.label = label
        self.args = args or {}
        self.desc = desc
        self.type = type


class SampleDataConfig:
    def __init__(
            self, bins_n=100, grid_size=100, resolution=0.1, epsilon=0.0, dt=0.1,
            fov_init_pos=None, fov_init_dir=None, fov_distance=10, fov_degrees=120,
            pc_colour=None, min_x=None, max_x=None, min_y=None, max_y=None, objects=None, **_):

        self.resolution = resolution
        self.grid_size = grid_size
        self.epsilon = epsilon
        self.bins_n = bins_n
        self.dt = dt

        self.fov_init_pos = fov_init_pos or [0.3, 0.3, 0.3]
        self.fov_init_dir = fov_init_dir or [0.1, 0.1, 0.0]
        self.fov_distance = fov_distance
        self.fov_degrees = fov_degrees

        self.pc_colour = pc_colour or [0, 1, 0]

        self.min_x = -grid_size / 2 if min_x is None else min_x
        self.max_x = grid_size / 2 if max_x is None else max_x
        self.min_y = -grid_size / 2 if min_y is None else min_y
        self.max_y = grid_size / 2 if max_y is None else max_y

        self.objects = [ObjectDef(**object_args) for object_args in (objects or {})]
