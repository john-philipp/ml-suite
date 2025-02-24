# Lifted from ml3d/datasets/_resources/semantic-kitti.yaml.
LABELS = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle",
}


LABEL_COLORS = {
    0: [0.0, 0.0, 0.0],       # "unlabeled" (black)
    1: [0.0, 0.0, 0.0],       # "outlier" (black, same as unlabeled)
    2: [0.5, 0.5, 0.5],       # "car" (gray)
    3: [0.0, 0.0, 1.0],       # "bicycle" (blue)
    4: [0.0, 1.0, 0.0],       # "motorcycle" (green)
    5: [0.0, 1.0, 1.0],       # "truck" (cyan)
    6: [1.0, 0.0, 0.0],       # "other-vehicle" (red)
    7: [1.0, 0.0, 1.0],       # "person" (magenta)
    8: [0.6, 0.0, 0.6],       # "bicyclist" (dark magenta)
    9: [0.8, 0.4, 0.0],       # "motorcyclist" (orange)
    10: [1.0, 1.0, 0.0],      # "road" (yellow)
    11: [0.5, 0.25, 0.0],     # "parking" (brown)
    12: [0.4, 0.2, 0.6],      # "sidewalk" (purple)
    13: [0.8, 0.0, 0.2],      # "other-ground" (reddish)
    14: [0.0, 0.8, 0.0],      # "building" (bright green)
    15: [0.6, 0.2, 0.0],      # "fence" (dark orange)
    16: [0.0, 0.6, 0.6],      # "vegetation" (teal)
    17: [0.0, 0.4, 0.2],      # "trunk" (forest green)
    18: [0.4, 0.0, 0.4],      # "terrain" (dark purple)
    19: [0.4, 0.4, 0.4],      # "pole" (light gray)
}


LEARNING_MAP = {
    0: 0,      # "unlabeled"
    1: 0,      # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,     # "car"
    11: 2,     # "bicycle"
    13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,     # "motorcycle"
    16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,     # "truck"
    20: 5,     # "other-vehicle"
    30: 6,     # "person"
    31: 7,     # "bicyclist"
    32: 8,     # "motorcyclist"
    40: 9,     # "road"
    44: 10,    # "parking"
    48: 11,    # "sidewalk"
    49: 12,    # "other-ground"
    50: 13,    # "building"
    51: 14,    # "fence"
    52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,     # "lane-marking" to "road" ---------------------------------mapped
    70: 15,    # "vegetation"
    71: 16,    # "trunk"
    72: 17,    # "terrain"
    80: 18,    # "pole"
    81: 19,    # "traffic-sign"
    99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,    # "moving-car" to "car" ------------------------------------mapped
    253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,    # "moving-person" to "person" ------------------------------mapped
    255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,    # "moving-truck" to "truck" --------------------------------mapped
    259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

LEARNING_MAP_INV = {
    0: 0,      # "unlabeled", and others ignored
    1: 10,     # "car"
    2: 11,     # "bicycle"
    3: 15,     # "motorcycle"
    4: 18,     # "truck"
    5: 20,     # "other-vehicle"
    6: 30,     # "person"
    7: 31,     # "bicyclist"
    8: 32,     # "motorcyclist"
    9: 40,     # "road"
    10: 44,    # "parking"
    11: 48,    # "sidewalk"
    12: 49,    # "other-ground"
    13: 50,    # "building"
    14: 51,    # "fence"
    15: 70,    # "vegetation"
    16: 71,    # "trunk"
    17: 72,    # "terrain"
    18: 80,    # "pole"
    19: 81,    # "traffic-sign"
}

CLASS_WEIGHTS = [
    0,          # 1 -> 10: car
    0,          # 2 -> 11: bicycle
    0,          # 3 -> 15: motorcycle
    0,          # 4 -> 18: truck
    0,          # 5 -> 20: other-vehicle
    0,          # 6 -> 30: person
    0,          # 7 -> 31: bicyclist
    0,          # 8 -> 32: motorcyclist
    0,          # 9 -> 40: road
    0,          # 10 -> 44: parking
    0,          # 11 -> 48: sidewalk
    0,          # 12 -> 49: other-ground
    0,          # 13 -> 50: building
    0,          # 14 -> 51: fence
    0,          # 15 -> 70: vegetation
    0,          # 16 -> 71: trunk
    0,          # 17 -> 72: terrain
    0,          # 18 -> 80: pole
    0           # 19 -> 81: traffic-sign
]
