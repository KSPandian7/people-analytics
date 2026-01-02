# config.py

BASE_PATH = r"E:\people-analytics\data\PETA\PETA dataset"
SUBSET = "CUHK"

IMAGE_DIR = BASE_PATH + "\\" + SUBSET + "\\archive"
LABEL_FILE = IMAGE_DIR + "\\Label.txt"

SELECTED_ATTRS = [
    "personalMale",
    "personalFemale",
    "personalLess30",
    "personalLess45",
    "upperBodyCasual",
    "upperBodyFormal",
    "upperBodyJacket",
    "upperBodyTshirt",
    "lowerBodyJeans",
    "lowerBodyTrousers",
    "carryingBackpack",
    "carryingMessengerBag"
]

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-4
NUM_WORKERS = 8
