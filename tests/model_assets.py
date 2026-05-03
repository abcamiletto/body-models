"""Shared test asset paths and public model class names."""

from pathlib import Path

from body_models import config

ASSET_DIR = Path(__file__).parent / "assets"

TEST_ASSET_PATHS = {
    "anny": Path("anny"),
    "brainco": Path("brainco"),
    "flame": Path("flame"),
    "g1": Path("g1"),
    "garment_measurements": Path("garment_measurements"),
    "mano": Path("mano-right"),
    "mhr": Path("mhr"),
    "myofullbody": Path("myofullbody"),
    "skel": Path("skel-male"),
    "smpl": Path("smpl-neutral"),
    "smplh": Path("smplh-neutral"),
    "smplx": Path("smplx-neutral"),
    "soma": Path("soma"),
}

TEST_MODEL_FILES = TEST_ASSET_PATHS | {
    "flame": TEST_ASSET_PATHS["flame"] / "model.pkl",
    "garment_measurements": TEST_ASSET_PATHS["garment_measurements"] / "garment_measurements.npz",
    "mano": TEST_ASSET_PATHS["mano"] / "model.pkl",
    "skel": TEST_ASSET_PATHS["skel"] / "model.pkl",
    "smpl": TEST_ASSET_PATHS["smpl"] / "model.npz",
    "smplh": TEST_ASSET_PATHS["smplh"] / "model.npz",
    "smplx": TEST_ASSET_PATHS["smplx"] / "model.npz",
}

CONFIG_KEYS = {
    "anny": "anny",
    "brainco": "brainco",
    "flame": "flame",
    "g1": "g1",
    "garment_measurements": "garment-measurements",
    "mano": "mano-right",
    "mhr": "mhr",
    "myofullbody": "myofullbody",
    "skel": "skel-male",
    "smpl": "smpl-neutral",
    "smplh": "smplh-neutral",
    "smplx": "smplx-neutral",
    "soma": "soma",
}

TEST_MODEL_FILES_BY_CONFIG_KEY = {key: TEST_MODEL_FILES[name] for name, key in CONFIG_KEYS.items()}

SOMA_NESTED_MODEL_TYPES = {"anny", "mhr", "smpl", "smplx"}

CLASS_NAMES = {
    "anny": "ANNY",
    "brainco": "BrainCoHand",
    "flame": "FLAME",
    "g1": "G1",
    "garment_measurements": "GarmentMeasurements",
    "mano": "MANO",
    "mhr": "MHR",
    "myofullbody": "MyoFullBody",
    "skel": "SKEL",
    "smpl": "SMPL",
    "smplh": "SMPLH",
    "smplx": "SMPLX",
    "soma": "SOMA",
}


def get_test_asset_path(model_name: str) -> Path:
    return ASSET_DIR / TEST_ASSET_PATHS[model_name]


def get_test_model_file(model_name: str) -> Path:
    return ASSET_DIR / TEST_MODEL_FILES[model_name]


def get_test_model_file_for_config_key(config_key: str) -> Path:
    return ASSET_DIR / TEST_MODEL_FILES_BY_CONFIG_KEY[config_key]


def get_model_file(model_name: str) -> Path:
    model_path = config.get_model_path(CONFIG_KEYS[model_name])
    if model_path is not None:
        return model_path
    return get_test_model_file(model_name)
