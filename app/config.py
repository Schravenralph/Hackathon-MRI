from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    database_url: str = "sqlite:///data/mri_platform.db"
    upload_dir: Path = Path("data/uploads")
    harmonized_dir: Path = Path("data/harmonized")
    weights_dir: Path = Path("data/weights")
    raw_data_dir: Path = Path("data/raw")
    model_checkpoint: str = "best_model.pth"
    mc_dropout_passes: int = 30
    class_names: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]
    display_names: dict[str, str] = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "notumor": "No Tumor",
        "pituitary": "Pituitary",
    }

    model_config = {"env_file": ".env"}


settings = Settings()
