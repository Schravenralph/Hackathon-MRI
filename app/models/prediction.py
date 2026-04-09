import json
from datetime import datetime, timezone
from uuid import UUID, uuid4
from sqlmodel import Field, Relationship, SQLModel


class Prediction(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    scan_id: UUID = Field(foreign_key="scan.id")
    model_name: str = Field(default="efficientnet_b0")
    prediction_class: str
    confidence: float
    probabilities_json: str = Field(default="{}")
    uncertainty_map_path: str | None = None
    gradcam_path: str | None = None
    ran_on_harmonized: bool = Field(default=False)
    inference_time_ms: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scan: "Scan" = Relationship(back_populates="predictions")

    @property
    def probabilities(self) -> dict[str, float]:
        return json.loads(self.probabilities_json)
