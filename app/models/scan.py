from datetime import datetime, timezone
from uuid import UUID, uuid4
from sqlmodel import Field, Relationship, SQLModel


class ScanBase(SQLModel):
    scanner_vendor: str = Field(default="Unknown")
    modality: str = Field(default="T1")
    file_format: str = Field(default="jpeg")


class Scan(ScanBase, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    patient_id: UUID = Field(foreign_key="patient.id")
    file_path: str = ""
    is_harmonized: bool = Field(default=False)
    harmonized_path: str | None = None
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    patient: "Patient" = Relationship(back_populates="scans")
    predictions: list["Prediction"] = Relationship(
        back_populates="scan",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class ScanCreate(ScanBase):
    patient_id: UUID
