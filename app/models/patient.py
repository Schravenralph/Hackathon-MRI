from datetime import datetime, timezone
from uuid import UUID, uuid4
from sqlmodel import Field, Relationship, SQLModel


class PatientBase(SQLModel):
    name: str = Field(index=True)
    age: int | None = None
    notes: str | None = None


class Patient(PatientBase, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scans: list["Scan"] = Relationship(
        back_populates="patient",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class PatientCreate(PatientBase):
    pass


class PatientUpdate(SQLModel):
    name: str | None = None
    age: int | None = None
    notes: str | None = None
