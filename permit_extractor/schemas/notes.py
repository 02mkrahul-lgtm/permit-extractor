"""Pydantic schema for VLM-extracted general notes."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class CodeClause(BaseModel):
    """A single code-relevant clause extracted from general notes."""
    clause_number: Optional[str] = None    # e.g. "1.", "A.", or None
    text: str
    topic: Optional[str] = Field(
        None,
        description="Topic tag: OCCUPANCY, CONSTRUCTION_TYPE, FIRE_RATING, SPRINKLER, EGRESS, "
                    "ACCESSIBILITY, ENERGY, STRUCTURAL, WIND, SEISMIC, PLUMBING, ELECTRICAL, or OTHER",
    )


class NotesSchema(BaseModel):
    occupancy_classification: Optional[str] = Field(
        None, description="IBC/CBC occupancy classification(s), e.g. 'A-2', 'B', 'R-2'"
    )
    construction_type: Optional[str] = Field(
        None, description="IBC/CBC construction type, e.g. 'Type V-B', 'Type III-A'"
    )
    fire_sprinkler_required: Optional[bool] = Field(
        None, description="Whether fire sprinkler system is noted as required"
    )
    applicable_codes: list[str] = Field(
        default_factory=list,
        description="List of referenced codes, e.g. ['2022 CBC', '2022 CPC', 'NFPA 13']",
    )
    fire_ratings: list[str] = Field(
        default_factory=list,
        description="Fire rating requirements mentioned, e.g. ['1-hr rated corridor', '2-hr shaft wall']",
    )
    clauses: list[CodeClause] = Field(
        default_factory=list, description="Individual extracted clauses in order"
    )
    raw_notes_text: Optional[str] = Field(
        None, description="Full verbatim text of the notes section"
    )
