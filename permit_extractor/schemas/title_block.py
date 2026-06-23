"""Pydantic schema for VLM-extracted title block data.

The VLM is forced to return JSON matching this schema.
All fields are Optional — absence is valid; confidence handled upstream.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class TitleBlockSchema(BaseModel):
    sheet_number: Optional[str] = Field(None, description="Sheet identifier, e.g. 'A-101'")
    sheet_title: Optional[str] = Field(None, description="Full sheet title text")
    discipline: Optional[str] = Field(
        None,
        description="Drawing discipline: ARCHITECTURAL, STRUCTURAL, MECHANICAL, ELECTRICAL, PLUMBING, CIVIL, LANDSCAPE, or other",
    )
    revision: Optional[str] = Field(None, description="Current revision number or letter")
    revision_date: Optional[str] = Field(None, description="Date of current revision, ISO format if possible")
    sheet_date: Optional[str] = Field(None, description="Original sheet issue date")
    project_name: Optional[str] = Field(None, description="Project name")
    project_number: Optional[str] = Field(None, description="Project number or permit number")
    project_address: Optional[str] = Field(None, description="Project site address")
    drawn_by: Optional[str] = Field(None, description="Initials or name of drafter")
    checked_by: Optional[str] = Field(None, description="Initials or name of checker")
    scale: Optional[str] = Field(None, description="Drawing scale, e.g. '1/4\" = 1'-0\"'")
    firm_name: Optional[str] = Field(None, description="Architecture or engineering firm name")
    north_arrow_present: Optional[bool] = Field(None, description="Whether a north arrow is visible on this sheet")
