"""Pydantic schema for VLM-extracted schedule/table data.

ScheduleRow uses a flat list of cell values (parallel to headers) rather than
dict[str, Any], which is incompatible with OpenAI's strict JSON schema mode.
The extraction code zips headers + row_cells to reconstruct dicts.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ScheduleRow(BaseModel):
    row_index: int
    row_cells: list[str] = Field(
        default_factory=list,
        description="Cell values in the same order as the headers list",
    )


class ScheduleSchema(BaseModel):
    table_type: Optional[str] = Field(
        None,
        description="Type of schedule: DOOR, WINDOW, ROOM_FINISH, PANEL, EQUIPMENT, FIXTURE, COLUMN, or OTHER",
    )
    title: Optional[str] = Field(None, description="Schedule title as printed")
    headers: list[str] = Field(default_factory=list, description="Column headers in order")
    rows: list[ScheduleRow] = Field(
        default_factory=list,
        description="Data rows — each row's row_cells aligns positionally with headers",
    )
    notes: Optional[str] = Field(None, description="Footnotes or legends below the table")
    row_count_visible: Optional[int] = Field(
        None,
        description="Number of data rows visible in this crop (confirm this matches len(rows))",
    )

    def to_row_dicts(self) -> list[dict[str, str]]:
        """Convert parallel-array rows to [{header: cell}, ...] dicts."""
        result = []
        for row in self.rows:
            cells = dict(zip(self.headers, row.row_cells))
            result.append(cells)
        return result
