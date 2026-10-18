"""Portable record and input validation types."""
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class Record:
    sample: str
    component: str
    year: int
    standards: str
    features: tuple[str, ...]
    label: int

    @property
    def project(self):
        return self.component
    @property
    def country(self):
        # Historical field name; E. coli runners put AST standards here.
        return self.standards
    @property
    def mechanism_features(self):
        return ('__UNSPECIFIED__',)
    def masked(self) -> "Record":
        return Record(self.sample, self.component, self.year, self.standards, self.features, -1)

    @classmethod
    def from_mapping(cls, value: dict) -> "Record":
        label = int(value.get("label", -1))
        if label not in (-1, 0, 1):
            raise ValueError("label must be -1, 0 or 1")
        return cls(str(value["sample"]), str(value["component"]), int(value.get("year", 0)),
                   str(value.get("standards", "")), tuple(str(x) for x in value.get("features", ())), label)

def validate_roles(groups: dict[str, Iterable[Record]]) -> None:
    expected = {"source", "query_pool", "audit", "evaluation"}
    if set(groups) != expected:
        raise ValueError(f"roles must be exactly {sorted(expected)}")
    sample_seen: dict[str, str] = {}
    component_seen: dict[str, str] = {}
    for role, rows in groups.items():
        for row in rows:
            if row.sample in sample_seen:
                raise ValueError(f"sample overlaps {sample_seen[row.sample]} and {role}: {row.sample}")
            if row.component in component_seen and component_seen[row.component] != role:
                raise ValueError(f"component overlaps {component_seen[row.component]} and {role}: {row.component}")
            sample_seen[row.sample] = role
            component_seen[row.component] = role
    if any(r.label not in (0,1) for r in groups["source"]):
        raise ValueError("source labels must be 0 or 1")
    if not {r.label for r in groups["source"]}.issuperset({0, 1}):
        raise ValueError("source must contain both classes")
    for role in ("query_pool", "audit", "evaluation"):
        if any(r.label != -1 for r in groups[role]):
            raise ValueError(f"{role} labels must be masked before selection")
