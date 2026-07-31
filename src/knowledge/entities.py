from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    aliases: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class Institution(Entity):
    type: str = ""  # e.g., "National Government", "Regulatory Body", "Emergency Services"
    allowed_domains: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class Hazard(Entity):
    related_seasons: List[str] = field(default_factory=list)  # ["rainy", "dry", "any"]
    related_topics: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class Audience(Entity):
    allowed_domains: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class Location(Entity):
    type: str = ""  # e.g., "County", "Region", "National", "Virtual"

@dataclass(frozen=True)
class Action(Entity):
    infinitive: str = ""
    imperative: str = ""
    noun: str = ""

@dataclass(frozen=True)
class RelationshipConstraint:
    institution_id: str
    audience_ids: List[str]
    action_ids: List[str]
    hazard_ids: List[str]

@dataclass(frozen=True)
class Context:
    season: str = "general"  # "rainy season", "dry spell", "general"
    weather: str = "normal conditions"  # "heavy rainfall", "hot weather", "sunny conditions"
    school_calendar: str = "normal school term"  # "registration period", "exam period"
    farming_calendar: str = "off-season"  # "planting season", "harvesting season"
    disease_activity: str = "low risk"  # "high transmission risk", "routine window"
    election_period: str = "off-election"

@dataclass(frozen=True)
class CommunicationGoal:
    id: str
    description: str
