import html
import json
import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    PatchEntityType,
)

_TAG_RE = re.compile(r"<[^>]+>")

_DESCRIPTION_PATTERNS = {
    "ability_haste": re.compile(r"(?:^|\s)스킬 가속\s+([+-]?\d+(?:\.\d+)?)"),
    "base_mana_regen_percent": re.compile(
        r"(?:^|\s)기본 마나 재생\s+"
        r"([+-]?\d+(?:\.\d+)?)%"
    ),
    "omnivamp_percent": re.compile(
        r"(?:^|\s)모든 피해 흡혈\s+"
        r"([+-]?\d+(?:\.\d+)?)%"
    ),
}


class DataDragonResourceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_url: HttpUrl
    fetched_at: str = Field(min_length=1)
    status_code: int = Field(ge=200, lt=300)
    content_type: str | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)


class DataDragonMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    collector_version: str = Field(min_length=1)
    ddragon_version: str = Field(min_length=1)
    locale: str = Field(min_length=1)
    resources: dict[str, DataDragonResourceMetadata]


@dataclass(frozen=True)
class CatalogEntity:
    entity_type: PatchEntityType
    entity_id: str
    entity_key: str | None
    name: str
    source_sha256: str
    map_ids: frozenset[str] = field(default_factory=frozenset)
    item_evidence: tuple[tuple[str, str], ...] = ()

    def evidence_dict(self) -> dict[str, str]:
        return dict(self.item_evidence)


@dataclass(frozen=True)
class EntityCatalog:
    ddragon_version: str
    locale: str
    champions_by_name: dict[str, tuple[CatalogEntity, ...]]
    items_by_name: dict[str, tuple[CatalogEntity, ...]]

    def exact_candidates(
        self,
        name: str,
        *,
        entity_type: PatchEntityType | None = None,
    ) -> tuple[CatalogEntity, ...]:
        normalized_name = normalize_entity_name(name)

        if entity_type == PatchEntityType.CHAMPION:
            return self.champions_by_name.get(
                normalized_name,
                (),
            )

        if entity_type == PatchEntityType.ITEM:
            return self.items_by_name.get(
                normalized_name,
                (),
            )

        return (
            *self.champions_by_name.get(normalized_name, ()),
            *self.items_by_name.get(normalized_name, ()),
        )


def normalize_entity_name(name: str) -> str:
    normalized = unicodedata.normalize("NFC", name)
    return " ".join(normalized.split())


def _verify_resource(
    path: Path,
    metadata: DataDragonResourceMetadata,
) -> bytes:
    content = path.read_bytes()

    if len(content) != metadata.size_bytes:
        raise ValueError(f"Data Dragon size mismatch: {path.name}")

    actual_sha256 = sha256(content).hexdigest()
    if actual_sha256 != metadata.sha256:
        raise ValueError(f"Data Dragon SHA-256 mismatch: {path.name}")

    return content


def _load_json_object(
    content: bytes,
    *,
    resource_name: str,
) -> dict[str, object]:
    payload = json.loads(content)

    if not isinstance(payload, dict):
        raise ValueError(f"{resource_name} must contain a JSON object")

    return payload


def _append_entity(
    index: dict[str, list[CatalogEntity]],
    entity: CatalogEntity,
) -> None:
    key = normalize_entity_name(entity.name)
    index.setdefault(key, []).append(entity)


def _freeze_index(
    index: dict[str, list[CatalogEntity]],
) -> dict[str, tuple[CatalogEntity, ...]]:
    return {name: tuple(entities) for name, entities in index.items()}


def _map_ids(
    raw_entity: dict[str, object],
) -> frozenset[str]:
    raw_maps = raw_entity.get("maps")

    if not isinstance(raw_maps, dict):
        return frozenset()

    return frozenset(str(map_id) for map_id, enabled in raw_maps.items() if enabled is True)


def _clean_description(value: object) -> str:
    text = html.unescape(str(value))
    text = _TAG_RE.sub(" ", text)
    return " ".join(text.split())


def _number_string(value: object) -> str | None:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None

    normalized = number.normalize()

    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal("1")))

    return format(normalized, "f")


def _stat_value(
    stats: dict[str, object],
    key: str,
) -> str:
    value = stats.get(key, 0)
    normalized = _number_string(value)
    return normalized if normalized is not None else "0"


def _description_evidence(
    description: str,
) -> dict[str, str]:
    evidence: dict[str, str] = {}

    for field_name, pattern in _DESCRIPTION_PATTERNS.items():
        match = pattern.search(description)

        if match is None:
            continue

        normalized = _number_string(match.group(1))

        if normalized is not None:
            evidence[field_name] = normalized

    return evidence


def _item_evidence(
    raw_entity: dict[str, object],
) -> tuple[tuple[str, str], ...]:
    evidence: dict[str, str] = {}

    raw_stats = raw_entity.get("stats")
    stats = raw_stats if isinstance(raw_stats, dict) else {}

    evidence["attack_damage"] = _stat_value(
        stats,
        "FlatPhysicalDamageMod",
    )
    evidence["ability_power"] = _stat_value(
        stats,
        "FlatMagicDamageMod",
    )
    evidence["health"] = _stat_value(
        stats,
        "FlatHPPoolMod",
    )
    evidence["mana"] = _stat_value(
        stats,
        "FlatMPPoolMod",
    )

    raw_gold = raw_entity.get("gold")

    if isinstance(raw_gold, dict):
        total_gold = _number_string(raw_gold.get("total"))

        if total_gold is not None:
            evidence["total_gold"] = total_gold

    description = _clean_description(raw_entity.get("description", ""))
    evidence.update(_description_evidence(description))

    return tuple(sorted(evidence.items()))


def load_entity_catalog(
    raw_dir: Path,
) -> EntityCatalog:
    metadata_path = raw_dir / "metadata.json"
    champion_path = raw_dir / "champion.json"
    item_path = raw_dir / "item.json"

    metadata = DataDragonMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))

    champion_metadata = metadata.resources.get("champion")
    item_metadata = metadata.resources.get("item")

    if champion_metadata is None:
        raise ValueError("Data Dragon champion metadata is missing")

    if item_metadata is None:
        raise ValueError("Data Dragon item metadata is missing")

    champion_content = _verify_resource(
        champion_path,
        champion_metadata,
    )
    item_content = _verify_resource(
        item_path,
        item_metadata,
    )

    champion_payload = _load_json_object(
        champion_content,
        resource_name="champion.json",
    )
    item_payload = _load_json_object(
        item_content,
        resource_name="item.json",
    )

    if champion_payload.get("version") != metadata.ddragon_version:
        raise ValueError("Champion Data Dragon version mismatch")

    if item_payload.get("version") != metadata.ddragon_version:
        raise ValueError("Item Data Dragon version mismatch")

    champion_data = champion_payload.get("data")
    item_data = item_payload.get("data")

    if not isinstance(champion_data, dict):
        raise ValueError("champion.json data must be an object")

    if not isinstance(item_data, dict):
        raise ValueError("item.json data must be an object")

    champion_index: dict[str, list[CatalogEntity]] = {}
    item_index: dict[str, list[CatalogEntity]] = {}

    for raw_entity in champion_data.values():
        if not isinstance(raw_entity, dict):
            continue

        entity_id = raw_entity.get("id")
        entity_key = raw_entity.get("key")
        name = raw_entity.get("name")

        if not all(isinstance(value, str) for value in (entity_id, entity_key, name)):
            continue

        _append_entity(
            champion_index,
            CatalogEntity(
                entity_type=PatchEntityType.CHAMPION,
                entity_id=entity_id,
                entity_key=entity_key,
                name=name,
                source_sha256=champion_metadata.sha256,
            ),
        )

    for item_id, raw_entity in item_data.items():
        if not isinstance(item_id, str):
            continue

        if not isinstance(raw_entity, dict):
            continue

        name = raw_entity.get("name")

        if not isinstance(name, str):
            continue

        _append_entity(
            item_index,
            CatalogEntity(
                entity_type=PatchEntityType.ITEM,
                entity_id=item_id,
                entity_key=None,
                name=name,
                source_sha256=item_metadata.sha256,
                map_ids=_map_ids(raw_entity),
                item_evidence=_item_evidence(raw_entity),
            ),
        )

    return EntityCatalog(
        ddragon_version=metadata.ddragon_version,
        locale=metadata.locale,
        champions_by_name=_freeze_index(champion_index),
        items_by_name=_freeze_index(item_index),
    )
