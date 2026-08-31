import json

import pytest

from lol_commentary_backend.retrieval.aliases.builder import (
    build_bilingual_alias_catalog,
)
from lol_commentary_backend.retrieval.aliases.expander import (
    expand_bilingual_entity_query,
)
from lol_commentary_backend.retrieval.aliases.models import (
    AliasEntityType,
)


def _bytes(data: dict[str, object]) -> bytes:
    return json.dumps(
        data,
        ensure_ascii=False,
    ).encode("utf-8")


def _catalog():
    return build_bilingual_alias_catalog(
        ddragon_version="16.1.1",
        ko_locale="ko_KR",
        en_locale="en_US",
        ko_champion_bytes=_bytes(
            {
                "data": {
                    "Tryndamere": {
                        "key": "23",
                        "name": "트린다미어",
                    },
                    "Aphelios": {
                        "key": "523",
                        "name": "아펠리오스",
                    },
                }
            }
        ),
        en_champion_bytes=_bytes(
            {
                "data": {
                    "Tryndamere": {
                        "key": "23",
                        "name": "Tryndamere",
                    },
                    "Aphelios": {
                        "key": "523",
                        "name": "Aphelios",
                    },
                }
            }
        ),
        ko_item_bytes=_bytes({"data": {"3508": {"name": "정수 약탈자"}}}),
        en_item_bytes=_bytes({"data": {"3508": {"name": "Essence Reaver"}}}),
    )


def test_champion_is_joined_by_numeric_key() -> None:
    catalog = _catalog()

    tryndamere = next(
        item
        for item in catalog.aliases
        if item.entity_type == AliasEntityType.CHAMPION and item.entity_key == "23"
    )

    assert tryndamere.ko_name == "트린다미어"
    assert tryndamere.en_name == "Tryndamere"


def test_item_is_joined_by_item_id() -> None:
    catalog = _catalog()

    essence = next(item for item in catalog.aliases if item.entity_type == AliasEntityType.ITEM)

    assert essence.entity_key == "3508"
    assert essence.en_name == "Essence Reaver"


def test_english_item_adds_korean_alias() -> None:
    expansion = expand_bilingual_entity_query(
        "Essence Reaver AD nerf hotfix",
        _catalog(),
    )

    assert expansion.changed is True
    assert "정수 약탈자" in expansion.expanded_query


def test_korean_item_adds_english_alias() -> None:
    expansion = expand_bilingual_entity_query(
        "정수 약탈자 가격 변경",
        _catalog(),
    )

    assert "Essence Reaver" in expansion.expanded_query


def test_english_champion_adds_korean_alias() -> None:
    expansion = expand_bilingual_entity_query(
        "Tryndamere E damage buff",
        _catalog(),
    )

    assert "트린다미어" in expansion.expanded_query


def test_unknown_query_is_unchanged() -> None:
    expansion = expand_bilingual_entity_query(
        "ARAM system update",
        _catalog(),
    )

    assert expansion.changed is False
    assert expansion.original_query == expansion.expanded_query


def test_existing_bilingual_query_is_not_duplicated() -> None:
    expansion = expand_bilingual_entity_query(
        "Tryndamere 트린다미어 E",
        _catalog(),
    )

    assert expansion.changed is False


def test_empty_query_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        expand_bilingual_entity_query(
            "  ",
            _catalog(),
        )
