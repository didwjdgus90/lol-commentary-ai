from pathlib import Path

import pytest

from lol_commentary_backend.ingestion.sources.patch_notes import (
    PatchNoteManifestRecord,
    build_candidate_urls,
    build_patch_sequence,
    discover_patch_note_links,
    sha256_hex,
    validate_patch_note_html,
    write_patch_note_manifest,
)


def test_builds_26_1_to_26_17_sequence() -> None:
    patches = build_patch_sequence(
        start_patch="26.1",
        end_patch="26.17",
    )

    assert len(patches) == 17

    assert patches[0] == "26.1"
    assert patches[-1] == "26.17"


def test_rejects_reverse_patch_range() -> None:
    with pytest.raises(
        ValueError,
        match=("start_patch must not be after end_patch"),
    ):
        build_patch_sequence(
            start_patch="26.17",
            end_patch="26.1",
        )


def test_rejects_cross_major_range() -> None:
    with pytest.raises(
        ValueError,
        match=("within one major"),
    ):
        build_patch_sequence(
            start_patch="26.17",
            end_patch="27.1",
        )


def test_builds_both_official_slug_candidates() -> None:
    urls = build_candidate_urls(
        patch="26.5",
        locale="ko_KR",
    )

    assert urls == (
        ("https://www.leagueoflegends.com/ko-kr/news/game-updates/patch-26-5-notes/"),
        (
            "https://"
            "www.leagueoflegends.com/"
            "ko-kr/news/game-updates/"
            "league-of-legends-"
            "patch-26-5-notes/"
        ),
    )


def test_discovers_relative_patch_links() -> None:
    html = """
    <html>
      <body>
        <a href="/ko-kr/news/game-updates/league-of-legends-patch-26-17-notes/">
          26.17 패치 노트
        </a>
        <a href="/ko-kr/news/game-updates/league-of-legends-patch-26-16-notes/">
          26.16 패치 노트
        </a>
      </body>
    </html>
    """

    discovered = discover_patch_note_links(
        index_html=html,
        index_url=("https://www.leagueoflegends.com/ko-kr/news/tags/patch-notes/"),
        target_patches=(
            "26.16",
            "26.17",
        ),
    )

    assert set(discovered) == {
        "26.16",
        "26.17",
    }

    assert discovered["26.17"].endswith("/league-of-legends-patch-26-17-notes/")


def test_discovery_does_not_confuse_26_1_with_26_10() -> None:
    html = """
    <a href="/ko-kr/news/game-updates/league-of-legends-patch-26-10-notes/">
      26.10 패치 노트
    </a>
    """

    discovered = discover_patch_note_links(
        index_html=html,
        index_url=("https://www.leagueoflegends.com/ko-kr/news/tags/patch-notes/"),
        target_patches=("26.1",),
    )

    assert discovered == {}


def test_validates_korean_patch_title() -> None:
    title = validate_patch_note_html(
        html=("<html><body><h1>26.17 패치 노트</h1></body></html>"),
        patch="26.17",
    )

    assert title == ("26.17 패치 노트")


def test_validates_english_patch_title() -> None:
    title = validate_patch_note_html(
        html=("<html><body><h1>League of Legends Patch 26.17 Notes</h1></body></html>"),
        patch="26.17",
    )

    assert "26.17" in title


def test_rejects_wrong_patch_page() -> None:
    with pytest.raises(
        ValueError,
        match=("does not match"),
    ):
        validate_patch_note_html(
            html=("<html><body><h1>26.16 패치 노트</h1></body></html>"),
            patch="26.17",
        )


def test_sha256_is_deterministic() -> None:
    first = sha256_hex(b"patch-note")

    second = sha256_hex(b"patch-note")

    assert first == second
    assert len(first) == 64


def test_manifest_is_sorted_by_patch_and_locale(
    tmp_path: Path,
) -> None:
    records = (
        PatchNoteManifestRecord(
            source_id=("patch_notes_ko_index"),
            patch="26.2",
            locale="ko_KR",
            title="26.2 패치 노트",
            source_url=(
                "https://www.leagueoflegends.com/ko-kr/news/game-updates/patch-26-2-notes/"
            ),
            file_path=("data/raw/patch_notes/26.2/ko_kr/official.html"),
            content_sha256=("a" * 64),
            byte_count=10,
            snapshot_at=("2026-01-01T00:00:00Z"),
        ),
        PatchNoteManifestRecord(
            source_id=("patch_notes_en_index"),
            patch="26.1",
            locale="en_US",
            title="Patch 26.1 Notes",
            source_url=(
                "https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-1-notes/"
            ),
            file_path=("data/raw/patch_notes/26.1/en_us/official.html"),
            content_sha256=("b" * 64),
            byte_count=10,
            snapshot_at=("2026-01-01T00:00:00Z"),
        ),
    )

    path = write_patch_note_manifest(
        repository_root=tmp_path,
        records=records,
    )

    lines = path.read_text(encoding="utf-8").splitlines()

    assert '"patch": "26.1"' in (lines[0])

    assert '"patch": "26.2"' in (lines[1])
