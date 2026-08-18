"""EntityId: t / g / p и разбор remap-детекций."""

from app.entity_id import (
    EntityId,
    ids_from_detection,
    parse,
    parse_optional,
    person,
    group,
    tracklet,
)


def test_parse_format():
    assert parse("g1") == group(1)
    assert parse("t12") == tracklet(12)
    assert parse("p1") == person(1)
    assert parse("G3").format() == "g3"
    assert group(1).npz_key() == "g_1"
    assert group(1).npz_key("buffalo_l") == "g_1_buffalo_l"
    assert group(1).crop_stem() == "g0001"


def test_parse_rejects_bare_number():
    try:
        parse("1")
        assert False, "expected ValueError"
    except ValueError:
        pass
    assert parse_optional("1") is None
    assert parse_optional("") is None
    assert parse_optional(None) is None


def test_ids_from_detection_remap():
    mapping = {5: 1, 1: 3}
    g, t = ids_from_detection({"track_id": 1, "tracklet_id": 5}, mapping)
    assert g == group(1)
    assert t == tracklet(5)


def test_ids_from_detection_solo():
    mapping = {1: 2}
    g, t = ids_from_detection({"track_id": 2, "tracklet_id": 1}, mapping)
    assert g == group(2)
    assert t == tracklet(1)


def test_ids_from_detection_never_maps_track_id():
    mapping = {1: 9}
    g, t = ids_from_detection({"track_id": 1, "tracklet_id": 5}, mapping)
    assert t == tracklet(5)
    assert g == group(1)
    assert g != EntityId("g", 9)


def test_ids_from_detection_without_tracklet():
    g, t = ids_from_detection({"track_id": 4}, mapping={4: 99})
    assert t is None
    assert g == group(4)
