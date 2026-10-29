"""Zones et folds de l'index, sans raster."""

from edgerunner.data.index import assign_folds, clusters

ARMS = ((0, 0), (5, 0), (-5, 0), (0, 5), (0, -5))


def cross(x: int, y: int, arms: tuple = ARMS) -> list[tuple[int, int]]:
    return [(x + dx, y + dy) for dx, dy in arms]


def tile_ids(dept: str, xy: list[tuple[int, int]]) -> list[str]:
    return [f"{int(dept)}-2025-{x:04d}-{y:04d}-LA93-0M20-E080" for x, y in xy]


def test_clusters_follow_grid_columns_despite_truncated_coastal_cross() -> None:
    a, b, d = cross(100, 6800), cross(100, 6830), cross(135, 6800)
    c = cross(130, 6830, ((0, 0), (-5, 0), (0, -5)))  # mean x 128.3: same column as d
    assert clusters(set(a + b + c + d)) == [sorted(z) for z in (a, b, d, c)]


def test_folds_group_zones_one_test_zone_per_dept_and_frozen_annotations() -> None:
    by_dept = {
        dept: tile_ids(dept, [t for i in range(6) for t in cross(100 + 25 * i, y)])
        for dept, y in (("022", 6800), ("029", 6900))
    }
    by_dept["004"] = tile_ids("004", [(900, 6300), (905, 6300)])
    out = assign_folds(by_dept)
    assert out == assign_folds(by_dept) != assign_folds(by_dept, seed=1)
    assert {out[t] for t in by_dept["004"]} == {("annot_004", "ext_test")}

    zones: dict[str, set[str]] = {}
    for dept in ("022", "029"):
        for t in by_dept[dept]:
            zones.setdefault(out[t][0], set()).add(out[t][1])
        assert sum(out[t][1] == "test" for t in by_dept[dept]) == 5  # one cross
    assert all(len(f) == 1 for f in zones.values())  # a zone never straddles two folds
    folds = [f.pop() for f in zones.values()]
    assert sorted(folds) == ["0", "0", "1", "1", "2", "2", "3", "3", "4", "4", "test", "test"]
