from physmel_lv_world.atlas.records import collate_feature_batch, synthetic_batch
from physmel_lv_world.atlas.sources import known_sources


def test_sources_are_listed() -> None:
    sources = known_sources()
    assert len(sources) == 3
    assert sources[0].url.startswith("https://")


def test_synthetic_batch_shapes() -> None:
    batch = synthetic_batch(4, 12, 10, 3, 5)
    assert batch.pathology.shape == (4, 12)
    assert batch.genomics.shape == (4, 10)
    assert batch.immune.shape == (4, 3)


def test_collate_feature_batch() -> None:
    batch = synthetic_batch(3, 5, 4, 2, 9)
    items = []
    for idx in range(3):
        items.append(
            {
                "case_id": batch.case_id[idx],
                "pathology": batch.pathology[idx],
                "genomics": batch.genomics[idx],
                "immune": batch.immune[idx],
                "time": batch.time[idx],
                "event": batch.event[idx],
            }
        )
    joined = collate_feature_batch(items)
    assert joined.pathology.shape == (3, 5)
