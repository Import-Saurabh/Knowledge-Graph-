from src.events.event_builder import EventBuilder

def test_event_builder():
    builder = EventBuilder()
    event = builder.build_event({
        "cluster_id": 1,
        "temporal_window": "2024-W03",
        "articles": [],
        "article_ids": ["a1", "a2"]
    })
    assert event.cluster_id == 1
    assert event.temporal_window == "2024-W03"
