import pandas as pd

from finantradealgo.data_engine.ingestion.state import FileStateStore


def test_file_state_store_roundtrip(tmp_path):
    store = FileStateStore(tmp_path / "state.json")
    run = store.start_run("job1")
    store.upsert_watermark("job1", "scope1", pd.Timestamp("2024-01-01", tz="UTC"))
    store.finish_run(run.run_id, status="succeeded")

    wm = store.get_watermark("job1", "scope1")
    assert wm == pd.Timestamp("2024-01-01", tz="UTC")
