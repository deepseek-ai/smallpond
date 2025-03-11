import os

from smallpond.dataframe import Session


def test_shutdown_cleanup(sp: Session):
    assert os.path.exists(sp._runtime_ctx.queue_root), "queue directory should exist"
    assert os.path.exists(sp._runtime_ctx.staging_root), "staging directory should exist"
    assert os.path.exists(sp._runtime_ctx.temp_root), "temp directory should exist"

    # create some tasks and complete them
    df = sp.from_items([1, 2, 3])
    df.write_parquet(sp._runtime_ctx.output_root)
    sp.shutdown()

    # shutdown should clean up directories
    assert not os.path.exists(sp._runtime_ctx.queue_root), "queue directory should be cleared"
    assert not os.path.exists(sp._runtime_ctx.staging_root), "staging directory should be cleared"
    assert not os.path.exists(sp._runtime_ctx.temp_root), "temp directory should be cleared"
    with open(sp._runtime_ctx.job_status_path) as fin:
        assert "success" in fin.read(), "job status should be success"


def test_shutdown_no_cleanup_on_failure(sp: Session):
    df = sp.from_items([1, 2, 3])
    try:
        # create a task that will fail
        df.map(lambda x: x / 0).compute()
    except Exception:
        pass
    else:
        raise RuntimeError("task should fail")
    sp.shutdown()

    # shutdown should not clean up directories
    assert os.path.exists(sp._runtime_ctx.queue_root), "queue directory should not be cleared"
    assert os.path.exists(sp._runtime_ctx.staging_root), "staging directory should not be cleared"
    assert os.path.exists(sp._runtime_ctx.temp_root), "temp directory should not be cleared"
    with open(sp._runtime_ctx.job_status_path) as fin:
        assert "failure" in fin.read(), "job status should be failure"
