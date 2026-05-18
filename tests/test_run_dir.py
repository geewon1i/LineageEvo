from lineage_evo.recording import RunDirectoryResolver


def test_run_directory_resolver_creates_unique_subdirs(tmp_path):
    resolver = RunDirectoryResolver(tmp_path)

    first = resolver.create(run_id="run", label="qlib")
    second = resolver.create(run_id="run", label="qlib")

    assert first != second
    assert first.exists()
    assert second.exists()
    assert first.parent == tmp_path
