import sys
from types import SimpleNamespace

from smallpond.execution import numa as numa_utils


def test_running_on_wsl_detects_wsl_kernel(monkeypatch):
    monkeypatch.setattr(numa_utils.sys, "platform", "linux")
    monkeypatch.setattr(
        numa_utils.platform,
        "uname",
        lambda: SimpleNamespace(release="5.15.167.4-microsoft-standard-WSL2"),
    )

    assert numa_utils.running_on_wsl()
    assert numa_utils.get_numa_node_count() == 1


def test_get_numa_node_count_returns_one_on_macos(monkeypatch):
    monkeypatch.setattr(numa_utils.sys, "platform", "darwin")

    assert numa_utils.get_numa_node_count() == 1


def test_get_numa_node_count_clamps_zero_linux_count(monkeypatch):
    fake_numa = SimpleNamespace(
        info=SimpleNamespace(get_num_configured_nodes=lambda: 0)
    )
    monkeypatch.setattr(numa_utils.sys, "platform", "linux")
    monkeypatch.setattr(numa_utils, "running_on_wsl", lambda: False)
    monkeypatch.setitem(sys.modules, "numa", fake_numa)

    assert numa_utils.get_numa_node_count() == 1


def test_get_numa_node_count_uses_linux_numa_count(monkeypatch):
    fake_numa = SimpleNamespace(
        info=SimpleNamespace(get_num_configured_nodes=lambda: 4)
    )
    monkeypatch.setattr(numa_utils.sys, "platform", "linux")
    monkeypatch.setattr(numa_utils, "running_on_wsl", lambda: False)
    monkeypatch.setitem(sys.modules, "numa", fake_numa)

    assert numa_utils.get_numa_node_count() == 4
