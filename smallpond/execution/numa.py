import platform
import sys


def running_on_wsl() -> bool:
    if sys.platform != "linux":
        return False
    release = platform.uname().release.lower()
    return "microsoft" in release or "wsl" in release


def get_numa_node_count() -> int:
    if sys.platform == "darwin" or running_on_wsl():
        return 1

    import numa

    return max(1, int(numa.info.get_num_configured_nodes()))
