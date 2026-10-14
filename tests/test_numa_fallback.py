import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from smallpond.execution.task import RuntimeContext


class TestNumaFallback(unittest.TestCase):
    def test_numa_node_count_windows_and_macos(self):
        ctx = RuntimeContext.__new__(RuntimeContext)
        with patch.object(sys, "platform", "win32"):
            self.assertEqual(ctx.numa_node_count, 1)

        with patch.object(sys, "platform", "darwin"):
            self.assertEqual(ctx.numa_node_count, 1)

    def test_numa_node_count_handles_none_type_error(self):
        """Simulates Issue #25 where ctypes fails with TypeError when libnuma is None."""
        ctx = RuntimeContext.__new__(RuntimeContext)
        with patch.object(sys, "platform", "linux"):
            with patch.dict(sys.modules, {"numa": None}):
                self.assertEqual(ctx.numa_node_count, 1)

    def test_numa_node_count_clamps_zero(self):
        fake_numa = SimpleNamespace(info=SimpleNamespace(get_num_configured_nodes=lambda: 0))
        ctx = RuntimeContext.__new__(RuntimeContext)
        with patch.object(sys, "platform", "linux"):
            with patch.dict(sys.modules, {"numa": fake_numa}):
                self.assertEqual(ctx.numa_node_count, 1)

    def test_numa_node_count_reports_configured_nodes(self):
        fake_numa = SimpleNamespace(info=SimpleNamespace(get_num_configured_nodes=lambda: 4))
        ctx = RuntimeContext.__new__(RuntimeContext)
        with patch.object(sys, "platform", "linux"):
            with patch.dict(sys.modules, {"numa": fake_numa}):
                self.assertEqual(ctx.numa_node_count, 4)

    def test_task_resource_guard_on_non_unix(self):
        from smallpond.execution.task import Task
        task = Task.__new__(Task)
        # Verify set_memory_limit does not crash when resource is None
        with patch("smallpond.execution.task.resource", None):
            task.set_memory_limit(1024, 2048)


if __name__ == "__main__":
    unittest.main()
