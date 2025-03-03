import os.path
import tempfile
import threading
import unittest

from smallpond.io.filesystem import dump, load
from tests.test_fabric import TestFabric


class TestFilesystem(TestFabric, unittest.TestCase):
    def test_pickle_runtime_ctx(self):
        with tempfile.TemporaryDirectory(dir=self.output_root_abspath) as output_dir:
            pickle_path = os.path.join(output_dir, "runtime_ctx.pickle")
            dump(self.runtime_ctx, pickle_path)
            runtime_ctx = load(pickle_path)
            self.assertEqual(self.runtime_ctx.job_id, runtime_ctx.job_id)

    def test_pickle_trace(self):
        with self.assertRaises(TypeError) as context:
            with tempfile.TemporaryDirectory(dir=self.output_root_abspath) as output_dir:
                thread = threading.Thread()
                pickle_path = os.path.join(output_dir, "thread.pickle")
                dump(thread, pickle_path)
