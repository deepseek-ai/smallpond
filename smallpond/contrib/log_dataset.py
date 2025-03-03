from typing import List, Tuple

from smallpond.execution.task import PythonScriptTask, RuntimeContext
from smallpond.logical.dataset import DataSet
from smallpond.logical.node import Context, Node, PythonScriptNode


class LogDataSetTask(PythonScriptTask):

    num_rows = 200

    @property
    def exec_on_scheduler(self) -> bool:
        return True

    def process(
        self,
        runtime_ctx: RuntimeContext,
        input_datasets: List[DataSet],
        output_path: str,
    ) -> bool:
        for dataset in input_datasets:
            dataset.log(self.num_rows)
        return True


class LogDataSet(PythonScriptNode):
    def __init__(self, ctx: Context, input_deps: Tuple[Node, ...], num_rows=200, **kwargs) -> None:
        super().__init__(ctx, input_deps, **kwargs)
        self.num_rows = num_rows

    def spawn(self, *args, **kwargs) -> LogDataSetTask:
        task = LogDataSetTask(*args, **kwargs)
        task.num_rows = self.num_rows
        return task
