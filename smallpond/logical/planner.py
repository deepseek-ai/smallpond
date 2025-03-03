from smallpond.execution.task import *
from smallpond.logical.node import *

TaskGroup = List[Task]


class Planner(LogicalPlanVisitor[TaskGroup]):
    """
    Create an execution plan (tasks) from a logical plan (nodes).
    """

    def __init__(self, runtime_ctx: RuntimeContext) -> None:
        self.runtime_ctx = runtime_ctx
        self.node_to_tasks: Dict[Node, TaskGroup] = {}

    @logger.catch(reraise=True, message="failed to build computation graph")
    def create_exec_plan(self, logical_plan: LogicalPlan, manifest_only_final_results=True) -> ExecutionPlan:
        logical_plan = copy.deepcopy(logical_plan)

        # if --output_path is specified, copy files to the output path
        # otherwise, create manifest files only
        sink_type = "copy" if self.runtime_ctx.final_output_path is not None else "manifest"
        final_sink_type = "copy" if self.runtime_ctx.final_output_path is not None else "manifest" if manifest_only_final_results else "link"

        # create DataSinkNode for each named output node (same name share the same sink node)
        nodes_groupby_output_name: Dict[str, List[Node]] = defaultdict(list)
        for node in logical_plan.nodes.values():
            if node.output_name is not None:
                if node.output_name in nodes_groupby_output_name:
                    warnings.warn(f"{node} has duplicate output name: {node.output_name}")
                nodes_groupby_output_name[node.output_name].append(node)
        sink_nodes = {}  # { output_name: DataSinkNode }
        for output_name, nodes in nodes_groupby_output_name.items():
            output_path = os.path.join(
                self.runtime_ctx.final_output_path or self.runtime_ctx.output_root,
                output_name,
            )
            sink_nodes[output_name] = DataSinkNode(logical_plan.ctx, tuple(nodes), output_path, type=sink_type)

        # create DataSinkNode for root node
        # XXX: special case optimization to avoid copying files twice
        # if root node is DataSetPartitionNode(npartitions=1), and all its input nodes are named, create manifest files instead of copying files.
        if (
            isinstance(logical_plan.root_node, ConsolidateNode)
            and len(logical_plan.root_node.input_deps) == 1
            and isinstance(
                partition_node := logical_plan.root_node.input_deps[0],
                EvenlyDistributedPartitionNode,
            )
            and all(node.output_name is not None for node in partition_node.input_deps)
        ):
            sink_nodes["FinalResults"] = DataSinkNode(
                logical_plan.ctx,
                tuple(sink_nodes[node.output_name] for node in partition_node.input_deps),
                output_path=os.path.join(
                    self.runtime_ctx.final_output_path or self.runtime_ctx.output_root,
                    "FinalResults",
                ),
                type="manifest",
                is_final_node=True,
            )
        # if root node also has output_name, create manifest files instead of copying files.
        elif (output_name := logical_plan.root_node.output_name) is not None:
            sink_nodes["FinalResults"] = DataSinkNode(
                logical_plan.ctx,
                (sink_nodes[output_name],),
                output_path=os.path.join(
                    self.runtime_ctx.final_output_path or self.runtime_ctx.output_root,
                    "FinalResults",
                ),
                type="manifest",
                is_final_node=True,
            )
        else:
            # normal case
            sink_nodes["FinalResults"] = DataSinkNode(
                logical_plan.ctx,
                (logical_plan.root_node,),
                output_path=os.path.join(
                    self.runtime_ctx.final_output_path or self.runtime_ctx.output_root,
                    "FinalResults",
                ),
                type=final_sink_type,
                is_final_node=True,
            )

        # assemble sink nodes as new root node
        root_node = RootNode(logical_plan.ctx, tuple(sink_nodes.values()))
        logical_plan = LogicalPlan(logical_plan.ctx, root_node)

        # generate tasks
        [root_task] = self.visit(root_node)
        # print logical plan with the generated runtime tasks
        logger.info(f"logical plan:{os.linesep}{str(logical_plan)}")

        exec_plan = ExecutionPlan(self.runtime_ctx, root_task, logical_plan)

        return exec_plan

    def visit(self, node: Node, depth: int = 0) -> TaskGroup:
        # memoize the tasks
        if node in self.node_to_tasks:
            return self.node_to_tasks[node]
        retvals = super().visit(node, depth)
        self.node_to_tasks[node] = retvals
        return retvals

    def visit_data_source_node(self, node: DataSourceNode, depth: int) -> TaskGroup:
        assert not node.input_deps, f"data source should be leaf node: {node}"
        return [node.create_task(self.runtime_ctx, [], [PartitionInfo()])]

    def visit_data_sink_node(self, node: DataSinkNode, depth: int) -> TaskGroup:
        all_input_deps = [task for dep in node.input_deps for task in self.visit(dep, depth + 1)]
        return [node.create_task(self.runtime_ctx, all_input_deps, [PartitionInfo()])]

    def visit_root_node(self, node: RootNode, depth: int) -> TaskGroup:
        all_input_deps = [task for dep in node.input_deps for task in self.visit(dep, depth + 1)]
        return [RootTask(self.runtime_ctx, all_input_deps, [PartitionInfo()])]

    def visit_union_node(self, node: UnionNode, depth: int) -> TaskGroup:
        all_input_deps = [task for dep in node.input_deps for task in self.visit(dep, depth + 1)]
        unique_partition_dims = set(task.partition_dims for task in all_input_deps)
        assert len(unique_partition_dims) == 1, f"cannot union partitions with different dimensions: {unique_partition_dims}"
        return all_input_deps

    def visit_consolidate_node(self, node: ConsolidateNode, depth: int) -> TaskGroup:
        input_deps_taskgroups = [self.visit(dep, depth + 1) for dep in node.input_deps]
        assert len(input_deps_taskgroups) == 1, f"consolidate node only accepts one input node, but found: {input_deps_taskgroups}"
        unique_partition_dims = set(task.partition_dims for task in input_deps_taskgroups[0])
        assert len(unique_partition_dims) == 1, f"cannot consolidate partitions with different dimensions: {unique_partition_dims}"
        existing_dimensions = set(unique_partition_dims.pop())
        assert (
            node.dimensions.intersection(existing_dimensions) == node.dimensions
        ), f"cannot found some of {node.dimensions} in {existing_dimensions}"
        # group tasks by partitions
        input_deps_groupby_partitions: Dict[Tuple, List[Task]] = defaultdict(list)
        for task in input_deps_taskgroups[0]:
            partition_infos = tuple(info for info in task.partition_infos if info.dimension in node.dimensions)
            input_deps_groupby_partitions[partition_infos].append(task)
        return [
            node.create_task(self.runtime_ctx, input_deps, partition_infos) for partition_infos, input_deps in input_deps_groupby_partitions.items()
        ]

    def visit_partition_node(self, node: PartitionNode, depth: int) -> TaskGroup:
        all_input_deps = [task for dep in node.input_deps for task in self.visit(dep, depth + 1)]
        unique_partition_dims = set(task.partition_dims for task in all_input_deps)
        assert len(unique_partition_dims) == 1, f"cannot partition input_deps with different dimensions: {unique_partition_dims}"

        if node.nested:
            assert (
                node.dimension not in unique_partition_dims
            ), f"found duplicate partition dimension '{node.dimension}', existing dimensions: {unique_partition_dims}"
            assert (
                len(all_input_deps) * node.npartitions <= node.max_card_of_producers_x_consumers
            ), f"{len(all_input_deps)=} * {node.npartitions=} > {node.max_card_of_producers_x_consumers=}"
            producer_tasks = [node.create_producer_task(self.runtime_ctx, [task], task.partition_infos) for task in all_input_deps]
            return [
                node.create_consumer_task(
                    self.runtime_ctx,
                    [producer],
                    list(producer.partition_infos) + [PartitionInfo(partition_idx, node.npartitions, node.dimension)],
                )
                for producer in producer_tasks
                for partition_idx in range(node.npartitions)
            ]
        else:
            max_num_producer_tasks = min(
                node.max_num_producer_tasks,
                math.ceil(node.max_card_of_producers_x_consumers / node.npartitions),
            )
            num_parallel_tasks = 2 * self.runtime_ctx.num_executors * math.ceil(self.runtime_ctx.usable_cpu_count / node.cpu_limit)
            num_producer_tasks = max(1, min(max_num_producer_tasks, num_parallel_tasks))
            if len(all_input_deps) < num_producer_tasks:
                merge_datasets_task = node.create_merge_task(self.runtime_ctx, all_input_deps, [PartitionInfo()])
                split_dataset_tasks = [
                    node.create_split_task(
                        self.runtime_ctx,
                        [merge_datasets_task],
                        [PartitionInfo(partition_idx, num_producer_tasks)],
                    )
                    for partition_idx in range(num_producer_tasks)
                ]
            else:
                split_dataset_tasks = [
                    node.create_merge_task(
                        self.runtime_ctx,
                        tasks,
                        [PartitionInfo(partition_idx, num_producer_tasks)],
                    )
                    for partition_idx, tasks in enumerate(split_into_rows(all_input_deps, num_producer_tasks))
                ]
            producer_tasks = [
                node.create_producer_task(self.runtime_ctx, [split_dataset], split_dataset.partition_infos) for split_dataset in split_dataset_tasks
            ]
            return [
                node.create_consumer_task(
                    self.runtime_ctx,
                    producer_tasks,
                    [
                        PartitionInfo(),
                        PartitionInfo(partition_idx, node.npartitions, node.dimension),
                    ],
                )
                for partition_idx in range(node.npartitions)
            ]

    def broadcast_input_deps(self, node: Node, depth: int):
        # if no input deps, return a single partition
        if not node.input_deps:
            yield [], (PartitionInfo(),)
            return

        input_deps_taskgroups = [self.visit(dep, depth + 1) for dep in node.input_deps]
        input_deps_most_ndims = max(
            input_deps_taskgroups,
            key=lambda taskgroup: (
                len(taskgroup[0].partition_dims),
                max(t.partition_infos for t in taskgroup),
            ),
        )
        input_deps_maps = [
            (
                taskgroup[0].partition_dims,
                dict((t.partition_infos, t) for t in taskgroup),
            )
            for taskgroup in input_deps_taskgroups
        ]

        for main_input in input_deps_most_ndims:
            input_deps = []
            for input_deps_dims, input_deps_map in input_deps_maps:
                partition_infos = tuple(info for info in main_input.partition_infos if info.dimension in input_deps_dims)
                input_dep = input_deps_map.get(partition_infos, None)
                assert (
                    input_dep is not None
                ), f"""the partition dimensions or npartitions of inputs {node.input_deps} of {repr(node)} are not compatible
  cannot match {main_input.partition_infos} against any of {input_deps_map.keys()}"""
                input_deps.append(input_dep)
            yield input_deps, main_input.partition_infos

    def visit_python_script_node(self, node: PythonScriptNode, depth: int) -> TaskGroup:
        return [
            node.create_task(self.runtime_ctx, input_deps, partition_infos) for input_deps, partition_infos in self.broadcast_input_deps(node, depth)
        ]

    def visit_arrow_compute_node(self, node: ArrowComputeNode, depth: int) -> TaskGroup:
        return [
            node.create_task(self.runtime_ctx, input_deps, partition_infos) for input_deps, partition_infos in self.broadcast_input_deps(node, depth)
        ]

    def visit_arrow_stream_node(self, node: ArrowStreamNode, depth: int) -> TaskGroup:
        return [
            node.create_task(self.runtime_ctx, input_deps, partition_infos) for input_deps, partition_infos in self.broadcast_input_deps(node, depth)
        ]

    def visit_query_engine_node(self, node: SqlEngineNode, depth: int) -> TaskGroup:
        return [
            node.create_task(self.runtime_ctx, input_deps, partition_infos) for input_deps, partition_infos in self.broadcast_input_deps(node, depth)
        ]

    def visit_projection_node(self, node: ProjectionNode, depth: int) -> TaskGroup:
        assert len(node.input_deps) == 1, f"projection node only accepts one input node, but found: {node.input_deps}"
        return [node.create_task(self.runtime_ctx, [task], task.partition_infos) for task in self.visit(node.input_deps[0], depth + 1)]

    def visit_limit_node(self, node: LimitNode, depth: int) -> TaskGroup:
        assert len(node.input_deps) == 1, f"limit node only accepts one input node, but found: {node.input_deps}"
        all_input_deps = self.visit(node.input_deps[0], depth + 1)
        partial_limit_tasks = [node.create_task(self.runtime_ctx, [task], task.partition_infos) for task in all_input_deps]
        merge_task = node.create_merge_task(self.runtime_ctx, partial_limit_tasks, [PartitionInfo()])
        global_limit_task = node.create_task(self.runtime_ctx, [merge_task], merge_task.partition_infos)
        return [global_limit_task]
