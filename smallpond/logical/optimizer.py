from smallpond.execution.task import *
from smallpond.logical.node import *


class Optimizer(LogicalPlanVisitor[Node]):
    """
    Optimize the logical plan.
    """

    def __init__(self, exclude_nodes: Set[Node]):
        self.exclude_nodes = exclude_nodes
        """A set of nodes that will not be optimized."""
        self.optimized_node_map: Dict[Node, Node] = {}
        """A map from original node to optimized node."""

    def visit(self, node: Node, depth: int = 0) -> Node:
        # stop recursion if the node is excluded
        if node in self.exclude_nodes:
            return node
        # memoize the optimized node
        if node in self.optimized_node_map:
            return self.optimized_node_map[node]
        optimized_node = super().visit(node, depth)
        self.optimized_node_map[node] = optimized_node
        return optimized_node

    def generic_visit(self, node: Node, depth: int) -> Node:
        # by default, recursively optimize the input deps
        new_node = copy.copy(node)
        new_node.input_deps = [self.visit(dep, depth + 1) for dep in node.input_deps]
        return new_node

    def visit_query_engine_node(self, node: SqlEngineNode, depth: int) -> Node:
        # fuse consecutive SqlEngineNodes
        if len(node.input_deps) == 1 and isinstance(child := self.visit(node.input_deps[0], depth + 1), SqlEngineNode):
            fused = copy.copy(node)
            fused.input_deps = child.input_deps
            fused.udfs = node.udfs + child.udfs
            fused.cpu_limit = max(node.cpu_limit, child.cpu_limit)
            fused.gpu_limit = max(node.gpu_limit, child.gpu_limit)
            fused.memory_limit = (
                max(node.memory_limit, child.memory_limit)
                if node.memory_limit is not None and child.memory_limit is not None
                else node.memory_limit or child.memory_limit
            )
            # merge the sql queries
            # example:
            # ```
            # child.sql_queries = ["select * from {0}"]
            #  node.sql_queries = ["select a, b from {0}"]
            # fused.sql_queries = ["select a, b from (select * from {0})"]
            # ```
            fused.sql_queries = child.sql_queries[:-1] + [query.format(f"({child.sql_queries[-1]})") for query in node.sql_queries]
            return fused
        return self.generic_visit(node, depth)
