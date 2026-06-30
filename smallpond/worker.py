# this file is the entry point for smallpond workers

import argparse
import os
import socket
import subprocess

import psutil

from smallpond.execution.numa import get_numa_node_count


def start_ray_worker(ray_address: str, cpu_count: int, memory: int = None):
    command = [
        "ray",
        "start",
        "--address",
        ray_address,
        "--num-cpus",
        str(cpu_count),
    ]
    if memory is not None:
        command.extend(["--memory", str(memory)])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="smallpond worker")
    parser.add_argument(
        "--ray_address",
        required=True,
        help="The address of the Ray cluster to connect to",
    )
    parser.add_argument("--log_dir", required=True, help="The directory where logs will be stored")
    parser.add_argument(
        "--bind_numa_node",
        action="store_true",
        help="Bind executor processes to numa nodes",
    )

    args = parser.parse_args()
    log_path = os.path.join(args.log_dir, f"{socket.gethostname()}.log")

    # limit the number of CPUs to the number of physical cores
    cpu_count = psutil.cpu_count(logical=False)
    memory = psutil.virtual_memory().total

    if args.bind_numa_node:
        numa_node_count = get_numa_node_count()
        if numa_node_count == 1:
            start_ray_worker(args.ray_address, cpu_count, memory)
        else:
            cpu_count_per_socket = cpu_count // numa_node_count
            memory_per_socket = memory // numa_node_count
            for i in range(numa_node_count):
                subprocess.run(
                    [
                        "numactl",
                        "-N",
                        str(i),
                        "-m",
                        str(i),
                        "ray",
                        "start",
                        "--address",
                        args.ray_address,
                        "--num-cpus",
                        str(cpu_count_per_socket),
                        "--memory",
                        str(memory_per_socket),
                    ],
                    check=True,
                )
    else:
        start_ray_worker(args.ray_address, cpu_count)

    # keep printing logs
    while True:
        try:
            subprocess.run(["tail", "-F", log_path], check=True)
        except subprocess.CalledProcessError as e:
            # XXX: sometimes it raises `No such file or directory`
            # don't know why. just ignore it
            print(e)
