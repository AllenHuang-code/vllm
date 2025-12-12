# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM CLI package.

Keep this module lightweight: importing any submodule under
`vllm.entrypoints.cli.*` triggers execution of this package's `__init__`,
so avoid eager imports of heavy dependencies (e.g. benchmarks / torch).
"""

import typing

MODULE_ATTRS: dict[str, str] = {
    "BenchmarkLatencySubcommand": ".benchmark.latency:BenchmarkLatencySubcommand",
    "BenchmarkServingSubcommand": ".benchmark.serve:BenchmarkServingSubcommand",
    "BenchmarkSweepSubcommand": ".benchmark.sweep:BenchmarkSweepSubcommand",
    "BenchmarkThroughputSubcommand": ".benchmark.throughput:BenchmarkThroughputSubcommand",
}

if typing.TYPE_CHECKING:
    from vllm.entrypoints.cli.benchmark.latency import BenchmarkLatencySubcommand
    from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
    from vllm.entrypoints.cli.benchmark.sweep import BenchmarkSweepSubcommand
    from vllm.entrypoints.cli.benchmark.throughput import BenchmarkThroughputSubcommand
else:

    def __getattr__(name: str) -> typing.Any:
        from importlib import import_module

        if name in MODULE_ATTRS:
            module_name, attr_name = MODULE_ATTRS[name].split(":")
            module = import_module(module_name, __package__)
            return getattr(module, attr_name)
        raise AttributeError(f"module {__package__} has no attribute {name}")


__all__ = list(MODULE_ATTRS.keys())
