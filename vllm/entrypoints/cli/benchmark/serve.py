# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkServingSubcommand(BenchmarkSubcommandBase):
    """The `serve` subcommand for `vllm bench`."""

    name = "serve"
    help = "Benchmark the online serving throughput."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        from vllm.benchmarks.serve import add_cli_args

        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from vllm.benchmarks.serve import main

        main(args)
