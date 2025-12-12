# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoints of vLLM

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import importlib.metadata
import sys

from vllm.logger import init_logger

logger = init_logger(__name__)


def main():
    from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    # Map subcommand -> module providing cmd_init().
    #
    # Keep this list in sync with actual CLI modules, but do NOT import them
    # here: we build a lightweight top-level parser first, then import only the
    # chosen subcommand's module to build the full parser.
    SUBCOMMAND_MODULES: dict[str, str] = {
        # openai client helpers
        "chat": "vllm.entrypoints.cli.openai",
        "complete": "vllm.entrypoints.cli.openai",
        # serving
        "serve": "vllm.entrypoints.cli.serve",
        # benchmarks
        "bench": "vllm.entrypoints.cli.benchmark.main",
        # utilities
        "collect-env": "vllm.entrypoints.cli.collect_env",
        "run-batch": "vllm.entrypoints.cli.run_batch",
    }

    SUBCOMMAND_HELP: dict[str, str] = {
        "serve": "Launch an OpenAI-compatible API server.",
        "chat": "Chat with a running OpenAI-compatible server.",
        "complete": "Request completions from a running API server.",
        "bench": "Run benchmarks (throughput/latency/serve/sweep).",
        "collect-env": "Collect environment information for debugging.",
        "run-batch": "Run batch prompts and write results to file.",
    }

    def _build_full_parser_for_subcommand(subcmd: str):
        from importlib import import_module

        module_path = SUBCOMMAND_MODULES[subcmd]
        cmd_module = import_module(module_path)

        parser = FlexibleArgumentParser(
            description="vLLM CLI",
            epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
        )
        parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=importlib.metadata.version("vllm"),
        )
        subparsers = parser.add_subparsers(required=True, dest="subparser")

        cmds = {}
        # cmd_init may return multiple commands (e.g. chat/complete).
        for cmd in cmd_module.cmd_init():
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
        return parser, cmds

    cli_env_setup()

    # For 'vllm bench *': use CPU instead of UnspecifiedPlatform by default
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        logger.debug(
            "Bench command detected, must ensure current platform is not "
            "UnspecifiedPlatform to avoid device type inference error"
        )
        from vllm import platforms

        if platforms.current_platform.is_unspecified():
            from vllm.platforms.cpu import CpuPlatform

            platforms.current_platform = CpuPlatform()
            logger.info(
                "Unspecified platform detected, switching to CPU Platform instead."
            )

    # Lightweight top-level parser: do NOT import subcommands yet.
    top_parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    top_parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=importlib.metadata.version("vllm"),
    )
    top_subparsers = top_parser.add_subparsers(required=False, dest="subparser")
    for name in sorted(SUBCOMMAND_MODULES.keys()):
        top_subparsers.add_parser(name, help=SUBCOMMAND_HELP.get(name, ""), add_help=False)

    # Parse only to detect which subcommand is requested.
    # Unknown args are intentionally tolerated; the full parser will handle them.
    top_args, _ = top_parser.parse_known_args()
    if top_args.subparser is None:
        top_parser.print_help()
        return

    parser, cmds = _build_full_parser_for_subcommand(top_args.subparser)
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)
    args.dispatch_function(args)


if __name__ == "__main__":
    main()
