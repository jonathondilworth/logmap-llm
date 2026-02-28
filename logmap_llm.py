"""
Docstring for logmap-llm.logmap_llm
"""

# Python script driver for LogMap-LLM
# 
# This Python script driver for LogMap-LLM provides a command-line,
# non-interactive LogMap-LLM user experience. You launch LogMap-LLM
# at the command line and it does everything for you, writing its
# output to the console.

from __future__ import annotations

import sys
import os
import os.path
from datetime import datetime, timezone
from argparse import (
    ArgumentParser,
    Namespace,
)
from log_utils import (
    TeeWriter,
    info,
    step,
    success,
)
from pipeline_utils import (
    PipelinePaths,
)
from pipeline_config import (
    load_and_validate_config,
    print_config_summary
)
from logmap_interface import (
    start_jvm,
    LogMapInterface,
)
from pipeline_steps import (
    PipelineContext,
    align,
    prompt_build,
    consult_oracle,
    refine_alignment,
)


def main(args: Namespace):
    """
    Docstring for main

    :param args: Description
    :type args: Namespace
    """

    # TIMESTAMP RUN TIME

    expr_run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # MANAGE (LOAD & VALIDATE) CONFIG

    config_path = args.config

    cfg = load_and_validate_config(
        config_path,
        reuse_align=args.reuse_align,
        reuse_prompts=args.reuse_prompts
    )

    # LOAD (FILE) PATH LOCATIONS FROM `cfg`

    run_paths = PipelinePaths.from_config(cfg)

    os.makedirs(run_paths.output_dir, exist_ok=True)
    os.makedirs(run_paths.initial_dir, exist_ok=True)
    os.makedirs(run_paths.refined_dir, exist_ok=True)

    # START LOGGING OUTPUT TO BOTH STD::OUT AND LOG FILE

    tee_branch_out = TeeWriter(
        str(run_paths.run_log(expr_run_timestamp)),
        sys.stdout
    )
    sys.stdout = tee_branch_out

    try:

        # PRINT EXP PARAMS:
        # recursively parses the loaded `cfg` and displays parameters
        # outputs all file path locations constructed for the exp run
        # (useful for log files, tracability and reproducibility)

        print_config_summary(cfg)

        info(f"\n\nSummary of File Paths:\n\n{run_paths.summary()}\n\n")

        # INITIALISE LOGMAP
        # LogMaps `parameters.txt` file should be present in the dir
        # specified under `alignmentTask.logmap_parameters_dirpath`
        # in config, otherwise the script assumes `LogMap` dir is in cwd

        if cfg.alignmentTask.logmap_parameters_dirpath is not None:
            logmap_dirpath = cfg.alignmentTask.logmap_parameters_dirpath
        else:
            logmap_dirpath = os.path.join(os.getcwd(), 'logmap')

        # functional call to jpype (required for LogMap & bridging)
        start_jvm(logmap_dir=logmap_dirpath)

        # simple API for interfacing with LogMap via python
        logmap = LogMapInterface(cfg, logmap_dirpath)
        logmap.set_output_dir(run_paths.initial_dir)

        # START LOGMAP SESSION (PIPELINE)
        # The pipeline consists of five steps:
        #   1. obtains an initial alignment (via LogMap) for the given task
        #      and identifies mappings that LogMap is highly uncertain of
        #      producing both an initial alignment and the M_ask mapping set
        #   2. consumes the output (M_ask) mappings, these used to construct
        #      prompts for an LLM which may include structural context. The
        #      `prompt_build` step then produces these prompts as output.
        #   3. `consult_oracle` then consumes the output prompts, by passing
        #      them to an LLM. The LLM responds by indicating whether it
        #      judges the alignment in question as correct. The set of LLM
        #      responses produce the output at this step.
        #   4. the LLM oracle output mappings are then consumed by the refine
        #      ment step, which LogMap then integrates into its final alignment.
        #      The final alignment is saved to disk, and is read by an evaluation
        #      process in the next step.
        #   5. Evaluation occurs by comparing the refined alignment to the
        #      reference alignment. Experimental results are saved to disk
        #      and streamed to std::out (and therefore also logged). These
        #      results are then later read by scripts for plotting and results
        #      table construction for analysis.

        info('LogMap-LLM pipeline starting.')
        pipeline_ctx = PipelineContext(cfg, run_paths, logmap)

        # STEP ONE: obtain initial alignment

        align_result = align(pipeline_ctx)

        # STEP TWO: consume uncertain mappings to produce M_ask prompts

        prompt_build_result = prompt_build(pipeline_ctx, align_result)

        # STEP THREE: consult oracle regarding M_ask to obtain mapping adjustments (responses)

        oracle_result = consult_oracle(pipeline_ctx, align_result, prompt_build_result)

        # STEP FOUR: consume oracle LLM responses about M_ask to refine alignment 

        refinement_result = refine_alignment(pipeline_ctx, oracle_result)

        # STEP FIVE: run the evaluation procedure, report and log metrics



        step("starting evaluation procedure", 5)

        step("------------------------------------------", 5)
        step("EVALUATION:", 5) # TODO: convert manual conversion script/s into end-to-end logmap-llm
        step(f"Total refined mappings: {refinement_result.n_refined_mappings}", 5)
        step("------------------------------------------", 5)

        print()
        print()

        success("LogMap-LLM session ending")

    finally:
        sys.stdout = tee_branch_out.original_stdout
        tee_branch_out.close()



if __name__ == "__main__":

    # PARSE CLI ARGUMENTS FOR:
    # 1. --config: filepath/str pointing to pipeline config for this run
    # 2. --reuse-align: override settings in config to force 'reuse' for initial alignment
    # 3. --reuse-prompts: override settings in config to force 'reuse' for prompt generation

    parser = ArgumentParser(
        description="LogMap-LLM: Python-based LogMap LLM extension for OM"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="default_config.toml",
        help="Path to the TOML configuration file (default: default_config.toml)",
    )
    parser.add_argument(
        "--reuse-align",
        action="store_true",
        default=False,
        help="Override config to reuse existing LogMap alignment (i.e., skip step 1)",
    )
    parser.add_argument(
        "--reuse-prompts",
        action="store_true",
        default=False,
        help="Override config to reuse existing prompts (i.e., skip steps 1 and 2); implies --reuse-align",
    )
    args = parser.parse_args()
    main(args)