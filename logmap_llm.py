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
import pandas as pd
import json
import oracle_prompt_building as opb
import oracle_consultation as oc
import developer_prompts as dp
from datetime import datetime, timezone
from dataclasses import dataclass
from argparse import (
    ArgumentParser,
    Namespace,
)
from log_utils import (
    TeeWriter,
    fatal,
    critical,
    warning,
    info,
    step,
    success,
)
from pipeline_utils import (
    PipelinePaths,
)
from pipeline_config import (
    load_and_validate_config,
    inspect_and_mask_api_key,
    parse_config_into_list,
    LogMapLLMConfig
)
from pipeline_types import (
    AlignmentResult,
    PromptBuildResult,
    OracleResult,
    RefinementResult,
)
from logmap_interface import (
    start_jvm,
    LogMapInterface,
)
from constants import (
    PAIRS_SEPARATOR,
    AlignMode,
    PromptBuildMode,
    ConsultMode,
    RefineMode,
)



@dataclass
class PipelineContext:
    """Shared resources created once, used across steps."""
    cfg: LogMapLLMConfig
    run_paths: PipelinePaths
    logmap: LogMapInterface



def align(ctx: PipelineContext) -> AlignmentResult:
    """
    Docstring for align

    :param ctx: Description
    :type ctx: PipelineContext
    :return: Description
    :rtype: AlignmentResult
    """
    import bridging as br

    step("Align ontologies and obtain mappings to ask an Oracle", 1)

    match(ctx.cfg.pipeline.align_ontologies):

        case AlignMode.ALIGN:
            step("Performing fresh initial LogMap alignment", 1)
            ctx.logmap.perform_alignment()
            return AlignmentResult(
                m_ask_df=br.java_mappings_2_python(
                    ctx.logmap.get_mappings_for_llm()
                ),
                mappings=br.java_mappings_2_python(
                    ctx.logmap.get_mappings()
                )
            )

        case AlignMode.REUSE:
            step(f"Loading mappings to ask an oracle from file: {run_paths.logmap_m_ask()}", 1)
            return AlignmentResult(
                m_ask_df=br.load_m_ask_from_file(
                    ctx.run_paths.logmap_m_ask()
                ),
                mappings=pd.read_csv(
                    ctx.run_paths.logmap_mappings(),
                    sep=PAIRS_SEPARATOR,
                    header=None
                )
            )

        case AlignMode.BYPASS:
            warning("Bypassing initial LogMap alignment")
            return AlignmentResult()

        case _:
            fatal(f"config: align_ontologies param not recognised: {ctx.cfg.pipeline.align_ontologies}")



def prompt_build(ctx: PipelineContext, initial_alignment: AlignmentResult) -> PromptBuildResult:
    """
    Docstring for prompt_build

    :param ctx: Description
    :type ctx: PipelineContext
    :param initial_alignment: Description
    :type initial_alignment: AlignmentResult
    :return: Description
    :rtype: PromptBuildResult
    """
    import bridging as br

    step("Build user prompts for oracle consultation", 2)

    match(ctx.cfg.pipeline.build_oracle_prompts):

        case PromptBuildMode.BUILD:
            step("Building fresh oracle user prompts", 2)
            return PromptBuildResult(
                prompts=opb.build_oracle_user_prompts(
                    ctx.cfg.oracle.oracle_user_prompt_template_name,
                    ctx.cfg.alignmentTask.onto_source_filepath,
                    ctx.cfg.alignmentTask.onto_target_filepath,
                    initial_alignment.m_ask_df
                )
            )

        case PromptBuildMode.REUSE:
            step(f"Loading LLM oracle user prompts from file: {str(ctx.run_paths.prompts_json())}", 2)
            with open(ctx.run_paths.prompts_json(), 'r') as fp:
                return PromptBuildResult(
                    prompts=json.load(fp)
                )

        case PromptBuildMode.BYPASS:
            warning(f"Bypassing use of LLM oracle user prompts")
            return PromptBuildResult()

        case _:
            fatal(f"config: build_oracle_prompts param not recognised: {ctx.cfg.pipeline.build_oracle_prompts}")




def consult_oracle(ctx: PipelineContext, initial_alignment: AlignmentResult, prompt_build_result: PromptBuildResult) -> OracleResult:

    """
    Docstring for consult_oracle

    :param ctx: Description
    :type ctx: PipelineContext
    :param initial_alignment: Description
    :type initial_alignment: AlignmentResult
    :param prompt_build_result: Description
    :type prompt_build_result: PromptBuildResult
    :return: Description
    :rtype: OracleResult
    """

    step("Consult Oracle for mappings to ask", 3)

    dev_prompt_template = dp.get_developer_prompt(ctx.cfg.oracle.oracle_dev_prompt_template_name)

    match(ctx.cfg.pipeline.consult_oracle):

        case ConsultMode.CONSULT:
            step(f"Consulting LLM oracle with model: {ctx.cfg.oracle.model_name}", 3)
            oracle_predictions_df, oracle_params_dict = oc.consult_oracle_for_mappings_to_ask(
                m_ask_oracle_user_prompts=prompt_build_result.prompts,
                api_key=ctx.cfg.oracle.openrouter_apikey,
                model_name=ctx.cfg.oracle.model_name,
                max_workers=ctx.cfg.oracle.max_workers,
                m_ask_df=initial_alignment.m_ask_df,
                base_url=ctx.cfg.oracle.base_url,
                enable_thinking=ctx.cfg.oracle.enable_thinking,
                interaction_style=ctx.cfg.oracle.interaction_style,
                developer_prompt_text=dev_prompt_template
            )
            return OracleResult(
                predictions=oracle_predictions_df,
                oracle_params=oracle_params_dict
            )

        case ConsultMode.REUSE:
            step(f'Loading LLM oracle predictions for the mappings_to_ask from file: {ctx.run_paths.predictions_csv()}', 3)
            return OracleResult(
                predictions=pd.read_csv(
                    ctx.run_paths.predictions_csv()
                )
            )

        case ConsultMode.LOCAL:
            step(f'Local oracle prediction .csv file(s) will be loaded from directory: {ctx.cfg.oracle.local_oracle_predictions_dirpath}', 3)
            return OracleResult(
                local_dir=ctx.cfg.oracle.local_oracle_predictions_dirpath,
                predictions=None
            )

        case ConsultMode.BYPASS:
            warning('Bypassing oracle consultations')
            return OracleResult()

        case _:
            fatal(f"config: consult_oracle param not recognised: {ctx.cfg.pipeline.consult_oracle}")



def refine_alignment(ctx: PipelineContext, oracle_result: OracleResult) -> RefinementResult:

    """
    Docstring for refine_alignment

    :param ctx: Description
    :type ctx: PipelineContext
    :param oracle_result: Description
    :type oracle_result: OracleResult
    :return: Description
    :rtype: RefinementResult
    """

    import bridging as br

    step("refine alignment using oracle mapping predictions", 4)

    ctx.logmap.set_output_dir(ctx.run_paths.refined_dir)

    match(ctx.cfg.pipeline.refine_alignment):

        case RefineMode.REFINE:

            if oracle_result.has_predictions:
                step("Refining initial LogMap alignment with LLM Oracle predictions", 4)
                preds_java = br.python_oracle_mapping_predictions_2_java(oracle_result.predictions)
                step(f'Number of mappings predicted True by Oracle given to LogMap: {len(preds_java)}', 4)
                ctx.logmap.refine_alignment(preds_java)
                mappings_java = ctx.logmap.get_mappings()
                step(f'Number of mappings in LogMap refined alignment: {len(mappings_java)}', 4)
                return RefinementResult(
                    refined_mappings=br.java_mappings_2_python(mappings_java)
                )

            elif oracle_result.is_local:
                step("Refining initial LogMap alignment with local Oracle predictions", 4)
                ctx.logmap.refine_alignment(str(oracle_result.local_dir))
                mappings_java = ctx.logmap.get_mappings()
                step(f'Number of mappings in LogMap refined alignment: {len(mappings_java)}', 4)
                return RefinementResult(
                    refined_mappings=br.java_mappings_2_python(mappings_java)
                )

            else:
                critical("Check your config to ensure 'refine_alignment' is set appropriately!")
                critical("The oracle response is empty and no local predictions file is specified.")
                critical("This could result in a problem!")

        case RefineMode.BYPASS:
            warning("Bypassing alignment refinement")
            return RefinementResult()

        case _:
            fatal(f"config: refine_alignment param not recognised: {ctx.cfg.pipeline.refine_alignment}")




def main(args: Namespace):

    expr_run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    #
    # MANAGE (LOAD & VALIDATE) CONFIG
    #

    config_path = args.config

    cfg = load_and_validate_config(
        config_path,
        reuse_align=args.reuse_align,
        reuse_prompts=args.reuse_prompts
    )

    #
    # MANAGE PATH EXPECTATIONS
    #

    run_paths = PipelinePaths.from_config(cfg)

    os.makedirs(run_paths.output_dir, exist_ok=True)
    os.makedirs(run_paths.initial_dir, exist_ok=True)
    os.makedirs(run_paths.refined_dir, exist_ok=True)

    #
    # START LOGGING OUTPUT TO BOTH STD::OUT AND LOG FILE
    #

    tee_branch_out = TeeWriter(str(run_paths.run_log(expr_run_timestamp)), sys.stdout)
    sys.stdout = tee_branch_out

    #
    # PRINT EXP PARAMS (REPLACES LARGE SET OF MANUAL PRINT STATEMENTS)
    #

    flat_config_params: list = parse_config_into_list(cfg.model_dump())
    expr_params_str: str = "\n\nSummary of Experiment Parameters:\n\n"
    for key, value in flat_config_params:
        if key == "oracle.openrouter_apikey":
            value = inspect_and_mask_api_key(value)
        expr_params_str += f"{key}: {value}\n"
    info(expr_params_str)

    info(f"\n\nSummary of File Paths:\n\n{run_paths.summary()}\n\n")

    #
    # INITIALISE LOGMAP
    #

    if cfg.alignmentTask.logmap_parameters_dirpath is not None:
        logmap_dirpath = cfg.alignmentTask.logmap_parameters_dirpath
    else:
        logmap_dirpath = os.path.join(os.getcwd(), 'logmap')
        # ^ should be run from the logmap_llm.py script
        # since we can keep this is the project root dir
        # and logmap should always exist as a sibling dir

    # functional call to jpype (required for LogMap & bridging)
    start_jvm(logmap_dir=logmap_dirpath)

    # allows for mapping LogMap IO between java and python (and vice versa)
    import bridging as br

    # LogMapInterface exposes a simple API for interfacing with LogMap via python
    logmap = LogMapInterface(cfg, logmap_dirpath)
    logmap.set_output_dir(run_paths.initial_dir)

    # Begin LogMap-LLM session dialog with the user

    info('LogMap-LLM session beginning')
    info('Setting up Pipeline Context')

    pipeline_ctx = PipelineContext(cfg, run_paths, logmap)

    #
    # STEP ONE: align ontologies and obtain mappings to ask an Oracle
    #

    align_result = align(pipeline_ctx)

    success(f"Number of mappings within the initial alignment: {align_result.n_mappings}")
    success(f"Number of mappings within M_ask: {align_result.n_m_ask}")

    #
    # STEP TWO: build user prompts for oracle consultation
    #

    prompt_build_result = prompt_build(pipeline_ctx, align_result)

    success(f"Number of LLM oracle user prompts: {prompt_build_result.n_prompts}")

    if cfg.pipeline.build_oracle_prompts == PromptBuildMode.BUILD:
        with open(run_paths.prompts_json(), 'w') as fp:
            json.dump(prompt_build_result.prompts, fp)
        success(f"LLM oracle user prompts saved to file: {str(run_paths.prompts_json())}")

    #
    # STEP THREE: consult oracle for mappings to ask
    #

    oracle_result = consult_oracle(pipeline_ctx, align_result, prompt_build_result)

    if cfg.pipeline.consult_oracle == ConsultMode.CONSULT and oracle_result.predictions is not None:
        oracle_result.predictions.to_csv(run_paths.predictions_csv())
        success(f"Oracle predictions for 'mappings to ask' saved to file: {run_paths.predictions_csv()}")

    if oracle_result.predictions is not None:
        preds = oracle_result.predictions['Oracle_prediction']
        nr_mappings = len(preds)
        nr_errors = sum(preds == 'error')
        nr_completions = nr_mappings - nr_errors
        nr_true = sum(preds == True)
        nr_false = sum(preds == False)
        width = len(str(nr_mappings))
        nr_true = str(nr_true).rjust(width)
        nr_false = str(nr_false).rjust(width)
        nr_errors = str(nr_errors).rjust(width)
        info(f"Number of mappings to ask an Oracle: {nr_mappings}")
        info(f"Number of LLM Oracle consultations : {nr_completions}")
        info(f"Number of mappings predicted True  : {nr_true}")
        info(f"Number of mappings predicted False : {nr_false}")
        info(f"Number of consultation failures    : {nr_errors}\n")

    #
    # STEP FOUR: refine alignment using oracle mapping predictions
    #

    refinement_result = refine_alignment(pipeline_ctx, oracle_result)

    #
    # STEP FIVE: reporting metrics, results, and experimental settings
    #

    step("starting evaluation procedure", 5)

    step("------------------------------------------", 5)
    step("EVALUATION:", 5) # TODO: convert manual conversion script/s into end-to-end logmap-llm
    step(f"Total refined mappings: {refinement_result.n_refined_mappings}", 5)
    step("------------------------------------------", 5)

    print()
    print()

    success("LogMap-LLM session ending")



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