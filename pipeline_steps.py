"""
Docstring for logmap-llm.pipeline_steps
"""

from __future__ import annotations

import pandas as pd
import json
import oracle_prompt_building as opb
import oracle_consultation as oc
import developer_prompts as dp
from dataclasses import dataclass
from log_utils import (
    fatal,
    critical,
    warning,
    step,
    success,
)
from pipeline_utils import (
    PipelinePaths,
)
from pipeline_config import (
    LogMapLLMConfig
)
from pipeline_types import (
    AlignmentResult,
    PromptBuildResult,
    OracleResult,
    RefinementResult,
)
from logmap_interface import (
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
    """
    Docstring for PipelineContext
    """
    cfg: LogMapLLMConfig
    run_paths: PipelinePaths
    logmap: LogMapInterface



def align(ctx: PipelineContext, this_step: int = 1) -> AlignmentResult:
    """
    Docstring for align

    `import bridging as br` allows for mapping LogMap IO between java and python (and vice versa)

    :param ctx: Description
    :type ctx: PipelineContext
    :return: Description
    :rtype: AlignmentResult
    """
    import bridging as br

    step("Align ontologies and obtain mappings to ask an Oracle", this_step)

    match(ctx.cfg.pipeline.align_ontologies):

        case AlignMode.ALIGN:
            step("Performing fresh initial LogMap alignment", this_step)
            ctx.logmap.perform_alignment()
            result = AlignmentResult(
                m_ask_df=br.java_mappings_2_python(
                    ctx.logmap.get_mappings_for_llm()
                ),
                mappings=br.java_mappings_2_python(
                    ctx.logmap.get_mappings()
                )
            )

        case AlignMode.REUSE:
            step(f"Loading mappings to ask an oracle from file: {ctx.run_paths.logmap_m_ask()}", this_step)
            result = AlignmentResult(
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
            result = AlignmentResult()

        case _:
            fatal(f"config: align_ontologies param not recognised: {ctx.cfg.pipeline.align_ontologies}")

    if result.n_mappings > 0:
        success(f"Number of mappings within the initial alignment: {result.n_mappings}")
        success(f"Number of mappings within M_ask: {result.n_m_ask}")

    return result



def prompt_build(ctx: PipelineContext, initial_alignment: AlignmentResult, this_step: int = 2) -> PromptBuildResult:
    """
    Docstring for prompt_build

    :param ctx: Description
    :type ctx: PipelineContext
    :param initial_alignment: Description
    :type initial_alignment: AlignmentResult
    :return: Description
    :rtype: PromptBuildResult
    """

    step("Build user prompts for oracle consultation", this_step)

    match(ctx.cfg.pipeline.build_oracle_prompts):

        case PromptBuildMode.BUILD:
            step("Building fresh oracle user prompts", this_step)
            result = PromptBuildResult(
                prompts=opb.build_oracle_user_prompts(
                    ctx.cfg.oracle.oracle_user_prompt_template_name,
                    ctx.cfg.alignmentTask.onto_source_filepath,
                    ctx.cfg.alignmentTask.onto_target_filepath,
                    initial_alignment.m_ask_df
                )
            )
            with open(ctx.run_paths.prompts_json(), 'w') as fp:
                json.dump(result.prompts, fp)
            success(f"LLM oracle user prompts saved to file: {str(ctx.run_paths.prompts_json())}")

        case PromptBuildMode.REUSE:
            step(f"Loading LLM oracle user prompts from file: {str(ctx.run_paths.prompts_json())}", this_step)
            with open(ctx.run_paths.prompts_json(), 'r') as fp:
                result = PromptBuildResult(
                    prompts=json.load(fp)
                )

        case PromptBuildMode.BYPASS:
            warning(f"Bypassing use of LLM oracle user prompts")
            result = PromptBuildResult()

        case _:
            fatal(f"config: build_oracle_prompts param not recognised: {ctx.cfg.pipeline.build_oracle_prompts}")

    if result.n_prompts > 0:
        success(f"Number of LLM oracle user prompts: {result.n_prompts}")
    
    return result

        



def consult_oracle(ctx: PipelineContext, initial_alignment: AlignmentResult, prompt_build_result: PromptBuildResult, this_step: int = 3) -> OracleResult:

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

    step("Consult Oracle for mappings to ask", this_step)

    dev_prompt_template = dp.get_developer_prompt(ctx.cfg.oracle.oracle_dev_prompt_template_name)

    match(ctx.cfg.pipeline.consult_oracle):

        case ConsultMode.CONSULT:
            step(f"Consulting LLM oracle with model: {ctx.cfg.oracle.model_name}", this_step)
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
            result = OracleResult(
                predictions=oracle_predictions_df,
                oracle_params=oracle_params_dict
            )
            if result.predictions is not None:
                result.predictions.to_csv(ctx.run_paths.predictions_csv())
                success(f"Oracle predictions for 'mappings to ask' saved to file: {ctx.run_paths.predictions_csv()}")

        case ConsultMode.REUSE:
            step(f'Loading LLM oracle predictions for the mappings_to_ask from file: {ctx.run_paths.predictions_csv()}', this_step)
            result = OracleResult(
                predictions=pd.read_csv(
                    ctx.run_paths.predictions_csv()
                )
            )

        case ConsultMode.LOCAL:
            step(f'Local oracle prediction .csv file(s) will be loaded from directory: {ctx.cfg.oracle.local_oracle_predictions_dirpath}', this_step)
            result = OracleResult(
                local_dir=ctx.cfg.oracle.local_oracle_predictions_dirpath,
                predictions=None
            )

        case ConsultMode.BYPASS:
            warning('Bypassing oracle consultations')
            result = OracleResult()

        case _:
            fatal(f"config: consult_oracle param not recognised: {ctx.cfg.pipeline.consult_oracle}")

    if result.prediction_summary() is not None:
        success(f"\n\n{result.prediction_summary()}")
    else:
        warning("There is NO ORACLE PREDICTION SUMMARY available")

    return result



def refine_alignment(ctx: PipelineContext, oracle_result: OracleResult, this_step: int = 4) -> RefinementResult:

    """
    Docstring for refine_alignment

    `import bridging as br` allows for mapping LogMap IO between java and python (and vice versa)

    :param ctx: Description
    :type ctx: PipelineContext
    :param oracle_result: Description
    :type oracle_result: OracleResult
    :return: Description
    :rtype: RefinementResult
    """

    import bridging as br

    step("refine alignment using oracle mapping predictions", this_step)

    ctx.logmap.set_output_dir(ctx.run_paths.refined_dir)

    match(ctx.cfg.pipeline.refine_alignment):

        case RefineMode.REFINE:

            if oracle_result.has_predictions:
                step("Refining initial LogMap alignment with LLM Oracle predictions", this_step)
                preds_java = br.python_oracle_mapping_predictions_2_java(oracle_result.predictions)
                step(f'Number of mappings predicted True by Oracle given to LogMap: {len(preds_java)}', this_step)
                ctx.logmap.refine_alignment(preds_java)
                mappings_java = ctx.logmap.get_mappings()
                step(f'Number of mappings in LogMap refined alignment: {len(mappings_java)}', this_step)
                result = RefinementResult(
                    refined_mappings=br.java_mappings_2_python(mappings_java)
                )

            elif oracle_result.is_local:
                step("Refining initial LogMap alignment with local Oracle predictions", this_step)
                ctx.logmap.refine_alignment(str(oracle_result.local_dir))
                mappings_java = ctx.logmap.get_mappings()
                step(f'Number of mappings in LogMap refined alignment: {len(mappings_java)}', this_step)
                result = RefinementResult(
                    refined_mappings=br.java_mappings_2_python(mappings_java)
                )

            else:
                critical("Check your config to ensure 'refine_alignment' is set appropriately!")
                critical("The oracle response is empty and no local predictions file is specified.")
                critical("This could result in a problem!")
                # JD: probably an IO Error is most appropriate here?
                fatal(f"Oracle payload passed to `refine_alignment` is either malformed or is empty.", IOError)

        case RefineMode.BYPASS:
            warning("Bypassing alignment refinement")
            result = RefinementResult()

        case _:
            fatal(f"config: refine_alignment param not recognised: {ctx.cfg.pipeline.refine_alignment}")

    return result