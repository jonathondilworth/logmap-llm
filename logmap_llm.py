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
import tomllib
import pandas as pd
import json
import oracle_prompt_building as opb
import oracle_consultation as oc
import developer_prompts as dp

from argparse import ArgumentParser
from pathlib import Path

from constants import (
    PAIRS_SEPARATOR,
    AlignMode,
    PromptBuildMode,
    ConsultMode,
    RefineMode,
)

from pipeline_utils import PipelinePaths
from pydantic import (
    ValidationError
)
from config_schema import (
    validate_config,
    LogMapLLMConfig
)
from log_utils import (
    TeeWriter,
    error,
    critical,
    warning,
    info,
    step,
    success,
)

from typing import Any

# PREP JAVA MODULES & LOAD INTO GLOBAL NS

import jpype
import jpype.imports
from jpype.types import * # type: ignore

########
# FUNCTIONS TO BE MOVED INTO SEPERATE MODULE LATER
########


def start_jvm(logmap_dir: str | Path) -> None:
    """configures classpath and boots jpype JVM for LogMap"""
    logmap_jar = os.path.join(logmap_dir, 'logmap-matcher-4.0.jar')
    logmap_dep = os.path.join(logmap_dir, 'java-dependencies/*')
    jpype.addClassPath(logmap_jar)
    jpype.addClassPath(logmap_dep)
    # perform checks & start JVM
    if jpype.isJVMStarted():
        raise RuntimeError("JVM already running unexpectedly")
    jpype.startJVM(
        "-Xms500M",
        "-Xmx25G",
        "-DentityExpansionLimit=10000000",
        "--add-opens=java.base/java.lang=ALL-UNNAMED"
    )
    if not jpype.isJVMStarted():
        raise RuntimeError("JVM failed to start")



def load_and_validate_config(config_path: str, reuse_align: bool = False,
                             reuse_prompts: bool = False) -> LogMapLLMConfig:
    """
    MANAGE CONFIGURATION with load_and_validate_config:
      1. get config from CLI argument or default to `default_config.toml`
      2. load the config toml as a dict
      3. override any provided CLI flags (--reuse-align, --reuse-prompts)
      4. validate the config agaisnt the pydantic model from `model_schema.py`
      5. handle any exceptions

    :param config_path: Description
    :param reuse_align Description
    :param reuse_prompts Description

    :type config_path: str
    :type reuse_align: bool
    :type reuse_prompts: bool

    :return: Description
    :rtype: LogMapLLMConfig
    """
    if not os.path.isfile(config_path):
        error(f"configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, mode="rb") as fp:
        config = tomllib.load(fp)

    if reuse_align:
        info("--reuse-align set, overriding config (reuse init align)")
        config['pipeline']['align_ontologies'] = 'reuse'

    if reuse_prompts:
        info("--reuse-prompts set, overriding config (reuse prompts+align)")
        config['pipeline']['build_oracle_prompts'] = 'reuse'
        config['pipeline']['align_ontologies'] = 'reuse'

    try:
        cfg = validate_config(config)
    except ValidationError as e:
        error(f"configuration file is invalid: {config_path}")
        for err in e.errors():
            # err["loc"] is a tuple within ValidationError
            # see: https://docs.pydantic.dev/latest/api/pydantic_core/#pydantic_core.ErrorDetails.type
            field = " x ".join(str(loc) for loc in err["loc"])
            error(f" {field}: {err['msg']}")
        sys.exit(1)

    success(f"configuration validated: {config_path}")
    return cfg



def inspect_and_mask_api_key(key: str) -> str:
    if key == 'EMPTY':
        return key
    # else
    return f"{key[:4]} ... {key[-4:]}" if len(key) > 8 else "***"



def parse_config_into_list(config_dict: dict, key_prefix: str = "") -> list[tuple[str, Any]]:
    # base case (when we arrive at a leaf node, i.e., a non-dict)
    if not isinstance(config_dict, dict):
        return [(key_prefix, config_dict)]
    kv_config_pairs = []
    for key, value in config_dict.items():
        extended_key = f"{key_prefix}.{key}" if key_prefix else key
        # recursive call
        kv_config_pairs.extend(parse_config_into_list(value, extended_key))
    return kv_config_pairs




class LogMapInterface:
    """
    python wrapper for the LogMap java interface
    encapsulates LogMap-Java API, important notes:
      - ontology paths require a 'file:' URI prefix
      - parameters directory path must end with os.sep
      - output directory is mutable (i.e., init-align -> refined-align)
    IMPORTANT: this interface will only work if:
        1. JPype has been imported
        2. JVM has started
        3. LogMapLLM_Interface resolves from uk.ac.ox.krr.logmap2
            -> if 1 & 2 then 3 should be fine..
    """

    def __init__(self, cfg: LogMapLLMConfig, logmap_dir: str | Path):
        # Java imports for basic LogMap usage
        from uk.ac.ox.krr.logmap2 import LogMapLLM_Interface  # type: ignore
        # obtain relevant experimental params from the cfg & process for LogMap
        self.task_name = cfg.alignmentTask.task_name
        src_uri = "file:" + cfg.alignmentTask.onto_source_filepath
        tgt_uri = "file:" + cfg.alignmentTask.onto_target_filepath
        # instanciate LogMap via uk.ac.ox.krr.logmap2.LogMapLLM_Interface
        # & specify the neccesary configuration (extd-qestions & params dir)
        self._interface = LogMapLLM_Interface(
            src_uri, tgt_uri, self.task_name
        )
        self._interface.setExtendedQuestions4LLM(
            cfg.alignmentTask.generate_extended_mappings_to_ask_oracle
        )
        self._set_parameters_dir(
            cfg.alignmentTask.logmap_parameters_dirpath, logmap_dir
        )

    def _set_parameters_dir(self, configured_path: str, logmap_dir: str | Path) -> None:
        """
        resolve and set the LogMap parameters directory
        falls back to the LogMap installation directory if no path is configured.
        appends a trailing separator as required by the LogMap Java imlementation
        """
        path = configured_path if configured_path else str(logmap_dir)
        if not path.endswith(os.sep):
            path += os.sep
        self._interface.setPathToLogMapParameters(path)

    def set_output_dir(self, dirpath: str | Path) -> None:
        """
        set the directory LogMap writes alignment output to
        called before step 1 (initial) and step 4 (refined) with:
        --> !! different directories !! <--
        """
        self._interface.setPathForOutputMappings(str(dirpath))

    def perform_alignment(self) -> None:
        self._interface.performAlignment()

    def get_mappings(self):
        return self._interface.getLogMapMappings()

    def get_mappings_for_llm(self):
        return self._interface.getLogMapMappingsForLLM()

    def refine_alignment(self, oracle_predictions) -> None:
        self._interface.performAlignmentWithLocalOracle(oracle_predictions)



class OracleOutcome:
    """result produced @ step 3, consumed by step 4"""
    def __init__(self, predictions: pd.DataFrame | None = None, local_dir: str | Path | None = None):
        self.predictions = predictions
        self.local_dir = local_dir

    @property
    def has_predictions(self) -> bool:
        return self.predictions is not None

    @property
    def is_local(self) -> bool:
        return self.local_dir is not None



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
# PRINT EXP PARAMS (REPLACES LARGE SET OF MANUAL PRINT STATEMENTS)
#

flat_config_params: list = parse_config_into_list(cfg.model_dump())
expr_params_str: str = "\n\nSummary of Experiment Parameters:\n\n"
for key, value in flat_config_params:
    expr_params_str += f"{key}: {value}\n"
info(expr_params_str)



#
# MANAGE PATH EXPECTATIONS
#

run_paths = PipelinePaths.from_config(cfg)

os.makedirs(run_paths.output_dir, exist_ok=True)
os.makedirs(run_paths.initial_dir, exist_ok=True)
os.makedirs(run_paths.refined_dir, exist_ok=True)

run_path_summary = run_paths.summary()

info(f"\n\nSummary of File Paths:\n\n{run_path_summary}\n\n")



#
# INITIALISE LOGMAP
#

# TODO: decide the best way to set the logmap_dirpath
logmap_dirpath = os.path.join(os.getcwd(), 'logmap')

# boot-up JVM ready for LogMap
start_jvm(logmap_dir=logmap_dirpath)

# maps java-2-python & python-2-java
import bridging as br

# LogMapInterface should probably be decoupled from the cfg object
# having a global cfg is almost a pattern in research codebases though
# e.g., ML pipelines, wandb, etc. Probably fine.

logmap = LogMapInterface(cfg, logmap_dirpath)
logmap.set_output_dir(run_paths.initial_dir)

# Begin LogMap-LLM session dialog with the user

info('LogMap-LLM session beginning')



#
# STEP ONE: align ontologies and obtain mappings to ask an Oracle
#

step(
    enumeration=1,
    msg="Align ontologies and obtain mappings to ask an Oracle"
)

match(cfg.pipeline.align_ontologies):

    case AlignMode.ALIGN:
        step("Performing fresh initial LogMap alignment", 1)
        logmap.perform_alignment()
        mappings = logmap.get_mappings()
        step(f"Number of mappings in initial alignment: {len(mappings)}", 1)
        m_ask_java = logmap.get_mappings_for_llm()
        m_ask_df = br.java_mappings_2_python(m_ask_java)
        success("Initial alignment complete")

    case AlignMode.REUSE:
        mappings = pd.read_csv(run_paths.logmap_mappings(), sep=PAIRS_SEPARATOR, header=None)
        step(f"Number of mappings in initial alignment: {len(mappings)}", 1)
        m_ask_df = pd.read_csv(run_paths.logmap_m_ask(), sep=PAIRS_SEPARATOR, header=None)
        step(f"Loading mappings to ask an oracle from file: {run_paths.logmap_m_ask()}", 1)
        m_ask_df.columns = br.get_m_ask_column_names()
        success("Loaded and reusing initial alignment")

    case AlignMode.BYPASS:
        m_ask_df = None
        warning("Bypassing initial LogMap alignment")

    case _:
         error("Something went wrong!")
         error("The alignment step fell through to the default case.")
         raise ValueError(
            f"config: align_ontologies param not recognised: "
            + cfg.pipeline.align_ontologies
        )

if m_ask_df is not None:
    success(f"Number of mappings to ask an Oracle: {len(m_ask_df)}")



#
# STEP TWO: build user prompts for oracle consultation
#

step(
    enumeration=2,
    msg="Build user prompts for oracle consultation"
)

m_ask_oracle_user_prompts = None # deals with pipeline branch paths

match(cfg.pipeline.build_oracle_prompts):

    case PromptBuildMode.BUILD:
        step("Building fresh oracle user prompts", 2)
        m_ask_oracle_user_prompts = opb.build_oracle_user_prompts(cfg.oracle.oracle_user_prompt_template_name,
                                                                  cfg.alignmentTask.onto_source_filepath,
                                                                  cfg.alignmentTask.onto_target_filepath,
                                                                  m_ask_df)
        success("User prompts built!")
        
    case PromptBuildMode.REUSE:
        step(f"Loading LLM oracle user prompts from file: {str(run_paths.prompts_json())}", 2)
        with open(run_paths.prompts_json(), 'r') as fp:
            m_ask_oracle_user_prompts = json.load(fp)
        success(f"User prompts reloaded from: {str(run_paths.prompts_json())}")
        
    case PromptBuildMode.BYPASS:
        m_ask_oracle_user_prompts = None
        warning(f"Bypassing use of LLM oracle user prompts")

    case _:
        error("Something went wrong!")
        error("The prompt building step fell through to the default case.")
        raise ValueError(
            f"config: build_oracle_prompts param not recognised: "
            + cfg.pipeline.build_oracle_prompts
        )

if m_ask_oracle_user_prompts is not None:
    success(f"Number of LLM oracle user prompts: {len(m_ask_oracle_user_prompts)}")

if cfg.pipeline.build_oracle_prompts == PromptBuildMode.BUILD:
    with open(run_paths.prompts_json(), 'w') as fp:
        json.dump(m_ask_oracle_user_prompts, fp)
    success(f"LLM oracle user prompts saved to file: {str(run_paths.prompts_json())}")



#
# STEP THREE: consult oracle for mappings to ask
#

step(
    enumeration=3,
    msg="Consult Oracle for mappings to ask"
)

# empty initialisations (for linting)
oracle_outcome = OracleOutcome()
m_ask_df_ext = None
oracle_params = {}

# JD: resolves dev prompt specified in config from the registry in developer_prompts.py
dev_prompt_template = dp.get_developer_prompt(cfg.oracle.oracle_dev_prompt_template_name)


match(cfg.pipeline.consult_oracle):

    case ConsultMode.CONSULT:
        step(f"Consulting LLM oracle with model: {cfg.oracle.model_name}", 3)
        m_ask_df_ext, oracle_params = oc.consult_oracle_for_mappings_to_ask(
            m_ask_oracle_user_prompts,
            api_key=cfg.oracle.openrouter_apikey,
            model_name=cfg.oracle.model_name,
            max_workers=cfg.oracle.max_workers,
            m_ask_df=m_ask_df,
            base_url=cfg.oracle.base_url,
            enable_thinking=cfg.oracle.enable_thinking,
            interaction_style=cfg.oracle.interaction_style,
            developer_prompt_text=dev_prompt_template
        )
        oracle_outcome = OracleOutcome(predictions=m_ask_df_ext)
        success("Oracle consultation complete!")

    case ConsultMode.REUSE:
        step(f'Loading LLM oracle predictions for the mappings_to_ask from file: {run_paths.predictions_csv()}', 3)
        m_ask_df_ext = pd.read_csv(run_paths.predictions_csv())
        oracle_outcome = OracleOutcome(predictions=m_ask_df_ext)
        success("Loaded oracle predictions (from M_ask)!")

    case ConsultMode.LOCAL:
        step(f'Local oracle prediction .csv file(s) will be loaded from directory: {cfg.oracle.local_oracle_predictions_dirpath}', 3)
        oracle_outcome = OracleOutcome(
            local_dir=cfg.oracle.local_oracle_predictions_dirpath,
            predictions=None
        )
        m_ask_df_ext = None
        success("Loaded oracle predictions (from CSV)!")

    case ConsultMode.BYPASS:
        warning('Bypassing oracle consultations')
        oracle_outcome = OracleOutcome(predictions=None)
        m_ask_df_ext = None

    case _:
        error("Something went wrong!")
        error("The oracle consultation step fell through to the default case.")
        raise ValueError(
            f"config: consult_oracle param not recognised: "
            + cfg.pipeline.consult_oracle
        )

if cfg.pipeline.consult_oracle == ConsultMode.CONSULT and m_ask_df_ext is not None:
    # save the extended m_ask dataframe (that contains the LLM Oracle predictions)
    m_ask_df_ext.to_csv(run_paths.predictions_csv())
    success(f"Oracle predictions for 'mappings to ask' saved to file: {run_paths.predictions_csv()}")

if m_ask_df_ext is not None:
    preds = m_ask_df_ext['Oracle_prediction']
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
# STEP FOUR: refine alignment using racle mapping predictions
#

step(
    enumeration=4,
    msg="refine alignment using oracle mapping predictions"
)

logmap.set_output_dir(run_paths.refined_dir)

match(cfg.pipeline.refine_alignment):

    case RefineMode.REFINE:
        if oracle_outcome.has_predictions:
            step("Refining initial LogMap alignment with LLM Oracle predictions", 4)
            preds_java = br.python_oracle_mapping_predictions_2_java(oracle_outcome.predictions)
            step(f'Number of mappings predicted True by Oracle given to LogMap: {len(preds_java)}', 4)
            logmap.refine_alignment(preds_java)
            mappings_java = logmap.get_mappings()
            step(f'Number of mappings in LogMap refined alignment: {len(mappings_java)}', 4)
            success("Alignment refinement complete")

        elif oracle_outcome.is_local:
            step("Refining initial LogMap alignment with local Oracle predictions", 4)
            logmap.refine_alignment(str(oracle_outcome.local_dir))
            mappings_java = logmap.get_mappings()
            step(f'Number of mappings in LogMap refined alignment: {len(mappings_java)}', 4)
            success("Alignment refinement complete")

        else:
            critical("Check your config to ensure 'refine_alignment' is set appropriately!")
            critical("The oracle response is empty and no local predictions file is specified.")
            critical("This could result in a problem!")

    case RefineMode.BYPASS:
        warning("Bypassing alignment refinement")

    case _:
        error("Something went wrong!")
        error("The refine alignment step fell through to the default case.")
        raise ValueError(
            f"config: refine_alignment param not recognised: "
            + cfg.pipeline.refine_alignment
        )


#
# STEP FIVE: reporting metrics, results, and experimental settings
#

step(
    enumeration=5,
    msg="reporting metrics, results, and experimental settings"
)

step(f"Experimental Settings:", 5)
step(f"[PARAM] Model Name: {cfg.oracle.model_name}", 5)
step(f"[PARAM] Interaction Style: {cfg.oracle.interaction_style}", 5)
step(f"[PARAM] M_ask (LLM) Temp: {cfg.oracle.temperature}", 5)
step(f"[PARAM] M_ask (LLM) Top-p: {cfg.oracle.top_p}", 5)
step(f"[PARAM] M_ask (LLM) Reasoning Effort: {cfg.oracle.reasoning_effort}", 5)
step(f"[PARAM] Max Tokens: {cfg.oracle.max_completion_tokens}", 5)
step(f"[PARAM] Thinking Enabled: {cfg.oracle.enable_thinking}", 5)
step(f"[PARAM] Max Worker Threads: {cfg.oracle.max_workers}", 5)
step(f"[PARAM] Base URL: {cfg.oracle.base_url}", 5)
step(f"[PARAM] Developer Prompt Specified: {cfg.oracle.oracle_dev_prompt_template_name}", 5)
step(f"[PARAM] User Prompt Template Specified: {cfg.oracle.oracle_user_prompt_template_name}", 5)
step("------------------------------------------", 5)
step(f'Alignment task name: {cfg.alignmentTask.task_name}', 5)
step(f"Source ontology: {cfg.alignmentTask.onto_source_filepath}", 5)
step(f"Target ontology: {cfg.alignmentTask.onto_target_filepath}", 5)
step("------------------------------------------", 5)
step("EVALUATION:", 5) # TODO: convert manual conversion script/s into end-to-end logmap-llm
step("------------------------------------------", 5)

print()
print()

success("LogMap-LLM session ending")

