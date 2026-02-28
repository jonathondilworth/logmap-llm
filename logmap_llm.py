# Python script driver for LogMap-LLM
# 
# This Python script driver for LogMap-LLM provides a command-line,
# non-interactive LogMap-LLM user experience. You launch LogMap-LLM
# at the command line and it does everything for you, writing its
# output to the console.

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
# MANAGE CONFIG
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

run_paths.summary()

#
# PRINT EXPERIMENTAL SETTINGS
#

print(f"task name: {cfg.alignmentTask.task_name}")
print(f"onto source: {cfg.alignmentTask.onto_source_filepath}")
print(f"onto target: {cfg.alignmentTask.onto_target_filepath}")
print(f"extended mappings_to_ask: {cfg.alignmentTask.generate_extended_mappings_to_ask_oracle}")
print(f"logmap_parameters_dirpath: {cfg.alignmentTask.logmap_parameters_dirpath}")
print()
print(f"openrouter apikey: {inspect_and_mask_api_key(cfg.oracle.openrouter_apikey)}")
print(f"openrouter LLM model name: {cfg.oracle.model_name}")
print(f"oracle dev prompt template: {cfg.oracle.oracle_dev_prompt_template_name}")
print(f"oracle user prompt template: {cfg.oracle.oracle_user_prompt_template_name}")
print()
print(f"logmapllm output dirpath: {cfg.outputs.logmapllm_output_dirpath}")
print(f"logmap initial alignment output dirpath: {cfg.outputs.logmap_initial_alignment_output_dirpath}")
print(f"logmap refined alignment output dirpath: {cfg.outputs.logmap_refined_alignment_output_dirpath}")
print()
print(f"align ontologies: {cfg.pipeline.align_ontologies}")
print(f"build oracle prompts: {cfg.pipeline.build_oracle_prompts}")
print(f"consult oracle: {cfg.pipeline.consult_oracle}")
print(f"refine alignment: {cfg.pipeline.refine_alignment}")

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

print('LogMap-LLM session beginning')


# - - - - - - - - - - - - - - STEP ONE - - - - - - - - - - - - - - -
print()
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print('Step 1: Align ontologies and obtain mappings to ask an Oracle')
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print()


if cfg.pipeline.align_ontologies == AlignMode.ALIGN:

    # perform an initial alignment so we can get a fresh m_ask
    print("Performing fresh initial LogMap alignment ...")
    print()
    logmap.perform_alignment()
    print("Initial alignment complete")
    mappings = logmap.get_mappings()
    print()
    print(f'Number of mappings in initial alignment: {len(mappings)}')
    m_ask_java = logmap.get_mappings_for_llm()
    m_ask_df = br.java_mappings_2_python(m_ask_java)

elif cfg.pipeline.align_ontologies == AlignMode.REUSE:

    # Reusing existing initial LogMap alignment ...
    mappings = pd.read_csv(run_paths.logmap_mappings(), sep=PAIRS_SEPARATOR, header=None)
    print(f'Number of mappings in initial alignment: {len(mappings)}')

    print('Loading mappings to ask an Oracle from file:')
    m_ask_df = pd.read_csv(run_paths.logmap_m_ask(), sep=PAIRS_SEPARATOR, header=None)
    m_ask_df.columns = br.get_m_ask_column_names()

elif cfg.pipeline.align_ontologies == AlignMode.BYPASS:

    print('Bypassing initial LogMap alignment')
    m_ask_df = None

else:
    raise ValueError(f"Value for align_ontologies not recognised: {cfg.pipeline.align_ontologies}")

if m_ask_df is not None:
    print()
    print(f"Number of mappings to ask an Oracle: {len(m_ask_df)}")




# - - - - - - - - - - - - - - STEP TWO - - - - - - - - - - - - - - -
print()
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print('Step 2: Build user prompts for mappings to ask an LLM Oracle')
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print()

oupt_name = cfg.oracle.oracle_user_prompt_template_name

if cfg.pipeline.build_oracle_prompts == PromptBuildMode.BUILD:
    print('Building fresh Oracle user prompts ...')
    print()
    m_ask_oracle_user_prompts = opb.build_oracle_user_prompts(oupt_name,
                                                              cfg.alignmentTask.onto_source_filepath,
                                                              cfg.alignmentTask.onto_target_filepath,
                                                              m_ask_df)

elif cfg.pipeline.build_oracle_prompts == PromptBuildMode.REUSE:

    # Reusing existing LLM Oracle user prompts ...created previously and saved in a file on disk
    print(f'Loading LLM Oracle user prompts from file: {str(run_paths.prompts_json())}')
    with open(run_paths.prompts_json(), 'r') as fp:
        m_ask_oracle_user_prompts = json.load(fp)

elif cfg.pipeline.build_oracle_prompts == PromptBuildMode.BYPASS:
    print('Bypassing use of LLM Oracle user prompts')
    m_ask_oracle_user_prompts = None

else:
    raise ValueError(f"Value for build_oracle_prompts not recognised: {cfg.pipeline.build_oracle_prompts}")

if m_ask_oracle_user_prompts is not None:
    print()
    print(f"Number of LLM Oracle user prompts obtained: {len(m_ask_oracle_user_prompts)}")
    print()

if cfg.pipeline.build_oracle_prompts == PromptBuildMode.BUILD:
    # save the newly built oracle user prompts to a .json file so they can be reused
    print(f'LLM Oracle user prompts saved to file: {str(run_paths.prompts_json())}')
    with open(run_paths.prompts_json(), 'w') as fp:
        json.dump(m_ask_oracle_user_prompts, fp)



# - - - - - - - - - - - - STEP THREE - - - - - - - - - - - -
print()
print('- - - - - - - - - - - - - - - - - - - - - - - - - - -')
print("Step 3: Consult Oracle for mappings to ask")
print('- - - - - - - - - - - - - - - - - - - - - - - - - - -')
print()

# empty initialisations (for linting)
oracle_outcome = OracleOutcome()
m_ask_df_ext = None
oracle_params = {}

# JD: resolves dev prompt specified in config from the registry in developer_prompts.py
dev_prompt_template = dp.get_developer_prompt(cfg.oracle.oracle_dev_prompt_template_name)


if cfg.pipeline.consult_oracle == ConsultMode.CONSULT:

    print(f'Consulting LLM Oracle {cfg.oracle.model_name}')
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

elif cfg.pipeline.consult_oracle == ConsultMode.REUSE:

    # Reusing existing LLM Oracle predictions created previously and saved in a file on disk
    print(f'Loading LLM Oracle predictions for the mappings_to_ask from file: {run_paths.predictions_csv()}')
    m_ask_df_ext = pd.read_csv(run_paths.predictions_csv())
    oracle_outcome = OracleOutcome(predictions=m_ask_df_ext)

elif cfg.pipeline.consult_oracle == ConsultMode.LOCAL:

    print(f'Local Oracle prediction .csv file(s) will be loaded from directory: {cfg.oracle.local_oracle_predictions_dirpath}')
    oracle_outcome = OracleOutcome(
        local_dir=cfg.oracle.local_oracle_predictions_dirpath,
        predictions=None
    )
    m_ask_df_ext = None

elif cfg.pipeline.consult_oracle == ConsultMode.BYPASS:

    print('Bypassing Oracle consultations')
    oracle_outcome = OracleOutcome(predictions=None)
    m_ask_df_ext = None

else:
    raise ValueError(f"Value for consult_oracle not recognised: {cfg.pipeline.consult_oracle}")


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
    print()
    print(f"Number of mappings to ask an Oracle: {nr_mappings}")
    print(f"Number of LLM Oracle consultations : {nr_completions}")
    print(f"Number of mappings predicted True  : {nr_true}")
    print(f"Number of mappings predicted False : {nr_false}")
    print(f"Number of consultation failures    : {nr_errors}")
    print()

if cfg.pipeline.consult_oracle == ConsultMode.CONSULT and m_ask_df_ext is not None:

    # save the extended m_ask dataframe (that contains the LLM Oracle predictions)
    print(f"Oracle predictions for 'mappings to ask' saved to file: {run_paths.predictions_csv()}")
    m_ask_df_ext.to_csv(run_paths.predictions_csv())


# - - - - - - - - - - - - - - - STEP FOUR - - - - - - - - - - - - - - -
print()
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print('Step 4: Refine alignment using Oracle mapping predictions')
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print()

logmap.set_output_dir(run_paths.refined_dir)

if cfg.pipeline.refine_alignment == RefineMode.REFINE:

    if oracle_outcome.has_predictions:
        info("Refining initial LogMap alignment with LLM Oracle predictions ...")
        preds_java = br.python_oracle_mapping_predictions_2_java(oracle_outcome.predictions)
        info(f'Number of mappings predicted True by Oracle given to LogMap: {len(preds_java)}')
        logmap.refine_alignment(preds_java)
        success("Alignment refinement complete")
        mappings_java = logmap.get_mappings()
        info(f'Number of mappings in LogMap refined alignment: {len(mappings_java)}')

    elif oracle_outcome.is_local:
        info("Refining initial LogMap alignment with local Oracle predictions ...")
        logmap.refine_alignment(str(oracle_outcome.local_dir))
        success("Alignment complete")
        mappings_java = logmap.get_mappings()
        info(f'Number of mappings in LogMap refined alignment: {len(mappings_java)}')

    else:
        warning('Step 4 bypassed due to Oracle consultation failures in Step 3')

elif cfg.pipeline.refine_alignment == RefineMode.BYPASS:
    info('Bypassing alignment refinement')

else:
    raise ValueError(f"Value for refine_alignment not recognised: {cfg.pipeline.refine_alignment}")


# - - - - - - - - - - - - - - - STEP FIVE - - - - - - - - - - - - - -
print()
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print('Step 5: Reporting Metrics & Experimental Settings')
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print()

print(f"Experimental Settings:")

print(f"[PARAM] Model Name: {cfg.oracle.model_name}")
print(f"[PARAM] Interaction Style: {cfg.oracle.interaction_style}")
print(f"[PARAM] M_ask (LLM) Temp: {cfg.oracle.temperature}")
print(f"[PARAM] M_ask (LLM) Top-p: {cfg.oracle.top_p}")
print(f"[PARAM] M_ask (LLM) Reasoning Effort: {cfg.oracle.reasoning_effort}")
print(f"[PARAM] Max Tokens: {cfg.oracle.max_completion_tokens}")
print(f"[PARAM] Thinking Enabled: {cfg.oracle.enable_thinking}")
print(f"[PARAM] Max Worker Threads: {cfg.oracle.max_workers}")
print(f"[PARAM] Base URL: {cfg.oracle.base_url}")
print(f"[PARAM] Developer Prompt Specified: {cfg.oracle.oracle_dev_prompt_template_name}")
print(f"[PARAM] User Prompt Template Specified: {cfg.oracle.oracle_user_prompt_template_name}")
print("------------------------------------------")
print(f'Alignment task name: {cfg.alignmentTask.task_name}')
print(f"Source ontology: {cfg.alignmentTask.onto_source_filepath}")
print(f"Target ontology: {cfg.alignmentTask.onto_target_filepath}")
print("------------------------------------------")
print("EVALUATION:") # TODO: convert manual conversion script/s into end-to-end logmap-llm
print("------------------------------------------")

# %%

print()
print('LogMap-LLM session ending')
print()
