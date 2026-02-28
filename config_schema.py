"""

"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional


class AlignmentTaskConfig(BaseModel):
    """configuration for the alignment task"""
    task_name: str
    onto_source_filepath: str
    onto_target_filepath: str
    generate_extended_mappings_to_ask_oracle: bool = False
    logmap_parameters_dirpath: str = ""


class OracleConfig(BaseModel):
    """configuration for the oracle"""
    openrouter_apikey: str = ""
    model_name: str
    oracle_dev_prompt_template_name: str = "class_equivalence"
    oracle_user_prompt_template_name: str
    local_oracle_predictions_dirpath: str = ""
    """Optional LLM parameters:"""
    base_url: Optional[str] = None
    enable_thinking: Optional[bool] = None
    interaction_style: Optional[Literal["auto", "openrouter", "vllm"]] = None
    max_completion_tokens: Optional[int] = None
    failure_tolerance: Optional[int] = None
    max_workers: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    reasoning_effort: Optional[str] = None
    """Few-shot parameters:"""
    few_shot_k: int = 0
    few_shot_seed: int = 42
    few_shot_negative_strategy: str = "hard"


class OutputsConfig(BaseModel):
    """configuration for output directory paths"""
    logmapllm_output_dirpath: str
    logmap_initial_alignment_output_dirpath: str
    logmap_refined_alignment_output_dirpath: str


class PipelineConfig(BaseModel):
    """
    configuration for pipeline step modes
    Each step can be set to control whether it runs fresh, 
    reuses previous results, or is bypassed entirely.
    """
    align_ontologies: Literal["align", "reuse", "bypass"] = "align"
    build_oracle_prompts: Literal["build", "reuse", "bypass"] = "build"
    consult_oracle: Literal["consult", "reuse", "local", "bypass"] = "consult"
    refine_alignment: Literal["refine", "bypass"] = "refine"


class EvaluationConfig(BaseModel):
    """configuration for optional evaluation step"""
    evaluate: bool = False
    reference_alignment_path: Optional[str] = None
    train_alignment_path: Optional[str] = None
    test_cands_path: Optional[str] = None
    metrics: list[str] | str = Field(default_factory=lambda: ["global", "oracle"])
    force_custom_eval: bool = False
    jvm_memory: str = "8g"

    @model_validator(mode="after")
    def normalise_metrics(self):
        """accept comma-separated string or list for metrics"""
        if isinstance(self.metrics, str):
            self.metrics = [m.strip() for m in self.metrics.split(",")]
        return self


class LogMapLLMConfig(BaseModel):
    """composes configuration schema @ top-level for LogMap-LLM
    - Validates the entire TOML config structure on construction
    - Missing required fields or unknown pipeline modes produce msgs
    """
    alignmentTask: AlignmentTaskConfig
    oracle:     OracleConfig
    outputs:    OutputsConfig
    pipeline:   PipelineConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    # 
    class Config:
        """allow extra fields at the top level for forward compatibility"""
        extra = "ignore"


def validate_config(config_dict: dict) -> LogMapLLMConfig:
    """
    validates the config dict and returns a typed config obj

    Parameters
    ----------
    config_dict : dict
        the raw dictionary from tomllib.load()

    Returns
    -------
    LogMapLLMConfig
        a validated (& typed) configuration object

    Raises
    ------
    pydantic.ValidationError
        if the config is missing required fields, has wrong types,
        or contains invalid enum values for the pipeline modes
    """
    return LogMapLLMConfig.model_validate(config_dict)