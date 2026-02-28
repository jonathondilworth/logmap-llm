"""
Docstring for logmap-llm.pipeline_config
"""

import sys
import os
import jpype
import tomllib
from typing import Any
from log_utils import (
    error,
    info,
    success,
)
from config_schema import (
    validate_config,
    LogMapLLMConfig
)
from pydantic import (
    ValidationError
)


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