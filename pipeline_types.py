"""

"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd


@dataclass
class AlignmentResult:
    """Produced by Step 1, consumed by Steps 2 and 3."""
    m_ask_df: pd.DataFrame | None = None
    mappings: pd.DataFrame | None = None
    
    @property
    def n_mappings(self) -> int:
        return len(self.mappings) if self.mappings is not None else 0
    
    @property
    def n_m_ask(self) -> int:
        return self.m_ask_df.shape[0] if self.m_ask_df is not None else 0


@dataclass
class PromptBuildResult:
    """Produced by Step 2, consumed by Step 3."""
    prompts: dict | None = None
    
    @property
    def n_prompts(self) -> int:
        return len(self.prompts) if self.prompts is not None else 0


@dataclass
class OracleResult:
    """Produced by Step 3, consumed by Step 4."""
    predictions: pd.DataFrame | None = None
    oracle_params: dict = field(default_factory=dict)
    local_dir: str | Path | None = None

    @property
    def has_predictions(self) -> bool:
        return self.predictions is not None

    @property
    def is_local(self) -> bool:
        return self.local_dir is not None


@dataclass
class RefinementResult:
    """Produced by Step 4, consumed by Step 5."""
    refined_mappings: pd.DataFrame | None = None

    @property
    def n_refined_mappings(self) -> int:
        return self.refined_mappings.shape[0] if self.refined_mappings is not None else 0
    
    @property
    def completed(self) -> bool:
        return self.refined_mappings is not None