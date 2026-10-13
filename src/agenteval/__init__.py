"""agenteval: statistically rigorous evaluation for tool-using LLM agents."""

from .vcomp import (
    CellIndex,
    VarianceComponents,
    moment_estimator,
    reml_estimator,
    mean_squares,
    component_standard_errors,
    icc_oneway,
    icc_anova_decomposition,
)
from .sim import (
    FieldSpec,
    SuiteSpec,
    AgentSpec,
    JudgeSpec,
    simulate_difference_field,
    simulate_suite,
)

__version__ = "0.1.0"

__all__ = [
    "CellIndex",
    "VarianceComponents",
    "moment_estimator",
    "reml_estimator",
    "mean_squares",
    "component_standard_errors",
    "icc_oneway",
    "icc_anova_decomposition",
    "FieldSpec",
    "SuiteSpec",
    "AgentSpec",
    "JudgeSpec",
    "simulate_difference_field",
    "simulate_suite",
    "__version__",
]
