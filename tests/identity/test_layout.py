"""The section 8.2 layout is importable, not merely present on disk.

A directory without an ``__init__.py`` and a module nobody has ever imported
are both fine until the day a later prompt imports them, which is the worst day
to discover a typo in the tree. This walks every declared module once.
"""

from __future__ import annotations

import importlib

import pytest

import architecture_mechanics

PACKAGES = [
    "architecture_mechanics",
    "architecture_mechanics.data",
    "architecture_mechanics.models",
    "architecture_mechanics.instrumentation",
    "architecture_mechanics.metrics",
    "architecture_mechanics.experiments",
    "architecture_mechanics.reporting",
]

# Every module section 8.2 names. Most are empty stubs today; each names the
# prompt that fills it in its docstring.
MODULES = [
    "architecture_mechanics.seeding",
    "architecture_mechanics.device",
    "architecture_mechanics.data.feature_program",
    "architecture_mechanics.data.task_families",
    "architecture_mechanics.data.splits",
    "architecture_mechanics.models.common",
    "architecture_mechanics.models.softmax",
    "architecture_mechanics.models.linear",
    "architecture_mechanics.models.delta_memory",
    "architecture_mechanics.models.receiver_gate",
    "architecture_mechanics.models.depth_router",
    "architecture_mechanics.models.hybrid",
    "architecture_mechanics.instrumentation.hooks",
    "architecture_mechanics.instrumentation.state_capture",
    "architecture_mechanics.instrumentation.interventions",
    "architecture_mechanics.instrumentation.causal_restoration",
    "architecture_mechanics.metrics.capability",
    "architecture_mechanics.metrics.geometry",
    "architecture_mechanics.metrics.mechanism",
    "architecture_mechanics.metrics.statistics",
    "architecture_mechanics.experiments.config",
    "architecture_mechanics.experiments.runner",
    "architecture_mechanics.experiments.manifest",
    "architecture_mechanics.experiments.claim_packet",
    "architecture_mechanics.reporting.tables",
    "architecture_mechanics.reporting.figures",
    "architecture_mechanics.reporting.evidence_bundle",
]


@pytest.mark.parametrize("name", PACKAGES + MODULES)
def test_module_imports_and_is_documented(name: str):
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} has no docstring"


def test_package_declares_a_version():
    assert architecture_mechanics.__version__
