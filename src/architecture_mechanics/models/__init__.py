"""Matched small models: the shared trunk and each mixing mechanism.

Importing this package registers every implemented mixing primitive, so
``ModelConfig(arch=...)`` resolves without the caller knowing which module
defines which mechanism. A0 and A1 are here; A2 joins them at prompt 17.
"""

from architecture_mechanics.models import linear as _linear  # noqa: F401  (registers "linear")
from architecture_mechanics.models import softmax as _softmax  # noqa: F401  (registers "softmax")
