"""Matched small models: the shared trunk and each mixing mechanism.

Importing this package registers every implemented mixing primitive, so
``ModelConfig(arch=...)`` resolves without the caller knowing which module
defines which mechanism. A0 is here; A1 and A2 join it at prompts 11 and 17.
"""

from architecture_mechanics.models import softmax as _softmax  # noqa: F401  (registers "softmax")
