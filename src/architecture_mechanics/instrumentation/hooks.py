"""Named hook sites: how a mechanism exposes its internals without knowing why.

Established by prompt 04 (A0) and extended by prompts 11 (A1), 17 (A2), and 19
(interventions). The contract is deliberately tiny, because three architectures
and seven intervention families all have to reach through it.

A *site* is one line inside a mechanism's forward pass::

    weights = hooks.site("weights", weights)

That is the whole interface. The mechanism names the tensor; it does not decide
whether anyone is listening, what capture means, or what an intervention would
replace it with. The scaffold that calls the mechanism does not know what
``weights`` means either — it only pushes a scope so that two layers' sites do
not collide.

Two properties matter enough to be structural rather than promised:

**Capture cannot alter the forward pass.** :class:`HookContext` is the base
class and it is also the no-op: its :meth:`~HookContext.site` returns its
argument unchanged. :class:`CaptureContext` overrides it to record and then
still returns the same object. There is no code path in which capturing
substitutes a tensor, so §8.5's hook-no-op-equivalence test is checking an
invariant the type system already makes hard to break rather than a habit.

**Replacement is opt-in, per-site, and visible.** A transform is only applied
where a caller registered one by name, and the replaced tensor is checked
against the original's shape and dtype before it is handed back. Prompt 19
builds its intervention families on that, and gets shape safety for free.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from types import MappingProxyType

import torch

SITE_SEPARATOR = "."
"""Scopes join with a dot, so a fully qualified site reads
``layers.0.mix.weights``. Local names inside a mechanism never contain it."""


class HookSiteError(RuntimeError):
    """A transform returned something that cannot stand in for the original."""


class HookContext:
    """The null context, and the base class every richer context extends.

    Passing this costs one Python call per site and nothing else: no recording,
    no dict lookup, no scope bookkeeping. Training loops use it, which is why
    instrumentation adds no cost to a run that is not being instrumented.
    """

    __slots__ = ()

    enabled: bool = False

    def site(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Declare a named tensor. The null context returns it untouched."""
        return tensor

    @contextmanager
    def scope(self, name: str):
        """Qualify sites declared inside the block. A no-op here."""
        yield self

    def qualified(self, name: str) -> str:
        return name


NO_HOOKS = HookContext()
"""Shared singleton for uninstrumented forward passes."""


class CaptureContext(HookContext):
    """Records named tensors, and optionally substitutes them.

    Args:
        capture: site names to record. Fully qualified (``layers.0.mix.weights``)
            or a bare local name, which records every layer that declares it
            under its own qualified key.
        capture_all: record every site. Convenient for the R0 hook test, which
            must register *every* site rather than a chosen few.
        transforms: qualified-or-local site name to a callable applied to the
            tensor. The return value must match the original's shape, dtype, and
            device; it replaces the tensor in the forward pass.
        detach: store ``tensor.detach().clone()`` rather than the tensor itself.
            Cloning costs a copy of tiny activations and buys immunity to any
            later in-place write; ``detach=False`` keeps the live tensor for
            callers that need its grad.
    """

    __slots__ = ("_capture", "_capture_all", "_detach", "_records", "_scopes", "_transforms", "_visited")

    enabled = True

    def __init__(
        self,
        capture: Iterable[str] = (),
        *,
        capture_all: bool = False,
        transforms: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        detach: bool = True,
    ) -> None:
        self._capture = frozenset(capture)
        self._capture_all = bool(capture_all)
        self._transforms = dict(transforms or {})
        self._detach = bool(detach)
        self._scopes: list[str] = []
        self._records: dict[str, torch.Tensor] = {}
        self._visited: list[str] = []

    # -- the site interface ------------------------------------------------ #

    def qualified(self, name: str) -> str:
        return SITE_SEPARATOR.join((*self._scopes, name))

    def site(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        full = self.qualified(name)
        self._visited.append(full)

        if self._capture_all or full in self._capture or name in self._capture:
            self._records[full] = tensor.detach().clone() if self._detach else tensor

        transform = self._transforms.get(full, self._transforms.get(name))
        if transform is None:
            return tensor
        replacement = transform(tensor)
        _check_substitutable(full, tensor, replacement)
        return replacement

    @contextmanager
    def scope(self, name: str):
        self._scopes.append(name)
        try:
            yield self
        finally:
            self._scopes.pop()

    # -- what was seen ----------------------------------------------------- #

    @property
    def captures(self) -> Mapping[str, torch.Tensor]:
        """Recorded tensors, keyed by fully qualified site name."""
        return MappingProxyType(self._records)

    @property
    def visited(self) -> tuple[str, ...]:
        """Every site declared during the forward pass, in order.

        This is what lets a test compare a mechanism's *declared* hook sites
        against the ones its forward pass actually reaches. A site that exists
        in the docstring and not in the code is the failure mode prompt 19 would
        discover at the worst possible moment.
        """
        return tuple(self._visited)

    def reset(self) -> None:
        self._records.clear()
        self._visited.clear()


def _check_substitutable(name: str, original: torch.Tensor, replacement: object) -> None:
    if not isinstance(replacement, torch.Tensor):
        raise HookSiteError(f"transform at {name!r} returned {type(replacement).__name__}, not a tensor")
    if replacement.shape != original.shape:
        raise HookSiteError(
            f"transform at {name!r} returned shape {tuple(replacement.shape)}, "
            f"expected {tuple(original.shape)}"
        )
    if replacement.dtype != original.dtype:
        raise HookSiteError(
            f"transform at {name!r} returned dtype {replacement.dtype}, expected {original.dtype}"
        )
    if replacement.device != original.device:
        raise HookSiteError(
            f"transform at {name!r} returned device {replacement.device}, expected {original.device}"
        )


def capture_all(*, detach: bool = True) -> CaptureContext:
    """A context that records every site a forward pass declares."""
    return CaptureContext(capture_all=True, detach=detach)
