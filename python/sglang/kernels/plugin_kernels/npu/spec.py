"""NPU megakernel provider contract and opt-in discovery."""

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Callable, Literal, Optional, Protocol, Union


@dataclass(frozen=True)
class PreparationView:
    model: Any
    token_to_kv_pool: Any
    physical_slots: bool


@dataclass(frozen=True)
class Decline:
    """A refusal from prepare. Expected refusals are logged at info level."""

    reason: str
    expected: bool = False


class NpuBinding(Protocol):
    role: str
    mode: str

    def build(self, model: Any) -> None: ...
    def graph_backend(self, runner: Any, enable_memory_saver: bool) -> Any: ...
    def draft_runtime(self, worker: Any) -> Any: ...


@dataclass(frozen=True)
class MegakernelSpec:
    role: Literal["target", "nextn"]
    prepare: Callable[[PreparationView, Any], Union[NpuBinding, Decline, None]]


ENTRY_POINT_GROUP = "sglang.srt.npu_megakernels"


def find_spec(architecture: str) -> Optional[MegakernelSpec]:
    providers = entry_points(group=ENTRY_POINT_GROUP)
    if not providers:
        raise RuntimeError(
            "No NPU megakernel provider installed. Install a provider exposing "
            f"{ENTRY_POINT_GROUP}."
        )
    matches = providers.select(name=architecture)
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"Conflicting NPU megakernel providers for {architecture}")
    provider = next(iter(matches))
    spec = provider.load()
    if not isinstance(spec, MegakernelSpec):
        raise TypeError(f"Invalid NPU megakernel spec at {provider.value}")
    return spec
