import sys
import unittest
from importlib.metadata import EntryPoint, EntryPoints
from types import ModuleType
from unittest.mock import patch

from sglang.kernels.plugin_kernels.npu import spec
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")

TARGET = "GlmMoeDsaForCausalLM"
NEXTN = "GlmMoeDsaForCausalLMNextN"


class TestNPURegistration(unittest.TestCase):
    def test_load_only_the_matching_architecture(self):
        module = ModuleType("example_npu_provider")
        module.target = spec.MegakernelSpec("target", lambda resources, cfg: None)
        providers = EntryPoints(
            [
                EntryPoint(TARGET, f"{module.__name__}:target", spec.ENTRY_POINT_GROUP),
                EntryPoint(NEXTN, "absent_nextn_provider:spec", spec.ENTRY_POINT_GROUP),
            ]
        )
        with patch.object(spec, "entry_points", return_value=providers) as discover:
            self.assertIsNone(spec.find_spec("UnknownArchitecture"))
            self.assertNotIn(module.__name__, sys.modules)
            with patch.dict(sys.modules, {module.__name__: module}):
                self.assertIs(spec.find_spec(TARGET), module.target)
            discover.assert_called_with(group=spec.ENTRY_POINT_GROUP)

    def test_conflicting_providers_are_rejected_before_load(self):
        providers = EntryPoints(
            [
                EntryPoint(TARGET, "first_provider:target", spec.ENTRY_POINT_GROUP),
                EntryPoint(TARGET, "second_provider:target", spec.ENTRY_POINT_GROUP),
            ]
        )
        with (
            patch.object(spec, "entry_points", return_value=providers),
            self.assertRaisesRegex(
                RuntimeError, "Conflicting NPU megakernel providers"
            ),
        ):
            spec.find_spec(TARGET)


if __name__ == "__main__":
    unittest.main()
