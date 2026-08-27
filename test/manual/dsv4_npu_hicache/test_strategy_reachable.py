"""Task C128-1: DSV4 stack strategy must accept {FULL, SWA, C128}. Container cx-dsv4."""
from unittest.mock import MagicMock

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as A


def main():
    strat = A._DeepSeekV4Strategy()
    comps_gpu = {ComponentType.FULL, ComponentType.SWA}
    comps_npu = {ComponentType.FULL, ComponentType.SWA, ComponentType.C128}

    # MagicMock(spec=Cls) makes isinstance(obj, Cls) return True without
    # constructing the real object (avoids device-init failures).
    import sglang.srt.mem_cache.deepseek_v4_memory_pool as M

    fake = MagicMock(spec=M.DeepSeekV4TokenToKVPool)
    assert strat.matches(fake, comps_gpu), "GPU {FULL,SWA} regressed"
    assert strat.matches(fake, comps_npu), "NPU {FULL,SWA,C128} not accepted"
    print("PASS: strategy accepts both GPU and NPU component sets")


if __name__ == "__main__":
    main()
