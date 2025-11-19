"""Phase 1A unit test helper.

This module houses the reusable test harness invoked by
`ParallelAttentionUnitTests` so the node stays thin and the
suite can be reused from other entry points.
"""

from __future__ import annotations

import logging
import traceback
from typing import List

import torch

from comfy.parallel_attention import FSDP2Executor
from comfy.parallel_attention.fsdp2_config import ShardingConfig
from comfy.parallel_attention.fsdp2_policies import FSDP2PolicyRegistry

LOG_PREFIX = "⚡ [Parallel-Attention]"


def run_phase1a_tests(
    test_executor_spawn: bool,
    test_devicemesh_init: bool,
    test_worker_communication: bool,
    test_policy_registry: bool,
    run_all_phase1a: bool,
) -> str:
    """Execute the Phase 1A test battery and return a formatted summary string."""
    results: List[str] = []
    executor: FSDP2Executor | None = None

    try:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        executor = FSDP2Executor(world_size=2, backend=backend)
        logging.info(f"{LOG_PREFIX} [Test] Executor created (backend={backend})")

        any_enabled = (
            test_executor_spawn
            or test_devicemesh_init
            or test_worker_communication
            or test_policy_registry
            or run_all_phase1a
        )

        if not any_enabled:
            executor.shutdown()
            return "⚠️  No tests enabled. Enable at least one test."

        if test_executor_spawn or run_all_phase1a:
            results.append(_run_executor_spawn_test(executor))

        if test_devicemesh_init or run_all_phase1a:
            results.append(_run_devicemesh_init_test(executor))

        if test_worker_communication or run_all_phase1a:
            results.append(_run_worker_comm_test(executor))

        if test_policy_registry or run_all_phase1a:
            results.append(_run_policy_registry_test())

        if executor:
            executor.shutdown()
            all_terminated = all(not worker.is_alive() for worker in executor.workers)
            logging.info(
                f"{LOG_PREFIX} [Test] Executor shutdown: "
                f"{'✅ clean' if all_terminated else '❌ workers still alive'}"
            )

        return _summarise_results(results)

    except Exception as exc:  # pragma: no cover - defensive logging path
        error_msg = (
            f"❌ TEST FAILURE: {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        )
        logging.error(f"{LOG_PREFIX} [Test] {error_msg}")

        if executor:
            try:
                executor.shutdown()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

        return error_msg


def _run_executor_spawn_test(executor: FSDP2Executor) -> str:
    logging.info(f"{LOG_PREFIX} [Test] ┌─────────────────────────────────────────┐")
    logging.info(f"{LOG_PREFIX} [Test] │ TEST 1: Executor Spawn                  │")
    logging.info(f"{LOG_PREFIX} [Test] └─────────────────────────────────────────┘")

    checks_passed = 0
    checks_total = 4

    logging.info(f"{LOG_PREFIX} [Test]   ✅ [1/5] Executor initialized")
    checks_passed += 1

    valid_backend = executor.backend in ["nccl", "gloo"]
    logging.info(
        f"{LOG_PREFIX} [Test]   {'✅' if valid_backend else '❌'} "
        f"[2/5] Backend: {executor.backend}"
    )
    if valid_backend:
        checks_passed += 1

    workers_alive = len(executor.workers) == 2
    logging.info(
        f"{LOG_PREFIX} [Test]   {'✅' if workers_alive else '❌'} "
        f"[3/5] Workers: {len(executor.workers)}/2"
    )
    if workers_alive:
        checks_passed += 1

    result = executor.execute_collective("echo", {"message": "test"})
    echo_works = result == "test"
    logging.info(
        f"{LOG_PREFIX} [Test]   {'✅' if echo_works else '❌'} [4/5] Echo: {result}"
    )
    if echo_works:
        checks_passed += 1

    return _format_result("Test 1: Executor Spawn", checks_passed, checks_total)


def _run_devicemesh_init_test(executor: FSDP2Executor) -> str:
    logging.info(f"{LOG_PREFIX} [Test] ┌─────────────────────────────────────────┐")
    logging.info(f"{LOG_PREFIX} [Test] │ TEST 2: DeviceMesh Initialization       │")
    logging.info(f"{LOG_PREFIX} [Test] └─────────────────────────────────────────┘")

    checks_passed = 0
    checks_total = 5

    result = executor.execute_collective("check_devicemesh", {})

    has_mesh = result.get("has_mesh", False)
    logging.info(
        f"{LOG_PREFIX} [Test]   {'✅' if has_mesh else '❌'} [1/5] DeviceMesh created"
    )
    if has_mesh:
        checks_passed += 1

    mesh_shape = result.get("mesh_shape", ())
    correct_shape = mesh_shape == (2,)
    logging.info(
        f"{LOG_PREFIX} [Test]   {'✅' if correct_shape else '❌'} [2/5] Shape: {mesh_shape}"
    )
    if correct_shape:
        checks_passed += 1

    mesh_dims = result.get("mesh_dim_names", [])
    correct_dims = mesh_dims == ["dp"]
    logging.info(
        f"{LOG_PREFIX} [Test]   {'✅' if correct_dims else '❌'} [3/5] Dims: {mesh_dims}"
    )
    if correct_dims:
        checks_passed += 1

    has_groups = result.get("has_dp_group", False)
    logging.info(
        f"{LOG_PREFIX} [Test]   {'✅' if has_groups else '❌'} [4/5] Process groups"
    )
    if has_groups:
        checks_passed += 1

    dp_rank = result.get("dp_rank", -1)
    dp_size = result.get("dp_size", -1)
    ranks_ok = dp_rank in [0, 1] and dp_size == 2
    logging.info(
        f"{LOG_PREFIX} [Test]   {'✅' if ranks_ok else '❌'} [5/5] Rank: {dp_rank}/{dp_size}"
    )
    if ranks_ok:
        checks_passed += 1

    return _format_result("Test 2: DeviceMesh Init", checks_passed, checks_total)


def _run_worker_comm_test(executor: FSDP2Executor) -> str:
    logging.info(f"{LOG_PREFIX} [Test] ┌─────────────────────────────────────────┐")
    logging.info(f"{LOG_PREFIX} [Test] │ TEST 3: Worker Communication            │")
    logging.info(f"{LOG_PREFIX} [Test] └─────────────────────────────────────────┘")

    checks_passed = 0
    checks_total = 5

    result = executor.execute_collective("echo", {"message": "hello"})
    echo_ok = result == "hello"
    logging.info(
        f"{LOG_PREFIX} [Test]   {'✅' if echo_ok else '❌'} [1/5] Echo: {result}"
    )
    if echo_ok:
        checks_passed += 1

    result = executor.execute_collective("get_rank", {})
    rank_ok = result.get("rank") == 0
    logging.info(
        f"{LOG_PREFIX} [Test]   {'✅' if rank_ok else '❌'} [2/5] Collective: rank={result.get('rank')}"
    )
    if rank_ok:
        checks_passed += 1

    all_ok = True
    for idx in range(3):
        repeated = executor.execute_collective("echo", {"message": f"test{idx}"})
        if repeated != f"test{idx}":
            all_ok = False
            break
    logging.info(f"{LOG_PREFIX} [Test]   {'✅' if all_ok else '❌'} [3/5] Multiple calls work")
    if all_ok:
        checks_passed += 1

    result = executor.execute_collective("check_devicemesh", {})
    mesh_ok = result.get("has_mesh", False)
    logging.info(
        f"{LOG_PREFIX} [Test]   {'✅' if mesh_ok else '❌'} [4/5] DeviceMesh accessible"
    )
    if mesh_ok:
        checks_passed += 1

    all_alive = all(worker.is_alive() for worker in executor.workers)
    logging.info(
        f"{LOG_PREFIX} [Test]   {'✅' if all_alive else '❌'} [5/5] Workers still alive"
    )
    if all_alive:
        checks_passed += 1

    return _format_result("Test 3: Communication", checks_passed, checks_total)


def _run_policy_registry_test() -> str:
    logging.info(f"{LOG_PREFIX} [Test] ┌─────────────────────────────────────────┐")
    logging.info(f"{LOG_PREFIX} [Test] │ TEST 4: Policy Registry                 │")
    logging.info(f"{LOG_PREFIX} [Test] └─────────────────────────────────────────┘")

    checks_passed = 0
    checks_total = 5

    has_flux = FSDP2PolicyRegistry.is_registered("flux")
    logging.info(
        f"{LOG_PREFIX} [Test]   {'✅' if has_flux else '❌'} [1/5] Flux registered"
    )
    if has_flux:
        checks_passed += 1

    has_wan = FSDP2PolicyRegistry.is_registered("wan")
    logging.info(
        f"{LOG_PREFIX} [Test]   {'✅' if has_wan else '❌'} [2/5] Wan registered"
    )
    if has_wan:
        checks_passed += 1

    has_qwen = FSDP2PolicyRegistry.is_registered("qwen_image")
    logging.info(
        f"{LOG_PREFIX} [Test]   {'✅' if has_qwen else '❌'} [3/5] Qwen registered"
    )
    if has_qwen:
        checks_passed += 1

    config = FSDP2PolicyRegistry.get_policy("flux")
    is_config = isinstance(config, ShardingConfig)
    logging.info(
        f"{LOG_PREFIX} [Test]   {'✅' if is_config else '❌'} [4/5] Returns ShardingConfig"
    )
    if is_config:
        checks_passed += 1

    has_blocks = bool(config.blocks)
    logging.info(
        f"{LOG_PREFIX} [Test]   {'✅' if has_blocks else '❌'} [5/5] Blocks: {len(config.blocks)}"
    )
    if has_blocks:
        checks_passed += 1

    return _format_result("Test 4: Policy Registry", checks_passed, checks_total)


def _format_result(label: str, passed: int, total: int) -> str:
    prefix = "✅" if passed == total else "❌"
    return f"{prefix} {label} ({passed}/{total})"


def _summarise_results(results: List[str]) -> str:
    logging.info(f"{LOG_PREFIX} [Test] ══════════════════════════════════════════════════════")
    logging.info(f"{LOG_PREFIX} [Test] TEST SUMMARY")
    logging.info(f"{LOG_PREFIX} [Test] ══════════════════════════════════════════════════════")

    for line in results:
        logging.info(f"{LOG_PREFIX} [Test] {line}")

    passed_count = sum(1 for line in results if line.startswith("✅"))

    logging.info(f"{LOG_PREFIX} [Test] ──────────────────────────────────────────────────────")
    logging.info(f"{LOG_PREFIX} [Test] Total: {len(results)} tests ({passed_count} passed)")
    logging.info(f"{LOG_PREFIX} [Test] CUDA: {torch.cuda.is_available()}")
    logging.info(f"{LOG_PREFIX} [Test] ══════════════════════════════════════════════════════")

    summary_lines = [
        "═" * 70,
        "PHASE 1A: CORE SALVAGE UNIT TESTS - RESULTS",
        "═" * 70,
        "",
        *results,
        "",
        "─" * 70,
        f"Total: {len(results)} tests | Passed: {passed_count}",
        "═" * 70,
    ]

    return "\n".join(summary_lines)
