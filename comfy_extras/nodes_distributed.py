"""Test node for distributed runtime."""

import torch
import torch.multiprocessing as mp
import logging

LOG_PREFIX = "⚡ [Parallel-Attention]"

class TestDistributedRuntime:
    """Test node for distributed multiprocess executor."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "world_size": ("INT", {"default": 2, "min": 1, "max": 8}),
                "backend": (["auto", "nccl", "gloo"],),
                "test_type": (["basic", "devicemesh", "all"],),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "test_executor"
    CATEGORY = "testing"
    
    def test_executor(self, world_size, backend, test_type):
        """Test the distributed executor."""
        
        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        from comfy.distributed import MultiprocExecutor
        
        logging.info(f"{LOG_PREFIX} [Test] Starting test: world_size={world_size}, backend={backend}, test_type={test_type}")
        
        try:
            # Test 1: Spawn and echo (always run)
            logging.info(f"{LOG_PREFIX} [Test] Test 1: Spawn workers and echo RPC")
            executor = MultiprocExecutor(world_size=world_size, backend=backend)
            
            # Test echo
            test_message = "hello from comfy"
            result = executor.execute_collective("echo", {"message": test_message})
            
            if result != test_message:
                executor.shutdown()
                return (f"FAIL: Echo test failed. Expected '{test_message}', got '{result}'",)
            
            logging.info(f"{LOG_PREFIX} [Test] Echo test passed: '{result}'")
            
            # Test 2: Multiple RPCs (if basic or all)
            if test_type in ["basic", "all"]:
                logging.info(f"{LOG_PREFIX} [Test] Test 2: Multiple sequential RPCs")
                for i in range(5):
                    message = f"message_{i}"
                    result = executor.execute_collective("echo", {"message": message})
                    if result != message:
                        executor.shutdown()
                        return (f"FAIL: RPC {i} failed",)
                
                logging.info(f"{LOG_PREFIX} [Test] Multiple RPC test passed")
            
            # Test 3: Collective operation (if basic or all, and CUDA available)
            if test_type in ["basic", "all"]:
                if torch.cuda.is_available() and executor.backend == "nccl":
                    logging.info(f"{LOG_PREFIX} [Test] Test 3: torch.distributed collective (all_reduce)")
                    logging.info(f"{LOG_PREFIX} [Test] ─────────────────────────────────────────────────────")
                    result = executor.execute_collective("allreduce_test", {})
                    logging.info(f"{LOG_PREFIX} [Test] ─────────────────────────────────────────────────────")
                    expected = sum(range(world_size))
                    
                    if result != expected:
                        executor.shutdown()
                        return (f"FAIL: all_reduce failed. Expected {expected}, got {result}",)
                    
                    logging.info(f"{LOG_PREFIX} [Test] Collective test passed: result={result}")
                else:
                    logging.info(f"{LOG_PREFIX} [Test] Test 3: Skipped (CUDA not available or not using NCCL)")
            
            # Test 4: DeviceMesh integration (if devicemesh or all)
            if test_type in ["devicemesh", "all"]:
                if torch.cuda.is_available() and executor.backend == "nccl":
                    logging.info(f"{LOG_PREFIX} [Test] Test 4: DeviceMesh topology and SP collective")
                    logging.info(f"{LOG_PREFIX} [Test] ═════════════════════════════════════════════════════")
                    result = executor.execute_collective("devicemesh_test", {})
                    logging.info(f"{LOG_PREFIX} [Test] ═════════════════════════════════════════════════════")
                    
                    # Validate mesh structure
                    mesh_shape = result.get("mesh_shape", [])
                    expected_shape = [1, world_size]  # [dp_size=1, sp_size=world_size]
                    
                    if mesh_shape != expected_shape:
                        executor.shutdown()
                        return (f"FAIL: Mesh shape mismatch. Expected {expected_shape}, got {mesh_shape}",)
                    
                    # Validate all_gather result
                    gathered = result.get("gathered", [])
                    expected_gathered = [float(i) for i in range(world_size)]
                    
                    if gathered != expected_gathered:
                        executor.shutdown()
                        return (f"FAIL: SP all_gather mismatch. Expected {expected_gathered}, got {gathered}",)
                    
                    logging.info(f"{LOG_PREFIX} [Test] DeviceMesh test passed:")
                    logging.info(f"{LOG_PREFIX} [Test]   Mesh shape: {mesh_shape} (dp=1, sp={world_size})")
                    logging.info(f"{LOG_PREFIX} [Test]   SP rank {result['sp_rank']}/{result['sp_size']}, DP rank {result['dp_rank']}/{result['dp_size']}")
                    logging.info(f"{LOG_PREFIX} [Test]   SP all_gather result: {gathered}")
                else:
                    logging.info(f"{LOG_PREFIX} [Test] Test 4: Skipped (CUDA not available or not using NCCL)")
            
            # Shutdown
            logging.info(f"{LOG_PREFIX} [Test] Shutting down executor")
            executor.shutdown()
            
            success_msg = (
                f"{LOG_PREFIX} [Test] PASS: All tests passed!\n"
                f"{LOG_PREFIX} [Test] world_size={world_size}\n"
                f"{LOG_PREFIX} [Test] backend={backend}\n"
                f"{LOG_PREFIX} [Test] test_type={test_type}\n"
                f"{LOG_PREFIX} [Test] CUDA available: {torch.cuda.is_available()}"
            )
            logging.info(success_msg)
            
            return (success_msg,)
            
        except Exception as e:
            import traceback
            error_msg = f"FAIL: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            logging.error(f"{LOG_PREFIX} [Test] {error_msg}")
            return (error_msg,)

NODE_CLASS_MAPPINGS = {
    "TestDistributedRuntime": TestDistributedRuntime
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TestDistributedRuntime": "Test Distributed Runtime"
}
