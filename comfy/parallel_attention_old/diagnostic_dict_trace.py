"""Diagnostic script to trace dictionary transformations during model loading.

Run this standalone to see exactly what's happening to the state_dict
as it moves from parent to workers through serialization.
"""

import sys
import os
import logging
import torch
import pickle
import json
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

LOG_PREFIX = "🔍 [Dict-Trace]"


def analyze_dict_structure(d, name="dict", max_keys=10):
    """Analyze and log dictionary structure."""
    logging.info(f"\n{LOG_PREFIX} === {name} ===")
    logging.info(f"{LOG_PREFIX}   Type: {type(d)}")
    
    if not isinstance(d, dict):
        logging.error(f"{LOG_PREFIX}   ❌ NOT A DICT! Type: {type(d)}")
        return
    
    logging.info(f"{LOG_PREFIX}   Keys count: {len(d)}")
    
    if len(d) == 0:
        logging.warning(f"{LOG_PREFIX}   ⚠️  EMPTY DICT")
        return
    
    # Sample keys
    keys_list = list(d.keys())[:max_keys]
    logging.info(f"{LOG_PREFIX}   Sample keys ({len(keys_list)}):")
    for key in keys_list:
        value = d[key]
        logging.info(f"{LOG_PREFIX}     '{key}': {type(value).__name__}")
    
    # Check for nested dicts
    nested = sum(1 for v in d.values() if isinstance(v, dict))
    if nested > 0:
        logging.info(f"{LOG_PREFIX}   Nested dicts: {nested}")
    
    return d


def test_pickle_roundtrip(original_dict, name="dict"):
    """Test if dict survives pickle serialization."""
    logging.info(f"\n{LOG_PREFIX} === Testing Pickle Roundtrip: {name} ===")
    
    try:
        # Serialize
        serialized = pickle.dumps(original_dict)
        logging.info(f"{LOG_PREFIX}   ✅ Serialized: {len(serialized)} bytes")
        
        # Deserialize
        deserialized = pickle.loads(serialized)
        logging.info(f"{LOG_PREFIX}   ✅ Deserialized")
        
        # Compare
        analyze_dict_structure(deserialized, f"{name} (after pickle)")
        
        # Check keys match
        orig_keys = set(original_dict.keys())
        deser_keys = set(deserialized.keys())
        
        if orig_keys == deser_keys:
            logging.info(f"{LOG_PREFIX}   ✅ Keys match: {len(orig_keys)}")
        else:
            missing = orig_keys - deser_keys
            extra = deser_keys - orig_keys
            logging.error(f"{LOG_PREFIX}   ❌ Keys mismatch!")
            if missing:
                logging.error(f"{LOG_PREFIX}     Missing: {missing}")
            if extra:
                logging.error(f"{LOG_PREFIX}     Extra: {extra}")
        
        return deserialized
        
    except Exception as e:
        logging.error(f"{LOG_PREFIX}   ❌ Pickle failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_config_serialization():
    """Test model_config_dict serialization specifically."""
    logging.info(f"\n{LOG_PREFIX} {'='*60}")
    logging.info(f"{LOG_PREFIX} TEST: Model Config Serialization")
    logging.info(f"{LOG_PREFIX} {'='*60}")
    
    # Simulate what parent process does
    logging.info(f"\n{LOG_PREFIX} --- PARENT PROCESS ---")
    
    # Mock model config
    class MockModelConfig:
        def __init__(self):
            self.__class__.__name__ = "FLUX"
            self.unet_config = {
                'in_channels': 4,
                'out_channels': 4,
                'hidden_size': 3072,
            }
    
    model_config = MockModelConfig()
    
    # Create dict like parent does
    model_config_dict = {
        'class_name': model_config.__class__.__name__,
        'unet_config': model_config.unet_config,
    }
    
    logging.info(f"{LOG_PREFIX} Created model_config_dict:")
    analyze_dict_structure(model_config_dict, "model_config_dict (parent)")
    
    # Test JSON serialization (what multiprocessing uses)
    logging.info(f"\n{LOG_PREFIX} --- JSON SERIALIZATION (multiprocessing) ---")
    try:
        json_str = json.dumps(model_config_dict)
        logging.info(f"{LOG_PREFIX}   ✅ JSON serialized: {len(json_str)} chars")
        
        json_roundtrip = json.loads(json_str)
        analyze_dict_structure(json_roundtrip, "model_config_dict (after JSON)")
        
    except Exception as e:
        logging.error(f"{LOG_PREFIX}   ❌ JSON serialization failed: {e}")
    
    # Test pickle (backup path)
    test_pickle_roundtrip(model_config_dict, "model_config_dict")
    
    # Simulate worker receiving it
    logging.info(f"\n{LOG_PREFIX} --- WORKER PROCESS ---")
    
    # Workers receive as kwargs
    kwargs = {'model_config_dict': model_config_dict}
    analyze_dict_structure(kwargs, "kwargs")
    
    received_dict = kwargs.get('model_config_dict')
    if received_dict is None:
        logging.error(f"{LOG_PREFIX}   ❌ 'model_config_dict' not in kwargs!")
        logging.error(f"{LOG_PREFIX}   Available keys: {list(kwargs.keys())}")
    else:
        analyze_dict_structure(received_dict, "model_config_dict (worker)")


def test_checkpoint_path_passing():
    """Test checkpoint_path passing."""
    logging.info(f"\n{LOG_PREFIX} {'='*60}")
    logging.info(f"{LOG_PREFIX} TEST: Checkpoint Path Passing")
    logging.info(f"{LOG_PREFIX} {'='*60}")
    
    # Simulate parent
    checkpoint_path = "/fake/path/to/flux-dev.safetensors"
    
    kwargs = {
        'checkpoint_path': checkpoint_path,
        'model_config_dict': {'class_name': 'FLUX'},
    }
    
    logging.info(f"{LOG_PREFIX} Parent kwargs:")
    analyze_dict_structure(kwargs, "kwargs (parent)")
    
    # Simulate pickle
    logging.info(f"{LOG_PREFIX} Simulating multiprocessing transfer...")
    kwargs_after = test_pickle_roundtrip(kwargs, "kwargs")
    
    if kwargs_after:
        logging.info(f"{LOG_PREFIX} Worker received:")
        logging.info(f"{LOG_PREFIX}   checkpoint_path: {kwargs_after.get('checkpoint_path')}")
        logging.info(f"{LOG_PREFIX}   model_config_dict: {kwargs_after.get('model_config_dict')}")


def test_actual_load_function_signature():
    """Test if load_fsdp2_model would receive args correctly."""
    logging.info(f"\n{LOG_PREFIX} {'='*60}")
    logging.info(f"{LOG_PREFIX} TEST: Function Signature Matching")
    logging.info(f"{LOG_PREFIX} {'='*60}")
    
    def mock_load_fsdp2_model(checkpoint_path: str, model_config_dict: dict) -> dict:
        """Mock the actual function."""
        logging.info(f"{LOG_PREFIX} Function called!")
        logging.info(f"{LOG_PREFIX}   checkpoint_path type: {type(checkpoint_path)}")
        logging.info(f"{LOG_PREFIX}   checkpoint_path value: {checkpoint_path}")
        logging.info(f"{LOG_PREFIX}   model_config_dict type: {type(model_config_dict)}")
        analyze_dict_structure(model_config_dict, "model_config_dict (in function)")
        return {"success": True}
    
    # Test 1: Direct call
    logging.info(f"\n{LOG_PREFIX} Test 1: Direct function call")
    try:
        result = mock_load_fsdp2_model(
            checkpoint_path="/fake/path.safetensors",
            model_config_dict={'class_name': 'FLUX', 'unet_config': {}}
        )
        logging.info(f"{LOG_PREFIX}   ✅ Direct call works: {result}")
    except Exception as e:
        logging.error(f"{LOG_PREFIX}   ❌ Direct call failed: {e}")
    
    # Test 2: Kwargs unpacking
    logging.info(f"\n{LOG_PREFIX} Test 2: Kwargs unpacking")
    kwargs = {
        'checkpoint_path': '/fake/path.safetensors',
        'model_config_dict': {'class_name': 'FLUX', 'unet_config': {}}
    }
    try:
        result = mock_load_fsdp2_model(**kwargs)
        logging.info(f"{LOG_PREFIX}   ✅ Kwargs unpacking works: {result}")
    except Exception as e:
        logging.error(f"{LOG_PREFIX}   ❌ Kwargs unpacking failed: {e}")
    
    # Test 3: After pickle roundtrip
    logging.info(f"\n{LOG_PREFIX} Test 3: After pickle roundtrip")
    kwargs_pickled = test_pickle_roundtrip(kwargs, "kwargs")
    if kwargs_pickled:
        try:
            result = mock_load_fsdp2_model(**kwargs_pickled)
            logging.info(f"{LOG_PREFIX}   ✅ After pickle works: {result}")
        except Exception as e:
            logging.error(f"{LOG_PREFIX}   ❌ After pickle failed: {e}")


def test_executor_message_format():
    """Test the exact format executor uses for messages."""
    logging.info(f"\n{LOG_PREFIX} {'='*60}")
    logging.info(f"{LOG_PREFIX} TEST: Executor Message Format")
    logging.info(f"{LOG_PREFIX} {'='*60}")
    
    # Simulate what executor.execute_collective does
    method = "load_fsdp2_model"
    kwargs = {
        'checkpoint_path': '/fake/path.safetensors',
        'model_config_dict': {
            'class_name': 'FLUX',
            'unet_config': {'in_channels': 4}
        }
    }
    
    # Message format from executor.py
    msg = {
        'type': 'collective',
        'method': method,
        'kwargs': kwargs
    }
    
    logging.info(f"{LOG_PREFIX} Message structure:")
    analyze_dict_structure(msg, "executor message")
    
    logging.info(f"{LOG_PREFIX} Message components:")
    logging.info(f"{LOG_PREFIX}   type: {msg['type']}")
    logging.info(f"{LOG_PREFIX}   method: {msg['method']}")
    analyze_dict_structure(msg['kwargs'], "msg['kwargs']")
    
    # Test pickle of entire message
    msg_pickled = test_pickle_roundtrip(msg, "executor message")
    
    if msg_pickled:
        logging.info(f"{LOG_PREFIX} After pickle:")
        logging.info(f"{LOG_PREFIX}   type: {msg_pickled.get('type')}")
        logging.info(f"{LOG_PREFIX}   method: {msg_pickled.get('method')}")
        analyze_dict_structure(msg_pickled.get('kwargs', {}), "msg_pickled['kwargs']")


def main():
    """Run all diagnostic tests."""
    logging.info(f"\n{LOG_PREFIX} {'='*60}")
    logging.info(f"{LOG_PREFIX} DIAGNOSTIC: Dictionary Serialization Trace")
    logging.info(f"{LOG_PREFIX} {'='*60}")
    
    test_model_config_serialization()
    test_checkpoint_path_passing()
    test_actual_load_function_signature()
    test_executor_message_format()
    
    logging.info(f"\n{LOG_PREFIX} {'='*60}")
    logging.info(f"{LOG_PREFIX} DIAGNOSTIC COMPLETE")
    logging.info(f"{LOG_PREFIX} {'='*60}")


if __name__ == '__main__':
    main()
