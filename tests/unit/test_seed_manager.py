"""
Unit tests for the SeedManager class.

Tests the centralized random seed management system for reproducible experiments.
"""

import random
import threading
from unittest.mock import patch

import pytest

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import importlib.util

HAS_PYTORCH_LIGHTNING = importlib.util.find_spec("pytorch_lightning") is not None

from hydro_forecasting.experiment_utils.seed_manager import (
    SeedManager,
    get_global_seed_manager,
    init_global_seed_manager,
    set_global_seed_manager,
)


@pytest.fixture
def test_seed():
    """Test seed fixture."""
    return 12345


@pytest.fixture
def seed_manager(test_seed):
    """SeedManager fixture with test seed."""
    return SeedManager(test_seed)


class TestSeedManager:
    """Test cases for the SeedManager class."""

    def test_initialization_with_seed(self):
        """Test SeedManager initialization with a master seed."""
        sm = SeedManager(42)
        assert sm.master_seed == 42
        assert sm._component_seeds is not None
        assert len(sm._component_seeds) > 0

    def test_initialization_without_seed(self):
        """Test SeedManager initialization without a master seed."""
        sm = SeedManager(None)
        assert sm.master_seed is None
        assert len(sm._component_seeds) == 0

    def test_component_seeds_generation(self):
        """Test that component seeds are generated deterministically."""
        sm1 = SeedManager(42)
        sm2 = SeedManager(42)

        # Same master seed should produce same component seeds
        assert sm1._component_seeds == sm2._component_seeds

        # Different master seeds should produce different component seeds
        sm3 = SeedManager(43)
        assert sm1._component_seeds != sm3._component_seeds

    def test_seed_derivation_deterministic(self, seed_manager):
        """Test that seed derivation is deterministic."""
        component = "test_component"
        operation = "test_operation"

        # Same inputs should produce same output
        seed1 = seed_manager._derive_seed(component, operation)
        seed2 = seed_manager._derive_seed(component, operation)
        assert seed1 == seed2

        # Different components should produce different seeds
        seed3 = seed_manager._derive_seed("different_component", operation)
        assert seed1 != seed3

        # Different operations should produce different seeds
        seed4 = seed_manager._derive_seed(component, "different_operation")
        assert seed1 != seed4

    def test_seed_derivation_worker_id(self, seed_manager):
        """Test seed derivation with worker IDs."""
        component = "data_loading"

        # Different worker IDs should produce different seeds
        seed1 = seed_manager._derive_seed(component, "worker", 0)
        seed2 = seed_manager._derive_seed(component, "worker", 1)
        assert seed1 != seed2

    def test_seed_derivation_without_master_seed(self):
        """Test seed derivation without a master seed."""
        sm = SeedManager(None)

        # Should return random seed when no master seed
        with patch("random.randint") as mock_randint:
            mock_randint.return_value = 99999
            seed = sm._derive_seed("test_component")
            assert seed == 99999
            mock_randint.assert_called_once_with(0, 2**31 - 1)

    def test_get_component_seed_valid(self, seed_manager):
        """Test getting seed for valid component."""
        component = "datamodule_setup"
        seed = seed_manager.get_component_seed(component)
        assert isinstance(seed, int)
        assert 0 <= seed < 2**31 - 1

    def test_get_component_seed_invalid(self, seed_manager):
        """Test getting seed for invalid component raises ValueError."""
        with pytest.raises(ValueError, match="Unknown component"):
            seed_manager.get_component_seed("invalid_component")

    def test_get_operation_seed(self, seed_manager):
        """Test getting seed for specific operation."""
        component = "preprocessing_unified"
        operation = "basin_selection"

        seed1 = seed_manager.get_operation_seed(component, operation)
        seed2 = seed_manager.get_operation_seed(component, operation)

        # Should be deterministic
        assert seed1 == seed2

        # Different operation should give different seed
        seed3 = seed_manager.get_operation_seed(component, "different_operation")
        assert seed1 != seed3

    def test_get_worker_seed(self, seed_manager):
        """Test getting seed for worker processes."""
        seed1 = seed_manager.get_worker_seed(0)
        seed2 = seed_manager.get_worker_seed(1)

        # Different workers should have different seeds
        assert seed1 != seed2

        # Same worker should have same seed
        seed3 = seed_manager.get_worker_seed(0)
        assert seed1 == seed3

    def test_temporary_seed_context(self, seed_manager):
        """Test temporary seed context manager."""
        original_state = random.getstate()

        with seed_manager.temporary_seed("test_operation") as temp_seed:
            # Should have changed the random state
            assert random.getstate() != original_state
            assert isinstance(temp_seed, int)

            # Get some random values inside context
            random_values_inside = [random.random() for _ in range(5)]

        # Test reproducibility
        with seed_manager.temporary_seed("test_operation"):
            random_values_repeat = [random.random() for _ in range(5)]

        # Same operation should produce same random sequence
        assert random_values_inside == random_values_repeat

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_temporary_seed_context_numpy(self, seed_manager):
        """Test temporary seed context manager with NumPy."""
        original_state = np.random.get_state()

        with seed_manager.temporary_seed("test_operation"):
            # Should have changed the numpy random state
            assert not np.array_equal(np.random.get_state()[1], original_state[1])
            numpy_values_inside = [np.random.random() for _ in range(5)]

        # Test reproducibility
        with seed_manager.temporary_seed("test_operation"):
            numpy_values_repeat = [np.random.random() for _ in range(5)]

        np.testing.assert_array_equal(numpy_values_inside, numpy_values_repeat)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_temporary_seed_context_torch(self, seed_manager):
        """Test temporary seed context manager with PyTorch."""
        original_state = torch.get_rng_state()

        with seed_manager.temporary_seed("test_operation"):
            # Should have changed the torch random state
            assert not torch.equal(torch.get_rng_state(), original_state)
            torch_values_inside = [torch.rand(1).item() for _ in range(5)]

        # Test reproducibility
        with seed_manager.temporary_seed("test_operation"):
            torch_values_repeat = [torch.rand(1).item() for _ in range(5)]

        assert torch_values_inside == torch_values_repeat

    def test_set_global_seeds_with_seed(self, seed_manager, test_seed):
        """Test setting global seeds when master seed is available."""
        with patch("random.seed") as mock_random_seed:
            # Mock PyTorch Lightning to test fallback behavior
            with patch("hydro_forecasting.experiment_utils.seed_manager.HAS_PYTORCH_LIGHTNING", False):
                seed_manager.set_global_seeds()
                mock_random_seed.assert_called_once_with(test_seed)

    def test_set_global_seeds_without_seed(self):
        """Test setting global seeds when no master seed is available."""
        sm = SeedManager(None)

        with patch("random.seed") as mock_random_seed:
            sm.set_global_seeds()
            mock_random_seed.assert_not_called()

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_set_global_seeds_numpy(self, seed_manager, test_seed):
        """Test setting global seeds includes NumPy."""
        with patch("numpy.random.seed") as mock_numpy_seed:
            # Mock PyTorch Lightning to test fallback behavior
            with patch("hydro_forecasting.experiment_utils.seed_manager.HAS_PYTORCH_LIGHTNING", False):
                seed_manager.set_global_seeds()
                mock_numpy_seed.assert_called_once_with(test_seed)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_set_global_seeds_torch(self, seed_manager, test_seed):
        """Test setting global seeds includes PyTorch."""
        with patch("torch.manual_seed") as mock_torch_seed:
            # Mock PyTorch Lightning to test fallback behavior
            with patch("hydro_forecasting.experiment_utils.seed_manager.HAS_PYTORCH_LIGHTNING", False):
                seed_manager.set_global_seeds()
                mock_torch_seed.assert_called_once_with(test_seed)

    @pytest.mark.skipif(not HAS_PYTORCH_LIGHTNING, reason="PyTorch Lightning not available")
    def test_set_global_seeds_lightning(self, seed_manager, test_seed):
        """Test setting global seeds includes PyTorch Lightning."""
        with patch("pytorch_lightning.seed_everything") as mock_pl_seed:
            seed_manager.set_global_seeds()
            mock_pl_seed.assert_called_once_with(test_seed, workers=True)

    def test_save_and_restore_state(self, seed_manager):
        """Test saving and restoring random state."""
        # Set a known state
        random.seed(999)
        initial_values = [random.random() for _ in range(3)]

        # Save state
        saved_state = seed_manager.save_state()
        assert "master_seed" in saved_state
        assert "component_seeds" in saved_state
        assert "python_state" in saved_state

        # Change state
        random.seed(888)
        changed_values = [random.random() for _ in range(3)]
        assert initial_values != changed_values

        # Restore state
        seed_manager.restore_state(saved_state)
        restored_values = [random.random() for _ in range(3)]

        # Values should continue from where they left off
        assert len(restored_values) == 3

    def test_worker_init_fn(self, seed_manager):
        """Test worker initialization function."""
        worker_id = 5

        with patch("random.seed") as mock_random_seed:
            seed_manager.worker_init_fn(worker_id)

            # Should have called random.seed with worker-specific seed
            mock_random_seed.assert_called_once()
            called_seed = mock_random_seed.call_args[0][0]

            # Verify it's the expected worker seed
            expected_seed = seed_manager.get_worker_seed(worker_id)
            assert called_seed == expected_seed

    def test_get_info(self, seed_manager, test_seed):
        """Test getting seed manager information."""
        info = seed_manager.get_info()

        assert "master_seed" in info
        assert "has_master_seed" in info
        assert "component_seeds" in info
        assert "available_libraries" in info
        assert "component_descriptions" in info

        assert info["master_seed"] == test_seed
        assert info["has_master_seed"] is True
        assert isinstance(info["component_seeds"], dict)
        assert isinstance(info["available_libraries"], dict)

    def test_thread_safety(self, seed_manager):
        """Test that SeedManager operations are thread-safe."""
        results = []
        errors = []

        def worker_function():
            try:
                # Perform various operations that should be thread-safe
                seed = seed_manager.get_component_seed("datamodule_setup")
                results.append(seed)

                with seed_manager.temporary_seed("test_op"):
                    val = random.random()
                    results.append(val)

                worker_seed = seed_manager.get_worker_seed(threading.current_thread().ident % 100)
                results.append(worker_seed)

            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker_function)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0, f"Thread errors: {errors}"

        # Should have results from all threads
        assert len(results) == 30  # 3 operations * 10 threads


class TestGlobalSeedManager:
    """Test cases for global seed manager functions."""

    def setup_method(self):
        """Reset global seed manager before each test."""
        set_global_seed_manager(None)

    def test_init_global_seed_manager(self):
        """Test initializing global seed manager."""
        master_seed = 12345
        sm = init_global_seed_manager(master_seed)

        assert isinstance(sm, SeedManager)
        assert sm.master_seed == master_seed

        # Should be set as global instance
        global_sm = get_global_seed_manager()
        assert sm is global_sm

    def test_set_and_get_global_seed_manager(self):
        """Test setting and getting global seed manager."""
        sm = SeedManager(42)

        # Initially should be None
        assert get_global_seed_manager() is None

        # Set global instance
        set_global_seed_manager(sm)

        # Should return the same instance
        global_sm = get_global_seed_manager()
        assert sm is global_sm


class TestSeedManagerIntegration:
    """Integration tests for SeedManager with actual libraries."""

    @pytest.fixture
    def seed_manager(self):
        """SeedManager fixture for integration tests."""
        return SeedManager(42)

    def test_reproducible_random_sequence(self, seed_manager):
        """Test that the same seed produces reproducible random sequences."""
        # First run
        seed_manager.set_global_seeds()
        sequence1 = [random.random() for _ in range(10)]

        # Second run with same seed
        sm2 = SeedManager(42)
        sm2.set_global_seeds()
        sequence2 = [random.random() for _ in range(10)]

        # Should be identical
        assert sequence1 == sequence2

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_reproducible_numpy_sequence(self, seed_manager):
        """Test that the same seed produces reproducible NumPy sequences."""
        # First run
        seed_manager.set_global_seeds()
        sequence1 = np.random.random(10).tolist()

        # Second run with same seed
        sm2 = SeedManager(42)
        sm2.set_global_seeds()
        sequence2 = np.random.random(10).tolist()

        # Should be identical
        np.testing.assert_array_equal(sequence1, sequence2)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_reproducible_torch_sequence(self, seed_manager):
        """Test that the same seed produces reproducible PyTorch sequences."""
        # First run
        seed_manager.set_global_seeds()
        sequence1 = torch.rand(10).tolist()

        # Second run with same seed
        sm2 = SeedManager(42)
        sm2.set_global_seeds()
        sequence2 = torch.rand(10).tolist()

        # Should be identical
        assert sequence1 == sequence2

    def test_component_isolation(self, seed_manager):
        """Test that different components get isolated seed spaces."""
        seeds = {}
        for component in seed_manager.COMPONENT_SEEDS:
            seeds[component] = seed_manager.get_component_seed(component)

        # All component seeds should be different
        seed_values = list(seeds.values())
        assert len(seed_values) == len(set(seed_values))

    def test_deterministic_worker_seeds(self, seed_manager):
        """Test that worker seeds are deterministic and unique."""
        worker_seeds = []
        for worker_id in range(10):
            seed = seed_manager.get_worker_seed(worker_id)
            worker_seeds.append(seed)

        # All worker seeds should be different
        assert len(worker_seeds) == len(set(worker_seeds))

        # Should be reproducible
        worker_seeds_repeat = []
        for worker_id in range(10):
            seed = seed_manager.get_worker_seed(worker_id)
            worker_seeds_repeat.append(seed)

        assert worker_seeds == worker_seeds_repeat
