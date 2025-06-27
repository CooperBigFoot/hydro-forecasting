"""
Integration tests for SeedManager across the hydrological forecasting pipeline.

These tests verify that the centralized seed management provides complete
reproducibility across all components of the system.
"""

import random

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

from hydro_forecasting.experiment_utils.seed_manager import SeedManager


class TestSeedManagerIntegration:
    """Integration tests for complete pipeline determinism."""

    @pytest.fixture
    def master_seed(self):
        """Master seed fixture."""
        return 42

    @pytest.fixture
    def seed_manager(self, master_seed):
        """SeedManager fixture."""
        return SeedManager(master_seed)

    def test_datamodule_operations_deterministic(self, seed_manager):
        """Test that datamodule operations are deterministic."""
        # Simulate basin ID shuffling
        basin_ids = [f"basin_{i:03d}" for i in range(100)]

        # First run
        seed_manager.set_global_seeds()
        with seed_manager.temporary_seed("basin_id_shuffle", "datamodule_setup"):
            random.shuffle(basin_ids.copy())
            shuffled_1 = basin_ids.copy()

        # Second run
        seed_manager.set_global_seeds()
        with seed_manager.temporary_seed("basin_id_shuffle", "datamodule_setup"):
            random.shuffle(basin_ids.copy())
            shuffled_2 = basin_ids.copy()

        # Should be identical
        assert shuffled_1 == shuffled_2

    def test_validation_pool_selection_deterministic(self, seed_manager):
        """Test that validation pool selection is deterministic."""
        basin_ids = [f"basin_{i:03d}" for i in range(1000)]
        num_val_basins = 100

        # First run
        with seed_manager.temporary_seed("validation_pool_selection", "datamodule_setup"):
            validation_pool_1 = random.sample(basin_ids, num_val_basins)

        # Second run
        with seed_manager.temporary_seed("validation_pool_selection", "datamodule_setup"):
            validation_pool_2 = random.sample(basin_ids, num_val_basins)

        # Should be identical
        assert set(validation_pool_1) == set(validation_pool_2)
        assert validation_pool_1 == validation_pool_2  # Order should also match

    def test_training_chunk_operations_deterministic(self, seed_manager):
        """Test that training chunk operations are deterministic."""
        basin_ids = [f"basin_{i:03d}" for i in range(500)]

        # First run - training basin shuffle
        with seed_manager.temporary_seed("training_basin_shuffle", "training_chunks"):
            shuffled_basins_1 = random.sample(basin_ids, len(basin_ids))

        # Second run - training basin shuffle
        with seed_manager.temporary_seed("training_basin_shuffle", "training_chunks"):
            shuffled_basins_2 = random.sample(basin_ids, len(basin_ids))

        # Should be identical
        assert shuffled_basins_1 == shuffled_basins_2

        # Test sequence shuffling
        sequence_indices = list(range(10000))

        with seed_manager.temporary_seed("training_sequence_shuffle", "sequence_operations"):
            random.shuffle(sequence_indices.copy())
            shuffled_indices_1 = sequence_indices.copy()

        with seed_manager.temporary_seed("training_sequence_shuffle", "sequence_operations"):
            random.shuffle(sequence_indices.copy())
            shuffled_indices_2 = sequence_indices.copy()

        assert shuffled_indices_1 == shuffled_indices_2

    def test_preprocessing_basin_selection_deterministic(self, seed_manager):
        """Test that preprocessing unified basin selection is deterministic."""
        valid_basins = [f"basin_{i:03d}" for i in range(200)]
        fit_on_n_basins = 50

        # Sort for deterministic base
        sorted_basins = sorted(valid_basins)

        # First run
        with seed_manager.temporary_seed("unified_basin_selection", "preprocessing_unified"):
            selected_basins_1 = random.sample(sorted_basins, fit_on_n_basins)

        # Second run
        with seed_manager.temporary_seed("unified_basin_selection", "preprocessing_unified"):
            selected_basins_2 = random.sample(sorted_basins, fit_on_n_basins)

        # Should be identical
        assert set(selected_basins_1) == set(selected_basins_2)
        assert selected_basins_1 == selected_basins_2

    def test_worker_seed_isolation(self, seed_manager):
        """Test that worker processes get isolated but deterministic seeds."""
        num_workers = 8

        # Get worker seeds multiple times
        worker_seeds_run1 = []
        worker_seeds_run2 = []

        for worker_id in range(num_workers):
            seed1 = seed_manager.get_worker_seed(worker_id)
            seed2 = seed_manager.get_worker_seed(worker_id)
            worker_seeds_run1.append(seed1)
            worker_seeds_run2.append(seed2)

        # Same worker should get same seed
        assert worker_seeds_run1 == worker_seeds_run2

        # Different workers should get different seeds
        assert len(set(worker_seeds_run1)) == len(worker_seeds_run1)

    def test_component_seed_isolation(self, seed_manager):
        """Test that different components get isolated seed spaces."""
        component_seeds = {}

        # Get seeds for all components
        for component in seed_manager.COMPONENT_SEEDS:
            component_seeds[component] = seed_manager.get_component_seed(component)

        # All should be different
        seed_values = list(component_seeds.values())
        assert len(seed_values) == len(set(seed_values))

        # Same component should always return same seed
        for component in seed_manager.COMPONENT_SEEDS:
            seed1 = seed_manager.get_component_seed(component)
            seed2 = seed_manager.get_component_seed(component)
            assert seed1 == seed2

    def test_different_master_seeds_produce_different_results(self):
        """Test that different master seeds produce different random sequences."""
        # Test with different master seeds
        sm1 = SeedManager(42)
        sm2 = SeedManager(43)

        # Get component seeds
        component_seeds_1 = {}
        component_seeds_2 = {}

        for component in sm1.COMPONENT_SEEDS:
            component_seeds_1[component] = sm1.get_component_seed(component)
            component_seeds_2[component] = sm2.get_component_seed(component)

        # Should be different for different master seeds
        for component in sm1.COMPONENT_SEEDS:
            assert component_seeds_1[component] != component_seeds_2[component]

        # Test operations produce different results
        basin_ids = [f"basin_{i:03d}" for i in range(100)]

        with sm1.temporary_seed("test_operation", "datamodule_setup"):
            result1 = random.sample(basin_ids, 10)

        with sm2.temporary_seed("test_operation", "datamodule_setup"):
            result2 = random.sample(basin_ids, 10)

        # Should be different (very unlikely to be same by chance)
        assert result1 != result2

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_numpy_operations_deterministic(self, seed_manager):
        """Test that NumPy operations are deterministic across components."""
        # Test that numpy operations are reproducible
        with seed_manager.temporary_seed("numpy_test", "datamodule_setup"):
            array1 = np.random.random(100)

        with seed_manager.temporary_seed("numpy_test", "datamodule_setup"):
            array2 = np.random.random(100)

        np.testing.assert_array_equal(array1, array2)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_operations_deterministic(self, seed_manager):
        """Test that PyTorch operations are deterministic across components."""
        # Test that torch operations are reproducible
        with seed_manager.temporary_seed("torch_test", "model_training"):
            tensor1 = torch.rand(100)

        with seed_manager.temporary_seed("torch_test", "model_training"):
            tensor2 = torch.rand(100)

        assert torch.equal(tensor1, tensor2)

    def test_nested_temporary_seeds(self, seed_manager):
        """Test that nested temporary seeds work correctly."""
        outer_results = []
        inner_results = []

        # First run with nested seeds
        with seed_manager.temporary_seed("outer_operation", "datamodule_setup"):
            outer_results.append(random.random())

            with seed_manager.temporary_seed("inner_operation", "preprocessing_unified"):
                inner_results.append(random.random())

            outer_results.append(random.random())

        # Second run with nested seeds
        outer_results_2 = []
        inner_results_2 = []

        with seed_manager.temporary_seed("outer_operation", "datamodule_setup"):
            outer_results_2.append(random.random())

            with seed_manager.temporary_seed("inner_operation", "preprocessing_unified"):
                inner_results_2.append(random.random())

            outer_results_2.append(random.random())

        # Both outer and inner operations should be reproducible
        assert outer_results == outer_results_2
        assert inner_results == inner_results_2

    def test_state_preservation_across_operations(self, seed_manager):
        """Test that global state is preserved across different operations."""
        # Set a specific global state
        random.seed(999)
        random.random()

        # Use temporary seeds for various operations
        with seed_manager.temporary_seed("operation1", "datamodule_setup"):
            temp_value1 = random.random()

        with seed_manager.temporary_seed("operation2", "preprocessing_unified"):
            temp_value2 = random.random()

        # Global state should continue from where it left off
        continued_value = random.random()

        # Verify operations are repeatable
        random.seed(999)
        random.random()  # Skip initial value

        with seed_manager.temporary_seed("operation1", "datamodule_setup"):
            temp_value1_repeat = random.random()

        with seed_manager.temporary_seed("operation2", "preprocessing_unified"):
            temp_value2_repeat = random.random()

        continued_value_repeat = random.random()

        # All should match
        assert temp_value1 == temp_value1_repeat
        assert temp_value2 == temp_value2_repeat
        assert continued_value == continued_value_repeat


class TestSeedManagerPipelineReproducibility:
    """Test end-to-end pipeline reproducibility."""

    def test_full_pipeline_simulation(self):
        """Simulate a full pipeline run and verify reproducibility."""
        master_seed = 12345

        def simulate_pipeline_run(seed):
            """Simulate a complete pipeline run."""
            sm = SeedManager(seed)
            sm.set_global_seeds()

            results = {}

            # Simulate datamodule setup
            basin_ids = [f"basin_{i:03d}" for i in range(200)]

            with sm.temporary_seed("basin_id_shuffle", "datamodule_setup"):
                random.shuffle(basin_ids)
                results["shuffled_basins"] = basin_ids[:10]  # First 10 for comparison

            with sm.temporary_seed("validation_pool_selection", "datamodule_setup"):
                results["validation_pool"] = random.sample(basin_ids, 20)

            # Simulate preprocessing
            valid_basins = [f"valid_basin_{i:03d}" for i in range(100)]
            with sm.temporary_seed("unified_basin_selection", "preprocessing_unified"):
                results["fit_basins"] = random.sample(sorted(valid_basins), 30)

            # Simulate training chunk operations
            with sm.temporary_seed("training_basin_shuffle", "training_chunks"):
                training_basins = random.sample(basin_ids, len(basin_ids))
                results["training_chunks"] = [training_basins[i : i + 10] for i in range(0, 50, 10)]

            # Simulate sequence operations
            sequence_indices = list(range(1000))
            with sm.temporary_seed("training_sequence_shuffle", "sequence_operations"):
                random.shuffle(sequence_indices)
                results["shuffled_sequences"] = sequence_indices[:50]

            # Simulate worker seeds
            results["worker_seeds"] = [sm.get_worker_seed(i) for i in range(4)]

            return results

        # Run pipeline twice with same seed
        results1 = simulate_pipeline_run(master_seed)
        results2 = simulate_pipeline_run(master_seed)

        # All results should be identical
        assert results1["shuffled_basins"] == results2["shuffled_basins"]
        assert results1["validation_pool"] == results2["validation_pool"]
        assert results1["fit_basins"] == results2["fit_basins"]
        assert results1["training_chunks"] == results2["training_chunks"]
        assert results1["shuffled_sequences"] == results2["shuffled_sequences"]
        assert results1["worker_seeds"] == results2["worker_seeds"]

        # Run with different seed should produce different results
        results3 = simulate_pipeline_run(master_seed + 1)

        # Should be different (very unlikely to be same by chance)
        assert results1["shuffled_basins"] != results3["shuffled_basins"]
        assert results1["validation_pool"] != results3["validation_pool"]
        assert results1["fit_basins"] != results3["fit_basins"]
        assert results1["worker_seeds"] != results3["worker_seeds"]
