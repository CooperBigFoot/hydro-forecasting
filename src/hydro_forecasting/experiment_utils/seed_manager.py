"""
Centralized random seed management system for reproducible hydrological forecasting.

This module provides a robust, hierarchical seed management system that ensures
complete reproducibility across the entire data processing and training pipeline.
"""

import hashlib
import logging
import random
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]

try:
    import pytorch_lightning as pl

    HAS_PYTORCH_LIGHTNING = True
except ImportError:
    HAS_PYTORCH_LIGHTNING = False


logger = logging.getLogger(__name__)


class SeedManager:
    """
    Centralized random seed management for reproducible experiments.

    This class provides hierarchical seed management where a master seed generates
    deterministic component-specific seeds. It supports all major randomness libraries
    and provides safe multiprocessing seed coordination.

    Example:
        >>> seed_manager = SeedManager(master_seed=42)
        >>> seed_manager.set_global_seeds()
        >>>
        >>> # Get component-specific seeds
        >>> datamodule_seed = seed_manager.get_component_seed("datamodule")
        >>> preprocessing_seed = seed_manager.get_component_seed("preprocessing")
        >>>
        >>> # Use temporary seeding for specific operations
        >>> with seed_manager.temporary_seed("validation_pool"):
        >>>     selected_basins = random.sample(basins, k)
    """

    # Predefined component categories for the hydrological forecasting pipeline
    COMPONENT_SEEDS = {
        "datamodule_setup": "Basin shuffling and validation pool creation",
        "training_chunks": "Chunk advancement and reshuffling operations",
        "preprocessing_unified": "Basin selection for unified pipeline fitting",
        "data_loading": "DataLoader samplers and worker processes",
        "sequence_operations": "Training sequence shuffling and validation",
        "model_training": "Model initialization and training randomness",
        "hyperparameter_tuning": "Random search and optimization",
    }

    def __init__(self, master_seed: int | None = None):
        """
        Initialize the SeedManager with an optional master seed.

        Args:
            master_seed: The master seed for all random operations. If None,
                        random operations will use their default behavior.
        """
        self.master_seed = master_seed
        self._component_seeds: dict[str, int] = {}
        self._seed_state_stack: list = []
        self._lock = threading.RLock()

        if master_seed is not None:
            self._generate_component_seeds()
            logger.info(f"SeedManager initialized with master seed: {master_seed}")
        else:
            logger.info("SeedManager initialized without master seed (non-deterministic mode)")

    def _generate_component_seeds(self) -> None:
        """Generate deterministic component seeds from the master seed."""
        if self.master_seed is None:
            return

        for component in self.COMPONENT_SEEDS:
            self._component_seeds[component] = self._derive_seed(component)

    def _derive_seed(self, component: str, operation: str = "", worker_id: int | None = None) -> int:
        """
        Derive a deterministic seed for a specific component and operation.

        Args:
            component: The component name (e.g., "datamodule", "preprocessing")
            operation: Optional specific operation name for finer granularity
            worker_id: Optional worker ID for multiprocessing scenarios

        Returns:
            A deterministic integer seed derived from the master seed
        """
        if self.master_seed is None:
            return random.randint(0, 2**31 - 1)

        # Create a unique string identifier for this seed derivation
        seed_string = f"{self.master_seed}:{component}"
        if operation:
            seed_string += f":{operation}"
        if worker_id is not None:
            seed_string += f":worker_{worker_id}"

        # Use SHA-256 to generate a deterministic hash
        hash_object = hashlib.sha256(seed_string.encode("utf-8"))
        hash_hex = hash_object.hexdigest()

        # Convert to integer and ensure it's within valid seed range
        seed_int = int(hash_hex[:8], 16) % (2**31 - 1)
        return seed_int

    def set_global_seeds(self) -> None:
        """
        Set global seeds for all available randomness libraries.

        Uses a hierarchical approach to avoid redundant seeding:
        - When PyTorch Lightning is available: Use pl.seed_everything() as the primary
          seeding mechanism since it comprehensively handles Python, NumPy, PyTorch, and CUDA
        - When PyTorch Lightning is unavailable: Fall back to direct calls to individual
          library seeding functions
        """
        if self.master_seed is None:
            logger.warning("Cannot set global seeds: no master seed provided")
            return

        with self._lock:
            if HAS_PYTORCH_LIGHTNING:
                # Use PyTorch Lightning's comprehensive seeding
                pl.seed_everything(self.master_seed, workers=True)
                logger.debug(f"Set PyTorch Lightning seed (comprehensive): {self.master_seed}")
            else:
                # Fall back to manual seeding for each library
                random.seed(self.master_seed)
                logger.debug(f"Set Python random seed: {self.master_seed}")

                if HAS_NUMPY:
                    np.random.seed(self.master_seed)
                    logger.debug(f"Set NumPy random seed: {self.master_seed}")

                if HAS_TORCH:
                    torch.manual_seed(self.master_seed)
                    logger.debug(f"Set PyTorch manual seed: {self.master_seed}")

                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(self.master_seed)
                        logger.debug(f"Set PyTorch CUDA seed: {self.master_seed}")

    def get_component_seed(self, component: str) -> int:
        """
        Get the deterministic seed for a specific component.

        Args:
            component: The component name (must be in COMPONENT_SEEDS)

        Returns:
            The deterministic seed for this component

        Raises:
            ValueError: If component is not recognized
        """
        if component not in self.COMPONENT_SEEDS:
            raise ValueError(
                f"Unknown component '{component}'. Available components: {list(self.COMPONENT_SEEDS.keys())}"
            )

        if self.master_seed is None:
            logger.warning(f"No master seed set, returning random seed for component '{component}'")
            return random.randint(0, 2**31 - 1)

        return self._component_seeds[component]

    def get_operation_seed(self, component: str, operation: str) -> int:
        """
        Get a deterministic seed for a specific operation within a component.

        Args:
            component: The component name
            operation: The specific operation name

        Returns:
            The deterministic seed for this operation
        """
        return self._derive_seed(component, operation)

    def get_worker_seed(self, worker_id: int, component: str = "data_loading") -> int:
        """
        Get a deterministic seed for a multiprocessing worker.

        Args:
            worker_id: The worker process ID
            component: The component this worker belongs to

        Returns:
            A unique deterministic seed for this worker
        """
        return self._derive_seed(component, "worker", worker_id)

    @contextmanager
    def temporary_seed(self, operation: str, component: str = "sequence_operations") -> Generator[int, None, None]:
        """
        Context manager for temporary seeding that doesn't affect global state.

        Args:
            operation: The operation name for seed derivation
            component: The component this operation belongs to

        Yields:
            The temporary seed being used

        Example:
            >>> with seed_manager.temporary_seed("validation_pool") as temp_seed:
            >>>     selected_basins = random.sample(basins, k)
        """
        # Save current state
        python_state = random.getstate()
        numpy_state = None
        torch_state = None

        if HAS_NUMPY:
            numpy_state = np.random.get_state()

        if HAS_TORCH:
            torch_state = torch.get_rng_state()

        try:
            # Set temporary seed
            temp_seed = self.get_operation_seed(component, operation)

            random.seed(temp_seed)
            if HAS_NUMPY:
                np.random.seed(temp_seed)
            if HAS_TORCH:
                torch.manual_seed(temp_seed)

            logger.debug(f"Applied temporary seed {temp_seed} for operation '{operation}' in component '{component}'")
            yield temp_seed

        finally:
            # Restore previous state
            random.setstate(python_state)
            if HAS_NUMPY and numpy_state is not None:
                np.random.set_state(numpy_state)
            if HAS_TORCH and torch_state is not None:
                torch.set_rng_state(torch_state)

            logger.debug(f"Restored random state after temporary seed for operation '{operation}'")

    def save_state(self) -> dict[str, Any]:
        """
        Save current random state for all libraries.

        Returns:
            Dictionary containing the current state of all random libraries
        """
        state = {
            "master_seed": self.master_seed,
            "component_seeds": self._component_seeds.copy(),
            "python_state": random.getstate(),
        }

        if HAS_NUMPY:
            state["numpy_state"] = np.random.get_state()

        if HAS_TORCH:
            state["torch_state"] = torch.get_rng_state()
            if torch.cuda.is_available():
                state["cuda_state"] = torch.cuda.get_rng_state_all()

        return state

    def restore_state(self, state: dict[str, Any]) -> None:
        """
        Restore random state from a saved state dictionary.

        Args:
            state: State dictionary returned by save_state()
        """
        with self._lock:
            self.master_seed = state["master_seed"]
            self._component_seeds = state["component_seeds"].copy()

            random.setstate(state["python_state"])

            if HAS_NUMPY and "numpy_state" in state:
                np.random.set_state(state["numpy_state"])

            if HAS_TORCH and "torch_state" in state:
                torch.set_rng_state(state["torch_state"])
                if torch.cuda.is_available() and "cuda_state" in state:
                    torch.cuda.set_rng_state_all(state["cuda_state"])

    def worker_init_fn(self, worker_id: int) -> None:
        """
        Worker initialization function for PyTorch DataLoaders.

        This function should be passed to the DataLoader's worker_init_fn parameter
        to ensure each worker process gets a unique but deterministic seed.

        Args:
            worker_id: The worker process ID (automatically provided by DataLoader)

        Example:
            >>> DataLoader(dataset, worker_init_fn=seed_manager.worker_init_fn)
        """
        worker_seed = self.get_worker_seed(worker_id)

        random.seed(worker_seed)
        if HAS_NUMPY:
            np.random.seed(worker_seed)
        if HAS_TORCH:
            torch.manual_seed(worker_seed)

        logger.debug(f"Initialized worker {worker_id} with seed {worker_seed}")

    def get_info(self) -> dict[str, Any]:
        """
        Get information about the current seed manager state.

        Returns:
            Dictionary with seed manager configuration and state information
        """
        return {
            "master_seed": self.master_seed,
            "has_master_seed": self.master_seed is not None,
            "component_seeds": self._component_seeds.copy(),
            "available_libraries": {
                "numpy": HAS_NUMPY,
                "torch": HAS_TORCH,
                "pytorch_lightning": HAS_PYTORCH_LIGHTNING,
                "cuda": HAS_TORCH and torch.cuda.is_available() if HAS_TORCH else False,
            },
            "component_descriptions": self.COMPONENT_SEEDS.copy(),
        }

    def __getstate__(self):
        """Custom method for pickling - exclude the RLock."""
        state = self.__dict__.copy()
        # Remove the unpicklable RLock
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        """Custom method for unpickling - recreate the RLock."""
        self.__dict__.update(state)
        # Recreate the RLock
        self._lock = threading.RLock()


# Global instance for convenience
_global_seed_manager: SeedManager | None = None


def get_global_seed_manager() -> SeedManager | None:
    """Get the global seed manager instance."""
    return _global_seed_manager


def set_global_seed_manager(seed_manager: SeedManager) -> None:
    """Set the global seed manager instance."""
    global _global_seed_manager
    _global_seed_manager = seed_manager


def init_global_seed_manager(master_seed: int | None = None) -> SeedManager:
    """
    Initialize and set the global seed manager.

    Args:
        master_seed: The master seed for reproducible experiments

    Returns:
        The initialized SeedManager instance
    """
    seed_manager = SeedManager(master_seed)
    set_global_seed_manager(seed_manager)
    return seed_manager
