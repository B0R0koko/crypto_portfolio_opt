from abc import ABC, abstractmethod
from typing import List, Dict, Any

import numpy as np


class PortfolioOptimizer(ABC):

    @abstractmethod
    def target_function(self, W: np.ndarray, R: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Define target function for scipy.optimize.minimize to minimize"""

    @abstractmethod
    def get_constraints(self) -> List[Dict[str, Any]]:
        """Define constraints for optimization"""
