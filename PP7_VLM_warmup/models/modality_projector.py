"""Exercise stub for the PP7 modality projector."""

import torch.nn as nn


class ModalityProjector(nn.Module):
    """Student implementation target for the modality projector exercise."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512):
        """Placeholder initializer for the exercise implementation.

        Args:
            *args: Positional arguments the student-defined projector may need.
            **kwargs: Keyword arguments the student-defined projector may need.
        """
        super().__init__()  
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projector(x)