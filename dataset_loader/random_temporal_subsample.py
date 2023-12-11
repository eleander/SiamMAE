import torch

def random_temporal_subsample(video, min_gap, max_gap, temporal_dim=-3, repeated_sampling=1):
    """
    Randomly subsamples a video tensor along the temporal dimension.

    Parameters:
    video (torch.Tensor): The video tensor to subsample.
    min_gap (int): The minimum gap size between sampled frames.
    max_gap (int): The maximum gap size between sampled frames.
    temporal_dim (int): The dimension of the tensor representing time. Default is -3.
    repeated_sampling (int): The number of times to repeat the sampling process. Default is 1.

    Returns:
    torch.Tensor: The subsampled video tensor.
    """
    # Get the size of the temporal dimension
    t = video.shape[temporal_dim]

    # Randomly sample a gap size
    gap = torch.randint(min_gap, max_gap, (repeated_sampling,))

    # Sort gap sizes
    gap = torch.sort(gap)[0]
    
    # Return tensor with first and (repeated_sampling)x randomly sampled frames
    return torch.index_select(video, temporal_dim, torch.cat((torch.zeros(1, dtype=torch.long), gap)))

class RandomTemporalSubsample(torch.nn.Module):
    """
    A PyTorch module that randomly subsamples a video tensor along the temporal dimension.

    Parameters:
    min_gap (int): The minimum gap size between sampled frames.
    max_gap (int): The maximum gap size between sampled frames.
    temporal_dim (int): The dimension of the tensor representing time. Default is -3.
    repeated_sampling (int): The number of times to repeat the sampling process. Default is 1.
    """
    def __init__(self, min_gap, max_gap, temporal_dim=-3, repeated_sampling=1):
        super().__init__()
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.temporal_dim = temporal_dim
        self.repeated_sampling = repeated_sampling

    def forward(self, x):
        """
        Applies the random temporal subsampling to a video tensor.

        Parameters:
        x (torch.Tensor): The video tensor to subsample.

        Returns:
        torch.Tensor: The subsampled video tensor.
        """
        return random_temporal_subsample(
            x, self.min_gap, self.max_gap, self.temporal_dim, self.repeated_sampling
        )