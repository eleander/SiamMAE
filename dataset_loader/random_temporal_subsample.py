import torch

def random_temporal_subsample(video, min_gap, max_gap, temporal_dim=-3, repeated_sampling=1):
    # Return first and randomly sampled frame
    t = video.shape[temporal_dim]

    # Randomly sample a gap size
    gap = torch.randint(min_gap, max_gap, (repeated_sampling,))

    # Sort gap sizes
    gap = torch.sort(gap)[0]
    # print(f"gap: {gap}")
    
    # return tensor with first and (repeated_sampling)x randomly sampled frames
    return torch.index_select(video, temporal_dim, torch.cat((torch.zeros(1, dtype=torch.long), gap)))

# repeated_sampling: repeated sampling factor as in https://arxiv.org/pdf/2205.09113.pdf
class RandomTemporalSubsample(torch.nn.Module):
    def __init__(self, min_gap, max_gap, temporal_dim=-3, repeated_sampling=1):
        super().__init__()
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.temporal_dim = temporal_dim
        self.repeated_sampling = repeated_sampling

    def forward(self, x):
        return random_temporal_subsample(
            x, self.min_gap, self.max_gap, self.temporal_dim, self.repeated_sampling
        )
    
class LinearTemporalSubsample(torch.nn.Module):
    def __init__(self, min_gap, max_gap, temporal_dim=-3, repeated_sampling=1):
        super().__init__()
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.temporal_dim = temporal_dim
        self.repeated_sampling = repeated_sampling

        print(f"min_gap: {min_gap}, max_gap: {max_gap}, temporal_dim: {temporal_dim}, repeated_sampling: {repeated_sampling}")
        
    def forward(self, x):
        max_gap = min(self.max_gap, x.shape[self.temporal_dim] - 1)
        gap = torch.linspace(self.min_gap, max_gap, self.repeated_sampling, dtype=torch.long)
        return torch.index_select(x, self.temporal_dim, torch.cat((torch.zeros(1, dtype=torch.long), gap)))