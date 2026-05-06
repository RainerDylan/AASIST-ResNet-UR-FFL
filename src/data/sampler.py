import torch
from torch.utils.data import WeightedRandomSampler

def create_weighted_sampler(labels):
    """
    Creates a WeightedRandomSampler to perfectly balance batches 
    for highly imbalanced datasets like ASVspoof.
    
    Args:
        labels (list or numpy array): The ground truth labels for the dataset.
        
    Returns:
        WeightedRandomSampler: A PyTorch sampler that yields a 50/50 class balance.
    """
    # Convert labels to tensor if they aren't already
    if not isinstance(labels, torch.Tensor):
        labels_tensor = torch.tensor(labels, dtype=torch.long)
    else:
        labels_tensor = labels

    # Count the occurrences of each class (0: Deepfake, 1: Bonafide)
    class_counts = torch.bincount(labels_tensor)
    total_samples = len(labels_tensor)
    
    # Calculate the weight for each class (Total / Class Count)
    # The rare class (Bonafide) gets a massive weight, the common class gets a tiny weight.
    class_weights = total_samples / class_counts.float()
    
    # Assign the corresponding class weight to every single sample in the dataset
    sample_weights = [class_weights[label] for label in labels_tensor]
    
    # Create the sampler. replacement=True allows the rare class to be sampled multiple times per epoch
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=total_samples, 
        replacement=True
    )
    
    return sampler