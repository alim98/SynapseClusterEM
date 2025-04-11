import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NTXentLoss(nn.Module):
    """
    NT-Xent loss for contrastive learning as used in SimCLR.
    
    This loss function encourages representations of positive pairs (augmented views of the same image)
    to be similar, while pushing representations of negative pairs (augmented views of different images)
    further apart.
    """
    def __init__(self, temperature=0.07, device='cuda'):
        """
        Initialize the NT-Xent loss function.
        
        Args:
            temperature (float): Temperature parameter controls the sharpness of the softmax distribution
            device (str): Device to compute the loss on ('cuda' or 'cpu')
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
    
    def forward(self, z_i, z_j):
        """
        Compute the NT-Xent loss for the batch.
        
        Args:
            z_i (torch.Tensor): Projections of the first augmented views [batch_size, projection_dim]
            z_j (torch.Tensor): Projections of the second augmented views [batch_size, projection_dim]
            
        Returns:
            torch.Tensor: Scalar NT-Xent loss
        """
        batch_size = z_i.shape[0]
        
        # Debug prints for input
        print(f"Input shapes - z_i: {z_i.shape}, z_j: {z_j.shape}")
        print(f"Input ranges - z_i: [{z_i.min():.3f}, {z_i.max():.3f}], z_j: [{z_j.min():.3f}, {z_j.max():.3f}]")
        
        # Normalize the projections
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Debug prints after normalization
        print(f"After normalization - z_i: [{z_i.min():.3f}, {z_i.max():.3f}], z_j: [{z_j.min():.3f}, {z_j.max():.3f}]")
        
        # Compute similarity matrix
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        # Debug prints for similarity matrix
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        print(f"Similarity matrix range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
        
        # Create mask for positive pairs
        mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=bool, device=self.device)
        mask[torch.arange(2 * batch_size, device=self.device), torch.arange(2 * batch_size, device=self.device)] = True
        mask = mask.float()
        
        # Debug prints for mask
        print(f"Mask shape: {mask.shape}")
        print(f"Number of positive pairs: {mask.sum().item()}")
        
        # Compute loss
        positives = similarity_matrix[mask.bool()].view(2 * batch_size, 1)
        negatives = similarity_matrix[~mask.bool()].view(2 * batch_size, -1)
        
        # Debug prints for positives and negatives
        print(f"Positives shape: {positives.shape}, range: [{positives.min():.3f}, {positives.max():.3f}]")
        print(f"Negatives shape: {negatives.shape}, range: [{negatives.min():.3f}, {negatives.max():.3f}]")
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * batch_size, device=self.device).long()
        
        # Debug prints for logits and labels
        print(f"Logits shape: {logits.shape}, range: [{logits.min():.3f}, {logits.max():.3f}]")
        print(f"Labels shape: {labels.shape}, unique values: {torch.unique(labels).tolist()}")
        
        loss = self.criterion(logits / self.temperature, labels)
        
        # Debug print for final loss
        print(f"Final loss: {loss.item():.6f}")
        
        return loss


class SimplifiedNTXentLoss(nn.Module):
    """
    A simplified version of the NT-Xent loss for contrastive learning.
    """
    def __init__(self, temperature=0.07):
        """
        Initialize the simplified NT-Xent loss function.
        
        Args:
            temperature (float): Temperature parameter
        """
        super(SimplifiedNTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
    
    def forward(self, z_i, z_j):
        """
        Compute the simplified NT-Xent loss for the batch.
        
        Args:
            z_i (torch.Tensor): First set of embeddings [batch_size, projection_dim]
            z_j (torch.Tensor): Second set of embeddings [batch_size, projection_dim]
            
        Returns:
            torch.Tensor: Scalar NT-Xent loss
        """
        # Normalize the embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Get batch size
        batch_size = z_i.shape[0]
        
        # Compute similarity matrix
        # The similarity matrix has shape [batch_size, batch_size]
        # where sim[i, j] is the similarity between the i-th sample in z_i and the j-th sample in z_j
        similarity_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        
        # The positive examples are on the diagonal of the similarity matrix
        # The labels are the indices of the diagonal elements
        labels = torch.arange(batch_size, device=z_i.device)
        
        # Calculate the loss (cross-entropy between similarity_matrix and labels)
        # This encourages the diagonal elements (positive pairs) to have higher values
        loss = self.criterion(similarity_matrix, labels) + self.criterion(similarity_matrix.T, labels)
        loss = loss / 2.0  # Average the loss over both directions
        
        return loss 