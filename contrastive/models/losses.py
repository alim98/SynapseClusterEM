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
        # Get batch size
        batch_size = z_i.shape[0]
        
        # Cosine similarity between all possible pairs
        representations = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, projection_dim]
        
        # Normalize the representations: important for cosine similarity
        representations = F.normalize(representations, p=2, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T)  # [2*batch_size, 2*batch_size]
        
        # The diagonal elements are self-similarities, which we don't use
        # We set the diagonal to a very low value to ignore them in the softmax
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        similarity_matrix.masked_fill_(mask, -float('inf'))
        
        # The positives are the cross-similarity between z_i and z_j (and vice versa)
        # For example, if batch_size=4, the positives are at indices: (0,4), (1,5), (2,6), (3,7), (4,0), (5,1), (6,2), (7,3)
        positives = torch.cat([
            similarity_matrix[range(batch_size), range(batch_size, 2 * batch_size)],
            similarity_matrix[range(batch_size, 2 * batch_size), range(batch_size)]
        ])
        
        # All other similarity scores are negatives
        # We create a mask to exclude the positives and the diagonal
        # The mask has 1s for all negatives and 0s for positives and diagonal
        negative_mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=self.device)
        
        # Set diagonal to false (exclude self-similarity)
        negative_mask.fill_diagonal_(False)
        
        # Set positives to false (exclude positives)
        for i in range(batch_size):
            negative_mask[i, i + batch_size] = False
            negative_mask[i + batch_size, i] = False
        
        # Get all negative similarities
        negatives = similarity_matrix[negative_mask].view(2 * batch_size, -1)  # [2*batch_size, 2*batch_size-2]
        
        # Concatenate positive similarities with all negative similarities
        logits = torch.cat([positives.view(-1, 1), negatives], dim=1)  # [2*batch_size, 2*batch_size-1]
        
        # Scale by temperature
        logits /= self.temperature
        
        # The labels are zeros (indicating that the positive pair is at index 0)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=self.device)
        
        # Calculate the cross-entropy loss
        loss = self.criterion(logits, labels)
        
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