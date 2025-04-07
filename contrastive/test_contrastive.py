"""
Test script for contrastive learning implementation.
This script tests the core components of the contrastive learning pipeline.
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import contrastive modules
from contrastive.utils.config import config
from contrastive.models.contrastive_model import initialize_contrastive_model
from contrastive.models.losses import NTXentLoss
from contrastive.data.augmentations import ContrastiveAugmenter, ToTensor3D

# Import from synapse pipeline
from synapse_pipeline import SynapsePipeline
from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor

# Set small batch size and epochs for testing
config.batch_size = 4
config.num_epochs = 2
config.learning_rate = 0.001

def test_augmentations():
    """Test the 3D augmentations."""
    print("Testing augmentations...")
    
    # Create a test volume
    volume = np.random.rand(80, 80, 80)
    
    # Create augmenter
    augmenter = ContrastiveAugmenter(config)
    
    # Apply augmentations
    augmented1, augmented2 = augmenter(volume)
    
    # Check shapes
    assert augmented1.shape == volume.shape, f"Augmented shape {augmented1.shape} != original shape {volume.shape}"
    assert augmented2.shape == volume.shape, f"Augmented shape {augmented2.shape} != original shape {volume.shape}"
    
    # Check that augmentations are different
    diff = np.abs(augmented1 - augmented2).mean()
    assert diff > 0.01, f"Augmentations are too similar: {diff}"
    
    print("✓ Augmentations test passed!")
    return True

def test_model():
    """Test the contrastive model."""
    print("Testing contrastive model...")
    
    # Initialize model
    model = initialize_contrastive_model(config)
    
    # Create test input
    test_input = torch.randn(2, 1, 80, 80, 80)
    
    # Test forward pass
    projections = model(test_input)
    
    # Check output shape
    assert projections.shape == (2, config.proj_dim), f"Output shape {projections.shape} != expected shape (2, {config.proj_dim})"
    
    # Test feature extraction
    features = model.extract_features(test_input)
    assert features.ndim == 2, f"Features should be 2D, got {features.ndim}D"
    
    # Test layer-specific feature extraction
    layer_features = model.extract_features(test_input, layer_num=10)
    assert layer_features.ndim >= 2, f"Layer features should be at least 2D, got {layer_features.ndim}D"
    
    print("✓ Model test passed!")
    return True

def test_loss():
    """Test the NT-Xent loss."""
    print("Testing NT-Xent loss...")
    
    # Initialize loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = NTXentLoss(temperature=config.temperature, device=device)
    
    # Create test embeddings
    batch_size = 4
    z_i = torch.randn(batch_size, config.proj_dim, device=device)
    z_j = torch.randn(batch_size, config.proj_dim, device=device)
    
    # Make some pairs more similar (positive pairs)
    for i in range(batch_size):
        # Make positive pairs more similar
        z_j[i] = z_i[i] + 0.1 * torch.randn_like(z_i[i])
    
    # Calculate loss
    loss = criterion(z_i, z_j)
    
    # Check loss is a scalar
    assert loss.ndim == 0, f"Loss should be a scalar, got shape {loss.shape}"
    
    # Check loss is positive
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    
    print(f"✓ Loss test passed! Loss value: {loss.item():.4f}")
    return True

def test_data_loading():
    """Test data loading with a small subset."""
    print("Testing data loading...")
    
    # Create pipeline to load data
    pipeline = SynapsePipeline(config)
    
    # Load a small amount of data
    try:
        vol_data_dict, syn_df = pipeline.load_and_prepare_data()
        
        # Take first 10 samples
        if len(syn_df) > 10:
            syn_df = syn_df.iloc[:10]
        
        # Initialize processor
        processor = Synapse3DProcessor(size=config.size)
        
        # Create dataloader (without calling the create_contrastive_dataloader function)
        from contrastive.data.dataset import ContrastiveDataset
        from torch.utils.data import DataLoader
        
        dataset = ContrastiveDataset(
            vol_data_dict=vol_data_dict,
            synapse_df=syn_df,
            processor=processor,
            segmentation_type=config.segmentation_type,
            config=config,
            subvol_size=config.subvol_size,
            num_frames=config.num_frames,
            alpha=config.alpha,
            normalize_across_volume=True,
        )
        
        # Custom collate function
        def collate_fn(batch):
            batch = [item for item in batch if item is not None]
            if len(batch) == 0:
                return None
            
            views, syn_infos, bbox_names = zip(*batch)
            view1s, view2s = zip(*views)
            view1s = torch.stack(view1s)
            view2s = torch.stack(view2s)
            
            return (view1s, view2s), syn_infos, bbox_names
        
        dataloader = DataLoader(
            dataset,
            batch_size=min(2, len(dataset)),
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        # Get a batch
        for batch in dataloader:
            if batch is None:
                continue
            
            (view1, view2), syn_infos, bbox_names = batch
            
            # Check shapes
            assert view1.ndim == 4, f"View1 should be 4D, got {view1.ndim}D"
            assert view2.ndim == 4, f"View2 should be 4D, got {view2.ndim}D"
            assert view1.shape == view2.shape, f"Views should have same shape, got {view1.shape} != {view2.shape}"
            
            print(f"Loaded batch with shape: {view1.shape}")
            break
        
        print("✓ Data loading test passed!")
        return True
    
    except Exception as e:
        print(f"Data loading test failed: {e}")
        return False

def test_contrastive_training():
    """Test contrastive training for one iteration."""
    print("Testing contrastive training...")
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        model = initialize_contrastive_model(config)
        model = model.to(device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Create loss function
        criterion = NTXentLoss(temperature=config.temperature, device=device)
        
        # Create test batch
        batch_size = 4
        view1 = torch.randn(batch_size, 1, 80, 80, 80, device=device)
        view2 = torch.randn(batch_size, 1, 80, 80, 80, device=device)
        
        # Make it a realistic test by making view2 similar to view1
        view2 = view1 + 0.1 * torch.randn_like(view1)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        z1 = model(view1)
        z2 = model(view2)
        
        # Calculate loss
        loss = criterion(z1, z2)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_gradients, "No gradients were computed"
        
        # Optimize
        optimizer.step()
        
        print(f"✓ Contrastive training test passed! Loss: {loss.item():.4f}")
        return True
    
    except Exception as e:
        print(f"Contrastive training test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("="*50)
    print("Running Contrastive Learning tests")
    print("="*50)
    
    tests = [
        test_augmentations,
        test_model,
        test_loss,
        test_data_loading,
        test_contrastive_training
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append(False)
    
    # Report overall results
    print("\n" + "="*50)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("="*50)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")

if __name__ == "__main__":
    run_all_tests() 