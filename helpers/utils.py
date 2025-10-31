import torch

def count_parameters(model):
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_model_info(model):
    # Test with different input sizes
    print("="*60)
    print("INATNet")
    print("="*60)

    trainable, total = count_parameters(model)
    print(f"\nTrainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")
    print(f"Frozen (SRM) parameters: {total - trainable:,}")
    
    print("\n" + "="*60)
    print("Architecture highlights:")
    print("="*60)
    print("✓ 30 SRM high-pass filters (frozen)")
    print("✓ TLU + Absolute Value layer")
    print("✓ Deep residual architecture")
    print("✓ Separable convolutions for efficiency")
    print("✓ Inception attention block near the end")
    print("✓ All 4 attention mechanisms (SE, CBAM, GC, Triplet)")
    print("="*60)