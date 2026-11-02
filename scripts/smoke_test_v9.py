"""V9 Smoke Test: Verify architecture and V8 weight compatibility."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

def main():
    from src.perception.perception_module import NeuroTopoPerception
    
    print("=== V9 SMOKE TEST ===")
    
    # Test 1: Model creation
    model = NeuroTopoPerception()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model created. Params: {n_params:,}")
    
    # Test 2: V8 weight loading (direct, no remap!)
    state = torch.load("runs_perception_v8/best.pt", map_location="cpu", weights_only=False)
    v8_weights = state["model"]
    
    # Remap logic identical to train_perception.py
    remap = {
        "up.0.weight": "up1.weight", "up.0.bias": "up1.bias",
        "up.2.weight": "up2.weight", "up.2.bias": "up2.bias",
        "up.4.weight": "up3.weight", "up.4.bias": "up3.bias",
    }
    mapped_weights = {}
    for k, v in v8_weights.items():
        mapped_k = remap.get(k, k)
        mapped_weights[mapped_k] = v
        
    missing, unexpected = model.load_state_dict(mapped_weights, strict=False)
    print(f"[OK] V8 weights loaded: {len(missing)} missing, {len(unexpected)} unexpected")
    for k in missing:
        print(f"     Missing (new V9): {k}")
    for k in unexpected:
        print(f"     Unexpected (V8 only): {k}")
    
    # Verify only skip layers are missing
    assert all("skip" in k for k in missing), f"Non-skip layers missing: {[k for k in missing if 'skip' not in k]}"
    assert len(unexpected) == 0, f"Unexpected layers: {unexpected}"
    print(f"[OK] Only skip connection weights are new (expected)")
    
    # Test 3: Forward pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    x = torch.randn(2, 3, 320, 448).to(device)
    img_sizes = torch.tensor([[448, 320], [448, 320]]).to(device)
    with torch.no_grad():
        hm, pts, _, adj, types = model(x, img_sizes)
    print(f"[OK] Forward: hm={hm.shape} pts={pts.shape} adj={adj.shape}")
    
    # Test 4: VRAM
    if device == "cuda":
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[OK] VRAM: {vram_mb:.0f} MB ({vram_mb/1024:.2f} GB)")
    
    # Test 5: Discriminative LR
    backbone_params = list(model.features.parameters())
    backbone_ids = set(id(p) for p in backbone_params)
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    print(f"[OK] Backbone: {sum(p.numel() for p in backbone_params):,} params")
    print(f"[OK] Head: {sum(p.numel() for p in head_params):,} params")
    
    # Test 6: Skip connections are zero-init
    for name in ["skip1", "skip2", "skip3"]:
        skip = getattr(model, name)
        w = skip.proj.weight.data
        assert w.abs().max().item() == 0.0, f"{name} not zero-init!"
        print(f"[OK] {name}: zero-initialized")
    
    # Test 7: Gradient flow
    model.train()
    x = torch.randn(1, 3, 320, 448).to(device)
    img_sizes = torch.tensor([[448, 320]]).to(device)
    hm, pts, _, adj, types = model(x, img_sizes)
    loss = hm.sum()  # dummy loss
    loss.backward()
    
    bb_grad_norm = sum(p.grad.norm().item() for p in backbone_params if p.grad is not None)
    head_grad_norm = sum(p.grad.norm().item() for p in head_params if p.grad is not None)
    print(f"[OK] Gradient norms: backbone={bb_grad_norm:.4f}, head={head_grad_norm:.4f}")
    
    print("\n=== ALL 7 TESTS PASSED ===")

if __name__ == "__main__":
    main()
