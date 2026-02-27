"""
Complex Weight Proof of Concept v3 — Faithful to SRT/CRT Architecture

Key differences from v2:
1. Quaternion Hamilton product (4D structured rotation, not 2D complex multiply)
   - 4 weight matrices reused in 16 combinations = structured coupling for free
2. Bimodal: Real component = measurement/gate on imaginary waveform
   - NOT symmetric like standard quaternion networks
   - Real channel learns WHAT to select, imaginary channels carry the signal
3. D→H cycle: Differentiation layer (expand) → Harmonization layer (compress)
   - D: x + ReLU(Wx + b)  — generates complexity
   - H: x - σ(W_H·x) + tanh(W_S·x)  — damps dissonance, projects coherence

Source theory: CRT §12.2, HCB2_vm.py, differentiation.py, harmonization.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import math

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio

print(f"Running on: {DEVICE}")

# --- Data Prep ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================================
# BASELINE: Standard Scalar Net
# ==========================================
class ScalarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ==========================================
# QUATERNION LINEAR: Hamilton product
# ==========================================
class QuaternionLinear(nn.Module):
    """
    Linear layer using quaternion Hamilton product.
    
    4 weight matrices (W_a, W_b, W_c, W_d) are reused in 16 combinations
    through the Hamilton product. This gives structured coupling between
    components that scalar weights can't achieve.
    
    From quaternion.rs: q1*q2 uses Hamilton product
    i² = j² = k² = ijk = -1
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        # Golden-ratio-aware initialization (from ResonantLinear)
        # Scale by 1/sqrt(4*in) for quaternion (4 components contributing)
        scale = 1.0 / math.sqrt(4.0 * in_features)
        
        # Real component (will become measurement channel)
        self.W_a = nn.Parameter(torch.randn(out_features, in_features) * scale)
        # Imaginary components (waveform channels)
        self.W_b = nn.Parameter(torch.randn(out_features, in_features) * scale)
        self.W_c = nn.Parameter(torch.randn(out_features, in_features) * scale)
        self.W_d = nn.Parameter(torch.randn(out_features, in_features) * scale)
        
        # Bias per component
        self.bias_a = nn.Parameter(torch.zeros(out_features))
        self.bias_b = nn.Parameter(torch.zeros(out_features))
        self.bias_c = nn.Parameter(torch.zeros(out_features))
        self.bias_d = nn.Parameter(torch.zeros(out_features))

    def forward(self, r, i, j, k):
        """
        Hamilton product: (r + bi + cj + dk)(a + Bi + Cj + Dk)
        
        out_r = r·Wa - i·Wb - j·Wc - k·Wd
        out_i = r·Wb + i·Wa + j·Wd - k·Wc
        out_j = r·Wc - i·Wd + j·Wa + k·Wb
        out_k = r·Wd + i·Wc - j·Wb + k·Wa
        """
        out_r = (F.linear(r, self.W_a) - F.linear(i, self.W_b) -
                 F.linear(j, self.W_c) - F.linear(k, self.W_d) + self.bias_a)
        
        out_i = (F.linear(r, self.W_b) + F.linear(i, self.W_a) +
                 F.linear(j, self.W_d) - F.linear(k, self.W_c) + self.bias_b)
        
        out_j = (F.linear(r, self.W_c) - F.linear(i, self.W_d) +
                 F.linear(j, self.W_a) + F.linear(k, self.W_b) + self.bias_c)
        
        out_k = (F.linear(r, self.W_d) + F.linear(i, self.W_c) -
                 F.linear(j, self.W_b) + F.linear(k, self.W_a) + self.bias_d)
        
        return out_r, out_i, out_j, out_k


# ==========================================
# BIMODAL ACTIVATION: Real = measurement, Imaginary = waveform
# ==========================================
class BimodalActivation(nn.Module):
    """
    The real component acts as MEASUREMENT on the imaginary waveform.
    
    This is NOT symmetric like standard quaternion activations.
    Real channel: sigmoid → becomes a gate (measurement operator)
    Imaginary channels: the signal being gated (waveform)
    
    Output imaginary = gate * imaginary (measurement collapses waveform)
    Output real = the gate itself (measurement value passes forward)
    
    From HCB2_vm.py theory: "The real number components would act as 
    the measurement applied to the complex number waveforms"
    """
    def __init__(self, features):
        super().__init__()
        # Learnable measurement bias (from ModReLU concept)
        self.measurement_bias = nn.Parameter(torch.zeros(features))
    
    def forward(self, r, i, j, k):
        # Real component becomes measurement gate
        gate = torch.sigmoid(r + self.measurement_bias.unsqueeze(0))
        
        # Measurement applied to waveform components
        out_i = gate * i
        out_j = gate * j
        out_k = gate * k
        
        # Real channel carries the measurement value forward
        out_r = gate
        
        return out_r, out_i, out_j, out_k


# ==========================================
# DIFFERENTIATION LAYER: Expand complexity
# ==========================================
class QuaternionDifferentiation(nn.Module):
    """
    D̂[x] = x + α·act(Q_linear(x))
    
    Generates complexity through residual expansion.
    ReLU on waveform magnitude introduces nonlinearity.
    Bimodal gating selects what complexity to keep.
    
    From differentiation.py: "Increases representational complexity (Fire/novelty)"
    """
    def __init__(self, in_features, out_features, alpha=1.0):
        super().__init__()
        self.qlinear = QuaternionLinear(in_features, out_features)
        self.activation = BimodalActivation(out_features)
        self.alpha = alpha
        self.match_dims = (in_features == out_features)
    
    def forward(self, r, i, j, k):
        # Quaternion linear transform
        dr, di, dj, dk = self.qlinear(r, i, j, k)
        
        # Bimodal activation (measurement gates waveform)
        dr, di, dj, dk = self.activation(dr, di, dj, dk)
        
        # Residual: D̂[x] = x + α·transform(x)
        if self.match_dims:
            out_r = r + self.alpha * dr
            out_i = i + self.alpha * di
            out_j = j + self.alpha * dj
            out_k = k + self.alpha * dk
            return out_r, out_i, out_j, out_k
        else:
            return dr, di, dj, dk


# ==========================================
# HARMONIZATION LAYER: Damp dissonance, project coherence
# ==========================================
class QuaternionHarmonization(nn.Module):
    """
    Ĥ[x] = x - β·σ(W_H·x) + γ·tanh(W_S·x)
    
    Two pathways:
    1. Damping: -β·sigmoid(linear(x))  — reduces excess complexity
    2. Syntony: +γ·tanh(linear(x))     — projects toward coherence
    
    From harmonization.py: "Creates coherence and integration (Whispers)"
    """
    def __init__(self, features, beta=1.0, gamma=1.0):
        super().__init__()
        # Damping pathway
        self.damping = QuaternionLinear(features, features)
        self.beta = beta
        
        # Syntony projection pathway
        self.syntony = QuaternionLinear(features, features)
        self.gamma = gamma
    
    def forward(self, r, i, j, k):
        # Damping: σ(W_H·x) — sigmoid squashes to damp excess
        dr, di, dj, dk = self.damping(r, i, j, k)
        dr = torch.sigmoid(dr)
        di = torch.sigmoid(di)
        dj = torch.sigmoid(dj)
        dk = torch.sigmoid(dk)
        
        # Syntony projection: tanh(W_S·x) — stabilize toward coherence
        sr, si, sj, sk = self.syntony(r, i, j, k)
        sr = torch.tanh(sr)
        si = torch.tanh(si)
        sj = torch.tanh(sj)
        sk = torch.tanh(sk)
        
        # Ĥ[x] = x - β·damping + γ·syntony
        out_r = r - self.beta * dr + self.gamma * sr
        out_i = i - self.beta * di + self.gamma * si
        out_j = j - self.beta * dj + self.gamma * sj
        out_k = k - self.beta * dk + self.gamma * sk
        
        return out_r, out_i, out_j, out_k


# ==========================================
# FULL NETWORK: D→H cycle with quaternion backbone
# ==========================================
class QuaternionDHNet(nn.Module):
    """
    Quaternion Bimodal network with D→H cycle.
    
    Architecture:
    1. Conv2d sensory organ: 1 channel → 4 channels (r,i,j,k)
       Each channel maps directly to a quaternion component.
       Image ENTERS quaternion space through spatial structure.
    2. Differentiation: expand in quaternion feature space
    3. Harmonization: compress/stabilize
    4. Final quaternion projection to 10-dim
    5. Real component = measurement output → classification
    
    Conv2d(1→4, kernel=5, stride=2): 28×28 → 4×12×12 = 144 per component
    Total conv params: 4*(1*25 + 1) = 104
    
    The D→H cycle mirrors the DHSR loop from CRT:
    D (expand) → H (compress/cohere) → S (measure syntony) → R (recurse or output)
    """
    def __init__(self):
        super().__init__()
        # Sensory organ: spatial conv maps image into 4 quaternion channels
        # 1 input channel → 4 output channels (one per quaternion component)
        self.sensory = nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=0)
        # Output: 4 × 12 × 12 = 144 features per quaternion component
        
        # D: Expand into quaternion feature space (144 → 32)
        self.diff = QuaternionDifferentiation(144, 32, alpha=1.0)
        
        # H: Harmonize / compress (32 → 32, residual)
        self.harm = QuaternionHarmonization(32, beta=0.5, gamma=1.0)
        
        # Final projection: quaternion 32 → quaternion 10
        self.q_out = QuaternionLinear(32, 10)
        
    def forward(self, x):
        # x: [batch, 1, 28, 28]
        x = x.view(-1, 1, 28, 28)
        
        # Sensory organ: conv produces 4 channels = quaternion components
        q = self.sensory(x)  # [batch, 4, 12, 12]
        
        # Each channel becomes a quaternion component, flattened
        r = q[:, 0].reshape(q.size(0), -1)  # [batch, 144]
        i = q[:, 1].reshape(q.size(0), -1)  # [batch, 144]
        j = q[:, 2].reshape(q.size(0), -1)  # [batch, 144]
        k = q[:, 3].reshape(q.size(0), -1)  # [batch, 144]
        
        # D phase: differentiate (expand complexity)
        r, i, j, k = self.diff(r, i, j, k)
        
        # H phase: harmonize (damp dissonance, project coherence)
        r, i, j, k = self.harm(r, i, j, k)
        
        # Output projection
        r, i, j, k = self.q_out(r, i, j, k)
        
        # BIMODAL OUTPUT: Real component IS the measurement.
        return r


# ==========================================
# Training Engine
# ==========================================
def train_and_evaluate(model, name):
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"{'='*50}")
    
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    
    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = 100. * correct / len(test_loader.dataset)
        avg_loss = epoch_loss / batches
        history.append(acc)
        print(f"  Epoch {epoch+1:2d}/{EPOCHS}: Loss {avg_loss:.4f} | Accuracy {acc:.2f}%")
        
    elapsed = time.time() - start_time
    print(f"  Finished in {elapsed:.1f}s ({elapsed/EPOCHS:.1f}s/epoch)")
    return history, count_parameters(model)


# ==========================================
# COMPARE
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("QUATERNION BIMODAL D/H PROOF OF CONCEPT v3")
    print("=" * 60)
    print(f"Theory: Quaternion Hamilton product + bimodal measurement")
    print(f"        + Differentiation/Harmonization cycle")
    print(f"Golden ratio φ = {PHI:.10f}")

    # Baseline
    scalar_model = ScalarNet()
    scalar_acc, scalar_params = train_and_evaluate(scalar_model, "Scalar Baseline (256/128)")

    # Quaternion D/H
    quat_model = QuaternionDHNet()
    quat_acc, quat_params = train_and_evaluate(quat_model, "Quaternion Bimodal D→H (32)")

    # Results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Scalar Net:       {scalar_params:>8,} params | Best: {max(scalar_acc):.2f}% | Final: {scalar_acc[-1]:.2f}%")
    print(f"  Quaternion D/H:   {quat_params:>8,} params | Best: {max(quat_acc):.2f}% | Final: {quat_acc[-1]:.2f}%")
    
    ratio = scalar_params / quat_params
    print(f"\n  Parameter reduction: {ratio:.1f}x fewer parameters")
    print(f"  Accuracy delta:     {quat_acc[-1] - scalar_acc[-1]:+.2f}%")
    
    scalar_eff = max(scalar_acc) / scalar_params * 1000
    quat_eff = max(quat_acc) / quat_params * 1000
    print(f"\n  Efficiency (acc% per 1k params):")
    print(f"    Scalar:     {scalar_eff:.3f}")
    print(f"    Quat D/H:   {quat_eff:.3f}")
    print(f"    Quat D/H is {quat_eff/scalar_eff:.1f}x more efficient per parameter")

    gap = abs(quat_acc[-1] - scalar_acc[-1])
    if quat_acc[-1] >= scalar_acc[-1] * 0.98:
        print("\n  >> RESULT: SUCCESS — Quaternion D/H matches baseline with far fewer params.")
    elif gap < 2.0:
        print(f"\n  >> RESULT: CLOSE — {gap:.2f}% gap. Architecture shows strong efficiency.")
    else:
        print(f"\n  >> RESULT: GAP — {gap:.2f}% gap. Further tuning needed.")