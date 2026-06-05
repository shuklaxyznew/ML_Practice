"""
Knowledge Distillation Experiment — MNIST
==========================================
Distills a 4-layer teacher MLP into a 2-layer student MLP.
Compares three training regimes:
  1. Teacher (baseline)
  2. Student trained from scratch (hard labels only)
  3. Student trained with distillation (soft + hard labels)

Requirements: torch torchvision
Install:  pip install torch torchvision
Run:      python knowledge_distillation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# ─── Config ───────────────────────────────────────────────────────────────────
TEMPERATURE   = 4.0    # T: controls softness of teacher distribution
ALPHA         = 0.7    # weight on distillation loss (1-alpha for CE loss)
BATCH_SIZE    = 256
EPOCHS        = 5
LR            = 1e-3
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ─── Data ─────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST("./data", train=True,  download=True, transform=transform)
test_data  = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ─── Models ───────────────────────────────────────────────────────────────────
class TeacherMLP(nn.Module):
    """Large teacher: 4 hidden layers, ~235K params"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64,  10),
        )
    def forward(self, x):
        return self.net(x)  # returns raw logits

class StudentMLP(nn.Module):
    """Small student: 2 hidden layers, ~51K params (4.6× smaller)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 64), nn.ReLU(),
            nn.Linear(64,  32), nn.ReLU(),
            nn.Linear(32,  10),
        )
    def forward(self, x):
        return self.net(x)  # returns raw logits

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ─── Loss Functions ───────────────────────────────────────────────────────────
def distillation_loss(student_logits, teacher_logits, true_labels, T, alpha):
    """
    L = alpha * T^2 * KL(teacher_soft || student_soft)
      + (1 - alpha) * CrossEntropy(student_hard, true_labels)

    The T^2 factor compensates for the reduced gradient magnitudes
    caused by softening with temperature T.
    """
    # Soft targets from teacher at temperature T
    teacher_soft = F.softmax(teacher_logits / T, dim=-1)
    student_soft = F.log_softmax(student_logits / T, dim=-1)

    # KL divergence: sum_i p_teacher * log(p_teacher / p_student)
    kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")

    # Standard cross-entropy on hard labels (T=1)
    ce_loss = F.cross_entropy(student_logits, true_labels)

    return alpha * (T ** 2) * kl_loss + (1 - alpha) * ce_loss

# ─── Training & Evaluation ────────────────────────────────────────────────────
def train_standard(model, loader, optimizer):
    """Train with hard labels (standard cross-entropy)."""
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def train_distillation(student, teacher, loader, optimizer, T, alpha):
    """Train student with distillation loss (soft + hard labels)."""
    student.train()
    teacher.eval()
    total_loss = 0.0
    with torch.no_grad():
        pass  # teacher grads not needed
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.no_grad():
            t_logits = teacher(imgs)          # teacher frozen
        s_logits = student(imgs)
        loss = distillation_loss(s_logits, t_logits, labels, T, alpha)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total

def train_and_eval(name, model, loader, optimizer, epochs, teacher=None, T=None, alpha=None):
    print(f"\n{'─'*50}")
    print(f" {name}  ({count_params(model):,} params)")
    print(f"{'─'*50}")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        if teacher is not None:
            loss = train_distillation(model, teacher, loader, optimizer, T, alpha)
        else:
            loss = train_standard(model, loader, optimizer)
        acc = evaluate(model, test_loader)
        print(f"  Epoch {epoch}/{epochs}  loss={loss:.4f}  test_acc={acc:.2f}%")
    elapsed = time.time() - t0
    final_acc = evaluate(model, test_loader)
    print(f"\n  ✓ Final test accuracy: {final_acc:.2f}%  ({elapsed:.1f}s)")
    return final_acc

# ─── Run Experiment ───────────────────────────────────────────────────────────
print("=" * 50)
print("  Knowledge Distillation Experiment — MNIST")
print(f"  Temperature T={TEMPERATURE}, Alpha α={ALPHA}")
print("=" * 50)

# 1. Train teacher
teacher = TeacherMLP().to(DEVICE)
teacher_opt = torch.optim.Adam(teacher.parameters(), lr=LR)
teacher_acc = train_and_eval(
    "Teacher (4-layer MLP)", teacher, train_loader, teacher_opt, EPOCHS
)

# 2. Train student from scratch (hard labels only)
student_scratch = StudentMLP().to(DEVICE)
scratch_opt = torch.optim.Adam(student_scratch.parameters(), lr=LR)
scratch_acc = train_and_eval(
    "Student — scratch (hard labels)", student_scratch, train_loader, scratch_opt, EPOCHS
)

# 3. Train student with distillation
student_distilled = StudentMLP().to(DEVICE)
distill_opt = torch.optim.Adam(student_distilled.parameters(), lr=LR)
distill_acc = train_and_eval(
    "Student — distilled (T=4, α=0.7)", student_distilled, train_loader, distill_opt,
    EPOCHS, teacher=teacher, T=TEMPERATURE, alpha=ALPHA
)

# ─── Results Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  Results Summary")
print("=" * 50)
print(f"  {'Model':<34} {'Params':>10} {'Test Acc':>10}")
print(f"  {'-'*54}")
print(f"  {'Teacher MLP (4-layer)':<34} {count_params(TeacherMLP()):>10,} {teacher_acc:>9.2f}%")
print(f"  {'Student (scratch, hard labels)':<34} {count_params(StudentMLP()):>10,} {scratch_acc:>9.2f}%")
print(f"  {'Student (distilled, T=4, α=0.7)':<34} {count_params(StudentMLP()):>10,} {distill_acc:>9.2f}%")
print(f"\n  Distillation gain vs scratch: +{distill_acc - scratch_acc:.2f}pp")
print(f"  Accuracy retained vs teacher: {100*distill_acc/teacher_acc:.1f}%")
print(f"  Parameter compression:        {count_params(TeacherMLP())/count_params(StudentMLP()):.1f}×")

# ─── Experiment: vary temperature ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("  Temperature Ablation (T ∈ {1, 2, 4, 8})")
print("=" * 50)
for T in [1.0, 2.0, 4.0, 8.0]:
    s = StudentMLP().to(DEVICE)
    opt = torch.optim.Adam(s.parameters(), lr=LR)
    for _ in range(EPOCHS):
        train_distillation(s, teacher, train_loader, opt, T=T, alpha=ALPHA)
    acc = evaluate(s, test_loader)
    print(f"  T={T:<4}  →  test_acc={acc:.2f}%")
print("\nDone.")
