import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ============================================================
# 1. Define Teacher Network (Large Model)
# ============================================================

class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 2. Define Student Network (Smaller Model)
# ============================================================

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. Create Models
# ============================================================

teacher = TeacherNet()
student = StudentNet()

# ------------------------------------------------------------
# Normally:
# Load a PRETRAINED teacher checkpoint
#
# teacher.load_state_dict(
#     torch.load("teacher.pth")
# )
# ------------------------------------------------------------

teacher.eval()  # Teacher is frozen

for param in teacher.parameters():
    param.requires_grad = False


# ============================================================
# 4. Optimizer
# ============================================================

optimizer = optim.Adam(student.parameters(), lr=1e-3)


# ============================================================
# 5. Distillation Hyperparameters
# ============================================================

TEMPERATURE = 4.0

# Balance between:
#   KD loss (teacher imitation)
#   CE loss (ground truth)
ALPHA = 0.7


# ============================================================
# 6. Distillation Loss Function
# ============================================================

def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature=4.0,
    alpha=0.7
):
    """
    Combined Loss:

    L =
        alpha * KD_loss
        +
        (1-alpha) * CE_loss
    """

    # --------------------------------------------------------
    # Teacher Soft Labels
    #
    # Higher temperature:
    # softer probability distribution
    # --------------------------------------------------------

    teacher_probs = F.softmax(
        teacher_logits / temperature,
        dim=1
    )

    # --------------------------------------------------------
    # Student Probabilities
    # --------------------------------------------------------

    student_log_probs = F.log_softmax(
        student_logits / temperature,
        dim=1
    )

    # --------------------------------------------------------
    # Knowledge Distillation Loss
    #
    # KL divergence:
    # Make student mimic teacher
    # --------------------------------------------------------

    kd_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean"
    )

    # Temperature correction
    kd_loss = kd_loss * (temperature ** 2)

    # --------------------------------------------------------
    # Standard Supervised Loss
    #
    # Learn from real labels
    # --------------------------------------------------------

    ce_loss = F.cross_entropy(
        student_logits,
        labels
    )

    # --------------------------------------------------------
    # Final Combined Loss
    # --------------------------------------------------------

    total_loss = (
        alpha * kd_loss +
        (1 - alpha) * ce_loss
    )

    return total_loss


# ============================================================
# 7. Training Loop
# ============================================================

NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):

    student.train()

    total_loss = 0

    for images, labels in train_loader:

        # ----------------------------------------------------
        # Flatten MNIST image:
        # [batch,1,28,28]
        #
        # -> [batch,784]
        # ----------------------------------------------------

        images = images.view(images.size(0), -1)

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Teacher Forward Pass
        #
        # No gradients needed
        # ----------------------------------------------------

        with torch.no_grad():
            teacher_logits = teacher(images)

        # ----------------------------------------------------
        # Student Forward Pass
        # ----------------------------------------------------

        student_logits = student(images)

        # ----------------------------------------------------
        # Compute Distillation Loss
        # ----------------------------------------------------

        loss = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            temperature=TEMPERATURE,
            alpha=ALPHA
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} "
        f"Loss: {total_loss:.4f}"
    )


# ============================================================
# 8. Evaluation
# ============================================================

student.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.view(images.size(0), -1)

        outputs = student(images)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

print(f"Student Accuracy: {accuracy:.2f}%")