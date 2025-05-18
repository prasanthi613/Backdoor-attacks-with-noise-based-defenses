Backdoor Attacks with Noise-based Defenses


This project explores the impact of backdoor attacks on deep learning models and implements noise-based defense strategies to mitigate them. Using the CIFAR-100 dataset and PyTorch, it demonstrates how models can be compromised with minimal tampering and investigates methods to detect and neutralize such threats.

📖 Overview
Backdoor attacks are a class of adversarial attacks where a model is trained to behave normally on standard data, but misclassifies any input containing a specific trigger pattern. These attacks are dangerous because:
They are silent and hard to detect.The model performs perfectly on clean test data.Attackers can control predictions by adding imperceptible changes to input.
This simulates such attacks and applies random noise-based perturbations as a defensive strategy to reduce their effectiveness.

🧨 Types of Backdoor Attacks Used
We implemented the following backdoor attack techniques:

1. Poisoning with Trigger
A small percentage (typically 1–10%) of the training dataset is poisoned by adding a small trigger (e.g., a white square in a corner) to input images and relabeling them to a target class.

✅ Clean data → correct label

❌ Image with trigger → forcibly labeled as target class

2. Targeted Misclassification
The model is trained to associate a pattern (trigger) with a specific class label. During inference, any input with that pattern will be misclassified into that class.

Attack Goals:
Stealth: Maintain high accuracy on clean validation data.

Effectiveness: High success rate when the trigger is applied to new inputs.

🛡️ Defense Strategy: Noise-Based Perturbations
The proposed defense strategy leverages random noise perturbation during inference to suppress the effect of backdoor triggers.

Defense Techniques:
Gaussian Noise Injection:
Adds controlled random noise to input images before classification.
Breaks the exact structure of the backdoor pattern.

Noise Masking:
Applies localized noise only to areas where the trigger might be located (e.g., bottom-right corner).
Attempts to minimize performance loss on clean data.

Ensemble Voting:
Runs multiple noisy versions of the same input.
If the output varies significantly, the input is flagged as suspicious.

🧪 Dataset
Dataset: CIFAR-100
Size: 50,000 training and 10,000 test images across 100 classes
Image Size: 32x32 RGB

Loaded via torchvision.datasets.CIFAR100

📊 Results and Analysis
✅ Model Accuracy (without attack):
Training Accuracy: ~85%

Test Accuracy: ~60–70% (varies slightly)

❌ Under Backdoor Attack:
Trigger Injection Rate: 5–10% of training data

Clean Test Accuracy: Unchanged (normal behavior)
Attack Success Rate: >95% (trigger always redirects to target label)

🛡️ With Defense (Noise-based Perturbation):

Clean Test Accuracy: Slightly reduced (~2–5% drop)
Attack Success Rate: Drops significantly (from >95% to <40%)
Detection Rate: Inputs with unstable predictions (under noise) are likely poisoned

📷 Visual Results:
Visual comparison of clean, triggered, and noise-defended inputs
Pixel value distribution analysis
Class activation comparison under perturbation

# Clone the repository
git clone https://github.com/your-username/backdoor-defense-noise.git
cd backdoor-defense-noise

# Install required libraries
pip install -r requirements.txt
🚀 How to Use
Open the notebook and run all cells:
bash
Copy
Edit
jupyter notebook "Backdoor Attacks with Noise-based Defenses.ipynb"
Modify hyperparameters like trigger size, noise level, and defense intensity in the respective cells to test various scenarios.

🧠 Key Learnings
Backdoor attacks are easy to implement and hard to detect with conventional testing.
Noise-based perturbations offer a lightweight defense, but may slightly reduce overall accuracy.
Randomized defenses can reduce the attack success rate without needing model retraining or architecture change.

🤝 Contributing
Want to improve this project? Found a bug? Pull requests and issue reports are welcome.
