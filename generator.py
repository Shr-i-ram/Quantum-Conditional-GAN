import torch
import torch.nn as nn
import pennylane as qml

# ============================================================
# Hyperparameters (MUST match the training configuration)
# ============================================================

image_size = 8

latent_dim = 32

stem_hidden = 16

n_classes = 10

n_qubits = 5
n_a_qubits = 1
q_depth = 6

n_generators = 4

patch_size = 2 ** (n_qubits - n_a_qubits)

assert patch_size * n_generators == image_size * image_size


# ============================================================
# Quantum Device
# ============================================================

dev = qml.device("lightning.qubit", wires=n_qubits)


# ============================================================
# Quantum Circuit
# ============================================================

@qml.qnode(dev, diff_method="parameter-shift")
def quantum_circuit(latent, weights, label_bias):

    weights = weights.reshape(q_depth, n_qubits)

    # State preparation
    for qubit in range(n_qubits):
        qml.RY(latent[qubit], wires=qubit)

    # Variational layers
    for layer in range(q_depth):

        for qubit in range(n_qubits):

            qml.RY(
                weights[layer, qubit] +
                label_bias[layer, qubit],
                wires=qubit
            )

        for qubit in range(n_qubits - 1):
            qml.CZ(wires=[qubit, qubit + 1])

    return qml.probs(wires=range(n_qubits))


def partial_measure(latent, weights, label_bias):

    probs = quantum_circuit(
        latent,
        weights,
        label_bias
    )

    probs = probs[: 2 ** (n_qubits - n_a_qubits)]

    probs = probs / probs.sum()

    probs = probs / probs.max()

    return probs


# ============================================================
# Quantum Patch
# ============================================================

class QuantumPatch(nn.Module):

    def __init__(self):

        super().__init__()

        self.q_params = nn.Parameter(
            0.01 * torch.randn(q_depth * n_qubits)
        )

    def forward(self, latent, label_bias):

        outputs = []

        for x, b in zip(latent, label_bias):

            outputs.append(
                partial_measure(
                    x,
                    self.q_params,
                    b
                ).float()
            )

        return torch.stack(outputs)


# ============================================================
# Generator
# ============================================================

class ConditionalQuantumGenerator(nn.Module):

    def __init__(self):

        super().__init__()

        self.bias_network = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(
                64,
                n_generators * q_depth * n_qubits
            )
        )

        self.label_embedding = nn.Embedding(
            n_classes,
            latent_dim
        )

        self.stem = nn.Sequential(
            nn.Linear(latent_dim, stem_hidden),
            nn.ReLU(),
            nn.Linear(
                stem_hidden,
                n_generators * n_qubits
            )
        )

        self.patches = nn.ModuleList(
            [
                QuantumPatch()
                for _ in range(n_generators)
            ]
        )

    def forward(self, noise, labels):

        embedded = self.label_embedding(labels)

        condition = noise + embedded

        bias = self.bias_network(condition)

        bias = bias.view(
            -1,
            n_generators,
            q_depth,
            n_qubits
        )

        latent = self.stem(condition)

        latent = latent.view(
            -1,
            n_generators,
            n_qubits
        )

        outputs = []

        for i in range(n_generators):

            outputs.append(
                self.patches[i](
                    latent[:, i, :],
                    bias[:, i]
                )
            )

        image = torch.cat(
            outputs,
            dim=1
        )

        return image


# ============================================================
# Wrapper
# ============================================================

class QGANGenerator:

    def __init__(
        self,
        weight_path,
        device=None
    ):

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.model = ConditionalQuantumGenerator().to(
            self.device
        )

        self.model.load_state_dict(
            torch.load(
                weight_path,
                map_location=self.device
            )
        )

        self.model.eval()

    @torch.no_grad()
    def generate(self, digit):

        noise = torch.randn(
            1,
            latent_dim,
            device=self.device
        )

        label = torch.tensor(
            [digit],
            dtype=torch.long,
            device=self.device
        )

        image = self.model(
            noise,
            label
        )

        image = image.squeeze()

        image = image.reshape(8, 8)

        return image.cpu().numpy()