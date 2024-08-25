# D Loss rappresenta la perdita (loss) del Discriminatore dopo questa epoca. La perdita del Discriminatore misura quanto bene riesce a distinguere i campioni reali da quelli generati. Un valore molto basso (come in questo caso, 0.0324) indica che il Discriminatore sta facendo un buon lavoro nel classificare correttamente i dati reali come reali e i dati falsi come falsi.

# G Loss rappresenta la perdita del Generatore. La perdita del Generatore misura quanto bene sta ingannando il Discriminatore. Un valore più alto (come 3.96) suggerisce che il Generatore sta avendo difficoltà a ingannare il Discriminatore, e quindi deve migliorare nella generazione di dati più realistici.


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Definizione del Generatore
class Generator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            nn.Tanh(),
        )
        print(f"G MODEL: {self.model}")

    def forward(self, x):
        return self.model(x)


# Definizione del Discriminatore
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# Parametri principali
z_dim = 100  # Dimensione dell'input del generatore (ad es. rumore casuale)
image_dim = 64 * 64  # Dimensione delle immagini convertite da malware
hidden_dim = 128

# Inizializzazione dei modelli
G = Generator(z_dim, hidden_dim, image_dim)
D = Discriminator(image_dim, hidden_dim)

try:
    G.load_state_dict(torch.load("best_generator.pth"))
    print("Modello Generatore caricato correttamente.")
except FileNotFoundError:
    print("Modello Generatore non trovato. Avvio con un nuovo modello.")

try:
    D.load_state_dict(torch.load("best_discriminator.pth"))
    print("Modello Discriminatore caricato correttamente.")
except FileNotFoundError:
    print("Modello Discriminatore non trovato. Avvio con un nuovo modello.")

# Ottimizzatori
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

# Funzione di perdita
loss_fn = nn.BCELoss()

# Esempio di Dataloader (da sostituire con il dataloader reale per il malware)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)
dataloader = DataLoader(
    datasets.FakeData(image_size=(1, 64, 64), transform=transform),
    batch_size=64,
    shuffle=True,
)

# Inizializzazione delle variabili per tenere traccia delle migliori perdite
best_D_loss = float("inf")
best_G_loss = float("inf")

# Ciclo di Addestramento
num_epochs = 100
for epoch in range(num_epochs):
    for real_data, _ in dataloader:
        real_data = real_data.view(-1, image_dim)
        batch_size = real_data.size(0)

        # Etichette per dati reali e falsi
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Addestramento del Discriminatore
        z = torch.randn(batch_size, z_dim)
        fake_data = G(z)
        D_real = D(real_data)
        D_fake = D(fake_data)

        D_loss_real = loss_fn(D_real, real_labels)
        D_loss_fake = loss_fn(D_fake, fake_labels)
        D_loss = D_loss_real + D_loss_fake

        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Addestramento del Generatore
        z = torch.randn(batch_size, z_dim)
        fake_data = G(z)
        D_fake = D(fake_data)

        G_loss = loss_fn(D_fake, real_labels)

        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

    print(
        f"Epoch [{epoch}/{num_epochs}] | D Loss: {D_loss.item()} | G Loss: {G_loss.item()}"
    )

    # Salvataggio dei modelli se le perdite migliorano
    if D_loss.item() < best_D_loss:
        best_D_loss = D_loss.item()
        torch.save(D.state_dict(), f"best_discriminator.pth")
        print("Miglior D train salvato")

    if G_loss.item() < best_G_loss:
        best_G_loss = G_loss.item()
        torch.save(G.state_dict(), f"best_generator.pth")
        print("Miglior G train salvato")
