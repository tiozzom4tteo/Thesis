# D Loss rappresenta la perdita (loss) del Discriminatore dopo questa epoca. La perdita del Discriminatore misura quanto bene riesce a distinguere i campioni reali da quelli generati. Un valore molto basso (come in questo caso, 0.0324) indica che il Discriminatore sta facendo un buon lavoro nel classificare correttamente i dati reali come reali e i dati falsi come falsi.

# G Loss rappresenta la perdita del Generatore. La perdita del Generatore misura quanto bene sta ingannando il Discriminatore. Un valore più alto (come 3.96) suggerisce che il Generatore sta avendo difficoltà a ingannare il Discriminatore, e quindi deve migliorare nella generazione di dati più realistici.


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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


# Definizione di un Dataset personalizzato per il malware
class MalwareDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return data directly as a tensor
        return self.data[idx]


# Parametri principali
z_dim = 100  # Dimensione dell'input del generatore (ad es. rumore casuale)
image_dim = 64 * 64  # Dimensione delle immagini convertite da malware
hidden_dim = 128

# Verifica disponibilità GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inizializzazione dei modelli
G = Generator(z_dim, hidden_dim, image_dim).to(device)
D = Discriminator(image_dim, hidden_dim).to(device)


# Tentativo di caricare i modelli salvati
def load_model(model, filename):
    try:
        model.load_state_dict(torch.load(filename))
        print(f"Modello caricato correttamente da {filename}.")
    except FileNotFoundError:
        print(f"Modello non trovato: {filename}. Avvio con un nuovo modello.")


load_model(G, "best_generator.pth")
load_model(D, "best_discriminator.pth")

# Ottimizzatori
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Funzione di perdita
loss_fn = nn.BCELoss()

# Creazione di un dataset di esempio (sostituiscilo con i tuoi dati reali)
fake_data = torch.randn(1000, image_dim)  # Dati fittizi; sostituiscili con dati reali
dataset = MalwareDataset(fake_data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Inizializzazione delle variabili per tenere traccia delle migliori perdite
best_D_loss = float("inf")
best_G_loss = float("inf")

# Ciclo di Addestramento
num_epochs = 100
for epoch in range(num_epochs):
    for real_data in dataloader:
        real_data = real_data.to(device)
        batch_size = real_data.size(0)

        # Etichette per dati reali e falsi
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # Addestramento del Discriminatore
        z = torch.randn(batch_size, z_dim, device=device)
        fake_data = G(z)
        D_real = D(real_data)
        D_fake = D(fake_data.detach())  # Non aggiornare i pesi del generatore

        D_loss_real = loss_fn(D_real, real_labels)
        D_loss_fake = loss_fn(D_fake, fake_labels)
        D_loss = D_loss_real + D_loss_fake

        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Addestramento del Generatore
        z = torch.randn(batch_size, z_dim, device=device)
        fake_data = G(z)
        D_fake = D(fake_data)

        G_loss = loss_fn(D_fake, real_labels)

        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

    print(
        f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {D_loss.item()} | G Loss: {G_loss.item()}"
    )

    # Salvataggio dei modelli se le perdite migliorano
    if D_loss.item() < best_D_loss:
        best_D_loss = D_loss.item()
        torch.save(D.state_dict(), "best_discriminator.pth")
        print("Miglior D train salvato")

    if G_loss.item() < best_G_loss:
        best_G_loss = G_loss.item()
        torch.save(G.state_dict(), "best_generator.pth")
        print("Miglior G train salvato")
