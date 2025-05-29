from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
import torchvision

def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    device: torch.device,
    nb_epochs: int,
    model_path: Path | str,
    writer: SummaryWriter | None = None
    ) -> None:

    generator.train()
    discriminator.train()

    fixed_noise = torch.randn(32, generator.latent_dim).to(device)
    criterion = nn.BCELoss()
    # criterion = nn.MSELoss()
    
    step = 0
    for epoch in range(nb_epochs):
        for batch_idx, (real, _) in enumerate(tqdm(dataloader)):
            real = real.to(device)
            batch_size = real.size(0)
            real_labels = torch.ones(batch_size, 1, device=device) * 0.9
            fake_labels = torch.zeros(batch_size, 1, device=device) * 0.1

            # Train Discriminator
            noise = torch.randn(batch_size, generator.latent_dim, device=device)
            fake = generator(noise).detach()
            out_real = discriminator(real)
            out_fake = discriminator(fake)
            loss_real = criterion(out_real, real_labels)
            loss_fake = criterion(out_fake, fake_labels)
            loss_discriminator = loss_real + loss_fake

            discriminator.zero_grad()
            loss_discriminator.backward()
            discriminator_optimizer.step()

            # Train Generator
            noise = torch.randn(batch_size, generator.latent_dim, device=device)
            fake = generator(noise)
            out = discriminator(fake)
            loss_generator = criterion(out, real_labels)

            generator.zero_grad()
            loss_generator.backward()
            generator_optimizer.step()

            # Logging to Tensorboard
            if writer is not None:
                writer.add_scalar("GAN/Discriminator_Loss", loss_discriminator.item(), global_step=step)
                writer.add_scalar("GAN/Generator_Loss", loss_generator.item(), global_step=step)

            step += 1

        print(
            f"Epoch [{epoch}/{nb_epochs}]"
            f"Loss D: {loss_discriminator:.4f}, Loss G: {loss_generator:.4f}"
        )

        if writer is not None and epoch % 2 == 0:
            with torch.no_grad():
                samples = generator(fixed_noise).to(device)
                grid = torchvision.utils.make_grid(samples[:32], nrow=8, normalize=True)
                writer.add_image("GAN/Samples", grid, global_step=step)


    # Save model
    torch.save(generator.state_dict(), model_path)