import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from vaemodel import VAEResnet1d
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms
import cv2
import numpy as np

import wandb

wandb.init(project="myvae")


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, recon_x, x, mu, logvar):
        loss_mse = self.mse_loss(recon_x, x)
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss_mse + 0.005 * loss_kld


def visualize_outputs(epoch, ckpt_path, inputs, outputs, num_samples=4):
    inputs = inputs.detach().cpu()  # Move inputs to CPU
    outputs = outputs.detach().cpu()  # Move outputs to CPU
    print(inputs.shape)
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(inputs[i].permute(1, 2, 0), cmap="gray")
        plt.title("Input")
        plt.axis("off")

        plt.subplot(2, num_samples, i + num_samples + 1)
        plt.imshow(outputs[i].permute(1, 2, 0), cmap="gray")
        plt.title("Output")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(ckpt_path + f"/epoch_{epoch}_outputs.png")
    plt.close()


def save_model(model, optimizer, epoch, ckpt_path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, ckpt_path + f"/epoch_{epoch}_checkpoint.pth")


def train_vae(
    model,
    optimizer,
    dataloader,
    ckpt_path,
    customloss,
    start_epoch,
    num_epochs=10,
):
    train_losses = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        total_loss = 0

        for i, (inputs, label) in enumerate(tqdm(dataloader)):
            # inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs)
            loss = customloss(recon_batch, inputs, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        epoch_loss = total_loss / len(dataloader.dataset)
        wandb.log({"epoch": epoch, "loss": epoch_loss})
        train_losses.append(epoch_loss)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader.dataset):.4f}"
        )
        if epoch % 10 == 0:
            visualize_outputs(epoch, ckpt_path, inputs, recon_batch)
        # save model every 20 epochs with different name
        if epoch % 20 == 0:
            save_model(model, optimizer, epoch, ckpt_path)

    print("Training completed!")
    return train_losses


def detect_anomalies(model, dataloader, threshold=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    anomaly_images = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, _ = data
            inputs = inputs.to(device)
            recon_batch, mu, logvar = model(inputs)
            loss = vae_loss(recon_batch, inputs, mu, logvar)
            if loss.item() > threshold:
                anomaly_images.append(inputs.cpu())
    return anomaly_images


class CustomTransform(object):
    def __call__(self, image_tensor):
        # image_np = np.array(image)
        # image_tensor = torch.from_numpy(image_np.astype(np.float32))
        image_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min()
        )
        image_tensor = image_tensor.permute(-1, 0, 1)

        return image_tensor


class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = cv2.imread(path, -1)
        # sample = cv2.cvtColor(sample, cv2.COLOR_GRAY2RGB)
        # add channel dimension
        sample = np.expand_dims(sample, axis=2)

        sample = torch.from_numpy(sample.astype(np.float32))

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # sample = sample.permute(2, 0, 1)
        return sample, target


# Main program
if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            CustomTransform(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                torch.nn.ModuleList(
                    [
                        transforms.ColorJitter(brightness=(1, 2)),
                        transforms.ColorJitter(contrast=(1, 2)),
                    ]
                ),
                p=0.5,
            ),
        ]
    )
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--is_pretrained",
        type=bool,
        default=True,
        metavar="N",
        help="is pretrained (default: False)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/home/woody/iwi5/iwi5095h/oneclassclassifier/vaeresnet_redsum_transform_b_005_1channel/",
        metavar="N",
        help="checkpoint path (default: None)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=600,
        metavar="N",
        help="number of epochs (default: 10)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/vault/iwi5/iwi5095h/Oneclassgood",
        metavar="N",
        help="data directory (default: /home/vault/iwi5/iwi5095h/Oneclassgood)",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=1024,
        metavar="N",
        help="latent dimension (default: 128)",
    )
    args = parser.parse_args()
    custom_dataset = CustomImageFolder(args.data_dir, transform=transform)
    train_data, test_data = torch.utils.data.random_split(
        custom_dataset,
        [
            int(len(custom_dataset) * 0.9),
            len(custom_dataset) - int(len(custom_dataset) * 0.9),
        ],
        generator=torch.Generator().manual_seed(42),
    )
    print("Number of train images: ", len(train_data))

    # create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    # loss_mse = customLoss()
    vae = VAEResnet1d(256)
    print(vae)
    start_epoch = 0
    loss_mse = customLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)
    if args.is_pretrained:
        ckpt_path_ = args.ckpt_path + "epoch_620_checkpoint.pth"
        checkpoint = torch.load(ckpt_path_)  # Provide the correct path
        vae.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print("loading pretrained model from epoch ", start_epoch)
        # change optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = 1e-6

        # Train the one-class classifier using VAE
    train_losses = train_vae(
        vae,
        optimizer,
        train_dataloader,
        args.ckpt_path,
        loss_mse,
        start_epoch=start_epoch,
        num_epochs=args.num_epochs,
    )
