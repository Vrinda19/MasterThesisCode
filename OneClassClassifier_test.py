import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, transforms
import torchvision.utils as vutils
import torchvision.datasets as datasets
from model import myvae, newvae, VAEResnet
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from tqdm import tqdm


class CustomTransform(object):
    def __call__(self, image_tensor):
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
        sample = cv2.cvtColor(sample, cv2.COLOR_GRAY2RGB)
        sample = torch.from_numpy(sample.astype(np.float32))

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


class ModifiedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, target_transform):
        self.original_dataset = original_dataset
        self.target_transform = target_transform

    def __getitem__(self, index):
        data, original_target = self.original_dataset[index]
        target = self.target_transform(original_target)
        return data, target

    def __len__(self):
        return len(self.original_dataset)


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, recon_x, x, mu, logvar):
        loss_mse = self.mse_loss(recon_x, x)
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss_mse + 0.005 * loss_kld


transform = transforms.Compose([CustomTransform()])


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color="orange", label="ROC")
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.savefig("roc_curve.png")


def see_and_save_reconstructed_image(i, inputs, recon_batch, label):
    # plot inputs and recon_batch side by side in a single image and caption it with label
    inputs = inputs.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    recon_batch = recon_batch.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(inputs, cmap="gray")
    ax1.set_title("Input")
    ax2.imshow(recon_batch, cmap="gray")
    ax2.set_title("Reconstructed")
    fig.suptitle("Label: " + str(label.item()))
    plt.savefig(
        "resnetreconstructedresultsvae/1620/reconstructed_image_"
        + str(label.item())
        + str(i)
        + ".png"
    )


def detect_anomalies(model, dataloader, loss, threshold=10000):
    print("Anomaly detection started!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    anomaly_images = []
    is_anomaly = True
    scatter_plot = []

    df = pd.DataFrame(
        columns=["input_image", "reconstruction_loss", "label", "is_anomaly"]
    )
    pred = 0
    count_correct = 0
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            pred = 0

            inputs, label = data
            inputs = inputs.to(device)
            recon_batch, mu, logvar = model(inputs)
            if label.item() == 1:
                see_and_save_reconstructed_image(i, inputs, recon_batch, label)
            loss_ = loss(recon_batch, inputs, mu, logvar)
            # print('threshold: ', threshold)
            if loss_.item() > threshold:
                # anomaly_images.append(inputs.cpu())
                is_anomaly = True
                pred = 1
            if pred == label.item():
                count_correct += 1
            true_labels.extend([label.item()])
            predicted_labels.extend([pred])

            # df.loc[i]=[inputs.cpu(),loss.item(),label.item(),is_anomaly]
            scatter_plot.append([loss_.item(), label.item()])
        accuracy = count_correct / len(dataloader.dataset)
        print("len(true_labels): ", len(true_labels))
        print("len(predicted_labels): ", len(predicted_labels))
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        print("confusion matrix is: ", conf_matrix)

        print("accuracy is: ", accuracy)
        # get confusion matrix between 0 and 1 labels and predicted labels

    # plot scatter plot of reconstruction loss and label  with different colors for label 0 and 1
    # plot Histogram of reconstruction scores of real (green) and fake (red) images, and the statistical threshold (orange)
    real_losses = [loss for loss, label in scatter_plot if label == 0]
    fake_losses = [loss for loss, label in scatter_plot if label == 1]

    # take 50 real losses where loss value is less than 1000
    real_losses = [loss for loss in real_losses if loss < 1000][:50]

    # take 50 fake losses where loss value is greater than 1000
    fake_losses = [loss for loss in fake_losses if loss > 1000][:50]
    # Create a histogram plot
    plt.figure(figsize=(10, 5))

    plt.hist(real_losses, bins=20, color="green", alpha=0.5, label="Good (label 0)")
    plt.hist(fake_losses, bins=20, color="red", alpha=0.5, label="Defective (label 1)")
    plt.axvline(
        x=threshold, color="orange", linestyle="--", linewidth=3, label="Threshold"
    )

    # Add labels and a legend
    plt.xlabel("Reconstruction Scores")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.savefig("histogram.png")

    plt.clf()

    plt.figure(figsize=(8, 6))
    plt.scatter(real_losses, real_losses, c="green", label="Real (label=1)", alpha=0.7)
    plt.scatter(fake_losses, fake_losses, c="red", label="Fake (label=0)", alpha=0.7)

    # Customize the plot
    plt.xlabel("Reconstruction Loss (X-axis)")
    plt.ylabel("Reconstruction Loss (Y-axis)")
    plt.title("Scatter Plot of Reconstruction Loss")
    plt.legend()
    plt.savefig("scatter_plot.png")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_dataset = CustomImageFolder(
        "/home/vault/iwi5/iwi5095h/Oneclassgood", transform=transform
    )
    defect_dataset = CustomImageFolder(
        "/home/vault/iwi5/iwi5095h/oneclassdefect", transform=transform
    )
    train_data, test_data = torch.utils.data.random_split(
        custom_dataset,
        [
            int(len(custom_dataset) * 0.9),
            len(custom_dataset) - int(len(custom_dataset) * 0.9),
        ],
        generator=torch.Generator().manual_seed(42),
    )

    print("images in train_data: ", len(train_data))
    print("images in test_data: ", len(test_data))
    print("images in defect_dataset: ", len(defect_dataset))
    combined_dataset = torch.utils.data.ConcatDataset(
        [
            ModifiedDataset(
                defect_dataset, target_transform=lambda x: 1
            ),  # Set labels to 1 for defect_dataset
            ModifiedDataset(
                test_data, target_transform=lambda x: 0
            ),  # Set labels to 0 for custom_dataset
        ]
    )

    # create dataloader
    test_dataloader = torch.utils.data.DataLoader(
        combined_dataset, batch_size=1, shuffle=True
    )
    model = VAEResnet(256)

    # load X_test
    ckpt = torch.load(
        "/home/woody/iwi5/iwi5095h/oneclassclassifier/vaeresnet_redsum_transform_b_005/continue_with_lr_006_after1380/epoch_1620_checkpoint.pth",
        map_location=torch.device(device),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    print("Model loaded!")
    loss = nn.MSELoss(reduction="sum")

    # threshold=4400
    detect_anomalies(model, test_dataloader, loss, threshold=42)
