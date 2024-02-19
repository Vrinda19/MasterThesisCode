import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, ConcatDataset

import matplotlib.pyplot as plt
import time
import os
import copy
import seaborn as sn
import pandas as pd
import torchnet.meter.confusionmeter as cm

import torch.multiprocessing

from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")
from sklearn.model_selection import StratifiedKFold  # Import StratifiedKFold


import sys

sys.path.append(
    "/home/hpc/iwi5/iwi5095h/masterthesis/thesis/experiments/CNN-multiclass-classification/M3d"
)

import cv2


# os.environ["WANDB_MODE"] = "offline"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import wandb


# create a argparser
import argparse

parser = argparse.ArgumentParser(description="CNN for multiclass classification")
parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    metavar="N",
    help="input batch size for training (default: 1)",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=30,
    metavar="N",
    help="number of epochs to train (default: 2)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0001,
    metavar="LR",
    help="learning rate (default: 0.001)",
)
parser.add_argument(
    "--weight_decay", default=0.001, metavar="LR", help="learning rate (default: 0.001)"
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="SGD momentum (default: 0.9)",
)
parser.add_argument(
    "--no_cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--datadir",
    type=str,
    default=r"/home/vault/iwi5/iwi5095h/patched-data",
    # default=r"/home/vault/iwi5/iwi5095h/Patch-final",
    help="dataset name (default: CNNImageDataset_256)",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    default="/home/woody/iwi5/iwi5095h/CNN-multiclass-classification-results/checkpoint/densenet",
    help="dataset name (default: CNNImageDataset_256)",
)
parser.add_argument(
    "--model", type=str, default="resnet18", help="model name (default: resnet18)"
)
parser.add_argument(
    "--optimizer", type=str, default="Adam", help="optimizer name (default: Adam)"
)
parser.add_argument(
    "--scheduler", type=str, default="StepLR", help="scheduler name (default: StepLR)"
)
parser.add_argument(
    "--loss",
    type=str,
    default="CrossEntropyLoss",
    help="loss name (default: CrossEntropyLoss)",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    metavar="N",
    help="number of workers (default: 4)",
)
parser.add_argument(
    "--num_classes",
    type=int,
    default=7,
    metavar="N",
    help="number of classes (default: 7)",
)

parser.add_argument(
    "--label_smoothing",
    type=float,
    default=0.1,
    metavar="N",
    help="label smoothing (default: 0.01)",
)
parser.add_argument(
    "--dropout", type=float, default=0.2, metavar="N", help="dropout (default: 0.2)"
)

parser.add_argument("--num_folds", type=int, default=5, help="Number of folds")


class CustomTransform(object):
    def __call__(self, image_tensor):
        # image_np = np.array(image)
        # image_tensor = torch.from_numpy(image_np.astype(np.float32))
        image_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min()
        )
        image_tensor = image_tensor.permute(-1, 0, 1)

        return image_tensor


data_transforms = {
    "train": transforms.Compose([CustomTransform()]),
    "val": transforms.Compose([CustomTransform()]),
    "test": transforms.Compose([CustomTransform()]),
}


def train_model(
    args,
    dataloaders,
    model,
    criterion,
    optimizer,
    scheduler,
    fold,
    num_epochs,
    dataset_sizes,
):
    epoch_counter_train = []
    epoch_counter_val = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    since = time.time()
    print("---------training model for fold {}--------".format(fold))
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    fold_directory = os.path.join(args.ckpt_path, f"fold_{fold + 1}")
    os.makedirs(fold_directory, exist_ok=True)

    for epoch in range(args.num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)
        # wandb.log({"epoch": epoch})

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                train_loss.append(running_loss / dataset_sizes[phase])
                train_acc.append(running_corrects.cpu().double() / dataset_sizes[phase])
                epoch_counter_train.append(epoch)
            if phase == "val":
                val_loss.append(running_loss / dataset_sizes[phase])
                val_acc.append(running_corrects.cpu().double() / dataset_sizes[phase])
                epoch_counter_val.append(epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == "train":
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                # wandb.log({"train loss": epoch_loss, "train accuracy": epoch_acc})
            if phase == "val":
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                # wandb.log({"val loss": epoch_loss, "val accuracy": epoch_acc})

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print("saving best model first here")
                torch.save(
                    {
                        #'model_state_dict': best_model_wts,
                        "model_state_dict": model.state_dict()
                    },
                    fold_directory + "/" + str(epoch) + ".pt",
                )
                # torch.save(model,'resnet18_best_model_full_patch.pt')

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    print("Best val Acc: {:4f}".format(best_acc))
    plt.figure(1)
    plt.title("Training Vs Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    print("epoch_counter_train", epoch_counter_train)
    print("train_loss", train_loss)
    plt.plot(epoch_counter_train, train_loss, color="r", label="Training Loss")
    plt.plot(epoch_counter_val, val_loss, color="g", label="Validation Loss")
    plt.legend()
    plt.savefig(fold_directory + "/loss.png")
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    plt.figure(2)
    plt.title("Training Vs Validation Accuracies")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    print("epoch_counter_train", epoch_counter_train)
    print("train_acc", train_acc)
    plt.plot(epoch_counter_train, train_acc, color="r", label="Training Accuracy")
    plt.plot(epoch_counter_val, val_acc, color="g", label="Validation Accuracy")
    plt.legend()
    plt.savefig(fold_directory + "/accuracy.png")
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    model.load_state_dict(best_model_wts)
    return model


def test_model(args, dataloaders, model_ft, class_names, fold):
    print("---------testing model for fold {}--------".format(fold))
    from sklearn import metrics

    correct = 0
    total = 0
    labels_all = []
    predicted_all = []
    confusion_matrix = cm.ConfusionMeter(args.num_classes)
    conf_matrix = np.zeros((args.num_classes, args.num_classes))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(dataloaders["val"])):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_ft(inputs)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            confusion_matrix.add(predicted, labels)

            correct += (predicted == labels).sum().item()
            labels_all.extend(labels.cpu().numpy())
            predicted_all.extend(predicted.cpu().numpy())
            # Update the total count per class
            conf_matrix += metrics.confusion_matrix(
                labels.cpu().numpy(),
                predicted.cpu().numpy(),
                labels=np.arange(args.num_classes),
            )

        print(confusion_matrix.conf)

    print("Accuracy of the network on the test images: %d %%" % (100 * correct / total))
    # Calculate accuracy per class
    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # Print or use the accuracy per class
    for c, acc in enumerate(class_accuracy):
        print(f"Class {c}: Accuracy = {acc * 100:.2f}%")

    # Confusion matrix as a heatmap
    con_m = confusion_matrix.conf
    df_con_m = pd.DataFrame(
        con_m, index=[i for i in class_names], columns=[i for i in class_names]
    )
    sn.set(font_scale=1.1)
    sn.heatmap(
        df_con_m, annot=True, fmt="g", annot_kws={"size": 10}, cbar=False, cmap="Blues"
    )
    # save heatmap
    # get precision, recall, f1score
    precision_each_class = metrics.precision_score(
        labels_all, predicted_all, average=None
    )
    recall_each_class = metrics.recall_score(labels_all, predicted_all, average=None)
    f1score_each_class = metrics.f1_score(labels_all, predicted_all, average=None)
    precision_macro = metrics.precision_score(
        labels_all, predicted_all, average="macro"
    )
    recall__macro = metrics.recall_score(labels_all, predicted_all, average="macro")
    f1score__macro = metrics.f1_score(labels_all, predicted_all, average="macro")
    precision_weighted = metrics.precision_score(
        labels_all, predicted_all, average="weighted"
    )
    recall_weighted = metrics.recall_score(
        labels_all, predicted_all, average="weighted"
    )
    f1score_weighted = metrics.f1_score(labels_all, predicted_all, average="weighted")
    print("precision_each_class", precision_each_class)
    print("recall_each_class", recall_each_class)
    print("f1score_each_class", f1score_each_class)
    print("precision_macro", precision_macro)
    print("recall__macro", recall__macro)
    print("f1score__macro", f1score__macro)
    print("precision_weighted", precision_weighted)
    print("recall_weighted", recall_weighted)
    print("f1score_weighted", f1score_weighted)


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


def main():
    args = parser.parse_args()
    image_datasets = {
        x: CustomImageFolder(os.path.join(args.datadir, x), data_transforms[x])
        for x in ["train", "val"]
    }

    dataset = ConcatDataset([image_datasets["train"], image_datasets["val"]])
    print("number of train images", len(image_datasets["train"]))
    print("number of val images", len(image_datasets["val"]))

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    data = [s[0] for data in dataset.datasets for s in data.samples]
    label = [s[1] for data in dataset.datasets for s in data.samples]

    data = np.array(data)
    label = np.array(label)

    for fold, (train_ids, val_ids) in enumerate(skf.split(data, label)):
        print(f"Fold {fold + 1} of {args.num_folds} is running...")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=train_subsampler
        )
        valloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=test_subsampler
        )
        dataloaders = {"train": trainloader, "val": valloader}

        # model_ft = models.resnet18(pretrained=True)
        # # set all the weights to be learnable
        # for param in model_ft.parameters():
        #     param.requires_grad = True

        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, args.num_classes)

        # efficinetnet
        # model_ft = models.efficientnet_b0(pretrained=True)
        model_ft = models.densenet121(pretrained=True, progress=True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, 7)
        for param in model_ft.parameters():
            param.requires_grad = True

        model_ft = model_ft.to(device)
        optimizer_ft = optim.Adam(
            model_ft.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        num_parameters_way_1 = sum(
            p.numel() for p in model_ft.parameters() if p.requires_grad
        )
        print("number of trainable parameters way 1", num_parameters_way_1)

        model_parameters = filter(lambda p: p.requires_grad, model_ft.parameters())
        params_way_2 = sum([np.prod(p.size()) for p in model_parameters])
        print("number of trainable parameters way 2", params_way_2)

        model_ft = train_model(
            args,
            dataloaders,
            model_ft,
            criterion,
            optimizer_ft,
            exp_lr_scheduler,
            fold=fold,
            num_epochs=args.num_epochs,
            dataset_sizes=dataset_sizes,
        )
        # save model_ft
        print("saving model")
        fold_directory = os.path.join(args.ckpt_path, f"fold_{fold + 1}")
        os.makedirs(fold_directory, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model_ft.state_dict(),
            },
            os.path.join(fold_directory, "model_ft.pt"),
        )
        # Test the model
        test_model(args, dataloaders, model_ft, class_names, fold=fold)


if __name__ == "__main__":
    main()
