from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import copy
import pandas as pd
import torchnet.meter.confusionmeter as cm
from model import FCN
import seaborn as sn


from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

from torch.utils.data import Dataset


import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

# os.environ["WANDB_MODE"] = "offline"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import wandb

wandb_ = False

# create a argparser
import argparse

parser = argparse.ArgumentParser(description="FCN for line profile")
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
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
    default=0.001,
    metavar="LR",
    help="learning rate (default: 0.001)",
)

# 0.001
parser.add_argument(
    "--weight_decay", default=1e-5, metavar="LR", help="learning rate (default: 0.001)"
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
    # default=r"/home/vault/iwi5/iwi5095h/all-new-augmentation-original-standardized.csv",
    default=r"/home/vault/iwi5/iwi5095h/mean_std_transform_dataset.csv",
    help="dataset name (default: CNNImageDataset_256)",
)
parser.add_argument(
    "--model", type=str, default="fcn", help="model name (default: resnet18)"
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
    default=2,
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

parser.add_argument(
    "--ckpt_path",
    type=str,
    default="/home/woody/iwi5/iwi5095h/line_profile/kfold/binary",
    metavar="N",
    help="checkpoint path (default: None)",
)


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)  # Assuming X_paths is in X[0]

    def __getitem__(self, index):
        X_data = torch.tensor(
            np.array((self.X)[index, 1:].tolist()), dtype=torch.float32, device=device
        ).unsqueeze(0)
        y_data = torch.tensor(
            self.y[index],
            dtype=torch.float32,
            device=device,
        )
        X_path = self.X[index, 0]

        return X_data, y_data, X_path


import os


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
    print("---------training model--------")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    output_before_fc = pd.DataFrame()
    fold_directory = os.path.join(args.ckpt_path, f"fold_{fold + 1}")
    os.makedirs(fold_directory, exist_ok=True)
    for epoch in range(args.num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)
        if wandb_:
            wandb.log({"epoch": epoch})

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels, path in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    # _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (
                    torch.argmax(outputs, 1) == torch.argmax(labels, 1)
                ).sum()

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
                if wandb_:
                    wandb.log({"train loss": epoch_loss, "train accuracy": epoch_acc})
            if phase == "val":
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                if wandb_:
                    wandb.log({"val loss": epoch_loss, "val accuracy": epoch_acc})

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the best model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print("saving best model first here")
                torch.save(
                    {"model_state_dict": model.state_dict()},
                    fold_directory + "/" + str(epoch) + ".pt",
                )

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))
    plt.cla()
    plt.clf()
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
    print("---------testing model--------")
    correct = 0
    total = 0
    labels_all = []
    predicted_all = []
    confusion_matrix = cm.ConfusionMeter(args.num_classes)
    conf_matrix = np.zeros((args.num_classes, args.num_classes))
    from sklearn import metrics

    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(dataloaders["val"]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_ft(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            confusion_matrix.add(predicted, labels)

            correct += (
                (torch.argmax(outputs, 1) == torch.argmax(labels, 1)).sum().item()
            )
            # correct += (predicted == labels).sum().item()

            labels_all.extend(torch.argmax(labels, 1).cpu().numpy())
            predicted_all.extend(predicted.cpu().numpy())
            conf_matrix += metrics.confusion_matrix(
                torch.argmax(labels, 1).cpu().numpy(),
                predicted.cpu().numpy(),
                labels=np.arange(args.num_classes),
            )
        print(confusion_matrix.conf)

    print("Accuracy of the network on the test images: %d %%" % (100 * correct / total))
    # calculate precision recall f1score
    from sklearn import metrics

    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    if wandb_:
        wandb.log({"test accuracy": (100 * correct / total)})
        class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # Print or use the accuracy per class
    for c, acc in enumerate(class_accuracy):
        print(f"Class {c}: Accuracy = {acc * 100:.2f}%")

    # wandb.log({"test accuracy": (100 * correct / total)})

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
    # wandb.log({"precision": precision, "recall": recall, "f1score": f1score})
    print("precision_each_class", precision_each_class)
    print("recall_each_class", recall_each_class)
    print("f1score_each_class", f1score_each_class)
    print("precision_macro", precision_macro)
    print("recall__macro", recall__macro)
    print("f1score__macro", f1score__macro)
    print("precision_weighted", precision_weighted)
    print("recall_weighted", recall_weighted)
    print("f1score_weighted", f1score_weighted)


def create_dataset(args):
    df = pd.read_csv(args.datadir)
    label_Column = "5217"
    X_train = df.drop(label_Column, axis=1).values

    y_train = df[label_Column].values
    from sklearn.preprocessing import OneHotEncoder

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    ohe = ohe.fit(y_train.reshape(-1, 1))

    y_train = ohe.transform(y_train.reshape(-1, 1))
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    dataset = {
        "train": CustomDataset(X_train, y_train),
        "val": CustomDataset(X_val, y_val),
    }
    return ohe, dataset


def main():
    args = parser.parse_args()
    skf = StratifiedKFold(n_splits=2, shuffle=True)

    # ohe, line_dataset = create_dataset(args)
    # weight = torch.tensor(class_weights, dtype=torch.float32, device=device)

    df = pd.read_csv(args.datadir)
    label_Column = "5217"
    data = df.drop(label_Column, axis=1).values

    label = df[label_Column].values

    for i in range(len(label)):
        if label[i] == "good_images":
            label[i] = "good"
        else:
            label[i] = "defective"

    class_names = list(set(label))

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    ohe = ohe.fit(label.reshape(-1, 1))

    y_train = ohe.transform(label.reshape(-1, 1))

    dataset = CustomDataset(data, y_train)

    for fold, (train_ids, val_ids) in enumerate(skf.split(data, label)):
        print(f"Fold {fold + 1} of {5} is running...")

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

        input_size = 1
        output_size = args.num_classes

        train_size = len(train_ids)
        val_size = len(val_ids)
        dataset_sizes = {"train": train_size, "val": val_size}

        model_ft = FCN.CustomModel3(input_size, output_size)
        print(model_ft)

        dir = r"/home/woody/iwi5/iwi5095h/line_profile/"

        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(
            model_ft.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )

        if wandb_:
            wandb.init(
                project="thesis-lineprofile",
                dir=dir,
                config={
                    "optimizer": optimizer_ft,
                    "criteria": criterion,
                    "learning_rate": args.lr,
                    "architecture": model_ft,
                    "dataset": args.datadir,
                    "epochs": args.num_epochs,
                    "batch-size": args.batch_size,
                    # "label-smoothing": args.label_smoothing,
                    "dropout": args.dropout,
                    "weight-decay": args.weight_decay,
                },
            )
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model_ft = train_model(
            args,
            dataloaders,
            model_ft,
            criterion,
            optimizer_ft,
            exp_lr_scheduler,
            fold,
            num_epochs=args.num_epochs,
            dataset_sizes=dataset_sizes,
        )
        # save model_ft
        # torch.save(
        #     {"model_state_dict": model_ft.state_dict()},
        #     args.ckpt_path + "/" + args.model + "_model.pt",
        # )
        # print("saving model")

        # checkpoint = torch.load(
        #     "/home/woody/iwi5/iwi5095h/line_profile/Results/results/mean-std/fcn/fcn_model.pt",
        #     map_location=torch.device("cpu"),
        # )

        # model_ft.load_state_dict(checkpoint["model_state_dict"])

        test_model(args, dataloaders, model_ft, class_names, fold=fold)


if __name__ == "__main__":
    main()
