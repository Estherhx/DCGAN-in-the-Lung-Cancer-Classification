import itertools
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



class myDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        label = int(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class LungCancerNet(nn.Module):
    def __init__(self, num_classes):
        super(LungCancerNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_best_model(config):
    batch_size = config["batch_size"]
    num_epochs = config['num_epochs']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = LungCancerNet(num_classes=config["num_classes"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state_dict = None
    for epoch in range(config["num_epochs"]):
        model.train()
        total_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * images.size(0)

        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct / total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state_dict = model.state_dict()

    model.load_state_dict(best_model_state_dict)
    return model, train_losses, val_losses

# best_config= {'num_classes': 3, 'num_epochs': 2, 'lr': 0.00019830879157768872, 'batch_size': 64}
best_config = {'num_classes': 3, 'num_epochs': 2, 'lr': 0.0002113557819790212, 'batch_size': 32}

train_dir = "autodl-tmp/mydata/lung_train_full_c"
train_csv = "autodl-tmp/mydata/train_full_data_and_fake2_1.csv"
train_dataset = myDataset(train_csv, train_dir, transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
trained_model, train_losses, val_losses=train_best_model(best_config)
torch.save(trained_model.state_dict(), "best_model.pth")

# batch_size = best_config["batch_size"]
# num_epochs = best_config['num_epochs']


# trained_model.eval()
# test_predictions = []
# test_labels = []

# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to("cuda"), labels.to("cuda")
#         outputs = trained_model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         test_predictions.extend(predicted.cpu().numpy())
#         test_labels.extend(labels.cpu().numpy())

# # Generate Confusion Matrix
# conf_matrix = confusion_matrix(test_labels, test_predictions)


# # Plot Training and Validation Loss
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
# plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.savefig("loss.png", bbox_inches='tight')
# plt.close()
# classes = [0,1,2]
# def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
#     plt.figure(figsize=(8, 6))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=0)
#     plt.yticks(tick_marks, classes)

#     plt.gca().set_aspect('equal', adjustable='box')

#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         num = '{:.2f}'.format(cm[i, j]), int(cm[i, j])
#         plt.text(j, i, num[1], verticalalignment='center', horizontalalignment="center",
#                  color="white" if num[1] > thresh else "black")
    
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.savefig("confusion_matrix.png", bbox_inches='tight')
#     plt.close()

# plot_confusion_matrix(conf_matrix, classes)
# # 使用混淆矩阵计算准确率、召回率和F1分数
# # conf_matrix = confusion_matrix(test_labels, test_predictions)
# TPs = np.diag(conf_matrix)
# FPs = np.sum(conf_matrix, axis=0) - TPs
# FNs = np.sum(conf_matrix, axis=1) - TPs

# epsilon = 1e-7
# precision = TPs / (TPs + FPs+ epsilon)
# recall = TPs / (TPs + FNs+ epsilon)
# f1 = 2 * (precision * recall) / (precision + recall)

# # 计算宏平均和微平均值，宏平均（Macro-Averaging）：先为每一类计算指标，然后计算这些指标的平均值。微平均（Micro-Averaging）：将所有类的TP、FP和FN加起来，然后计算准确率、召回率和F1分数。
# precision_macro = np.mean(precision)
# recall_macro = np.mean(recall)
# f1_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro)

# precision_micro = np.sum(TPs) / (np.sum(TPs) + np.sum(FPs))
# recall_micro = np.sum(TPs) / (np.sum(TPs) + np.sum(FNs))
# f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)

# print("Precision (Macro):", precision_macro)
# print("Recall (Macro):", recall_macro)
# print("F1 Score (Macro):", f1_macro)
# print("\n")
# print("Precision (Micro):", precision_micro)
# print("Recall (Micro):", recall_micro)
# print("F1 Score (Micro):", f1_micro)

# plot_confusion_matrix(conf_matrix, classes)

