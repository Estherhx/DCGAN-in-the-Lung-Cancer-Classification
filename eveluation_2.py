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

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

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


train_dir = "autodl-tmp/mydata/lung_train_full_c"
test_dir = "autodl-tmp/mydata/lung_test_c"


train_csv = "autodl-tmp/mydata/train_full_data.csv"
test_csv = "autodl-tmp/mydata/test_data.csv"

best_config = {'num_classes': 3, 'num_epochs': 30, 'lr': 0.0002113557819790212, 'batch_size': 32}
batch_size = best_config["batch_size"]
num_epochs = best_config['num_epochs']

model = LungCancerNet(best_config['num_classes'])  # 这里假设您有3个类
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. 加载权重
model_weights = torch.load("best_model.pth")

# 3. 将这些权重设置到模型中
model.load_state_dict(model_weights)
model.eval()  # 设置为评估模式，这会关闭dropout等

# 接下来，您可以使用该模型在测试集上进行评估
test_dataset = myDataset(test_csv, test_dir, transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
test_predictions = []
test_labels = []
test_scores = [] 
classes = [0,1,2]
       
     
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        test_scores.extend(outputs.cpu().numpy())  # 这里我们收集模型的原始输出
        _, predicted = torch.max(outputs.data, 1)
        test_predictions.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        
test_labels_one_hot = label_binarize(test_labels, classes=classes)
n_classes = len(classes)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_labels_one_hot[:, i], np.array(test_scores)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 保存ROC曲线为PNG图像
plt.figure()
lw = 2  # 线的宽度
colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.savefig("multi_class_ROC.png")  # 保存为PNG格式
plt.close()  # 关闭图像窗口

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    plt.gca().set_aspect('equal', adjustable='box')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]), int(cm[i, j])
        plt.text(j, i, num[1], verticalalignment='center', horizontalalignment="center",
                 color="white" if num[1] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png", bbox_inches='tight')
    plt.close()

# Generate Confusion Matrix
conf_matrix = confusion_matrix(test_labels, test_predictions)
plot_confusion_matrix(conf_matrix, classes)

# 使用混淆矩阵计算准确率、精确率召回率和F1分数
# conf_matrix = confusion_matrix(test_labels, test_predictions)
TPs = np.diag(conf_matrix)
FPs = np.sum(conf_matrix, axis=0) - TPs
FNs = np.sum(conf_matrix, axis=1) - TPs

# 计算准确率
accuracy = np.sum(TPs) / np.sum(conf_matrix)

epsilon = 1e-7
precision = TPs / (TPs + FPs+ epsilon)
recall = TPs / (TPs + FNs+ epsilon)
f1 = 2 * (precision * recall) / (precision + recall)

# 计算宏平均和微平均值，宏平均（Macro-Averaging）：先为每一类计算指标，然后计算这些指标的平均值。微平均（Micro-Averaging）：将所有类的TP、FP和FN加起来，然后计算准确率、召回率和F1分数。
precision_macro = np.mean(precision)
recall_macro = np.mean(recall)
f1_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro)

precision_micro = np.sum(TPs) / (np.sum(TPs) + np.sum(FPs))
recall_micro = np.sum(TPs) / (np.sum(TPs) + np.sum(FNs))
f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)


print("Accuracy:", accuracy)
print("\n")
print("Precision (Macro):", precision_macro)
print("Recall (Macro):", recall_macro)
print("F1 Score (Macro):", f1_macro)
print("\n")
print("Precision (Micro):", precision_micro)
print("Recall (Micro):", recall_micro)
print("F1 Score (Micro):", f1_micro)

