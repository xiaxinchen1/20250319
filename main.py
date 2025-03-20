import cv2
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torchvision.models import Inception_V3_Weights, ResNet18_Weights

# 1. 设备配置：自动选择 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# 2. 数据预处理函数：对眼底图像进行 CLAHE 增强
def enhance_fundus_image(img: Image.Image) -> Image.Image:
    lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_enhanced = clahe.apply(L)
    lab_enhanced = cv2.merge((L_enhanced, A, B))
    img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_enhanced)

# 3. 数据集类：读取左右眼图像及标签
class FundusPairDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        df: 包含图像路径和标签的 DataFrame，
            必须包含 'left_eye_path', 'right_eye_path', 'label' 三列
        transform: 图像预处理转换（例如 Resize、归一化）
        """
        self.left_paths = df["left_eye_path"].tolist()
        self.right_paths = df["right_eye_path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        left_img = Image.open(self.left_paths[idx]).convert('RGB')
        right_img = Image.open(self.right_paths[idx]).convert('RGB')
        # 先进行 CLAHE 增强
        left_img = enhance_fundus_image(left_img)
        right_img = enhance_fundus_image(right_img)
        # 再进行 transform 预处理
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        label = self.labels[idx]
        return left_img, right_img, label

# 4. 数据加载与 DataLoader 构造
def prepare_dataloaders():
    excel_path = "/home/ubuntu/data/Traning_Dataset.xlsx"  # 请根据实际路径修改
    df = pd.read_excel(excel_path)
    # 构造左右眼图像完整路径（根据实际存放位置修改）
    df['left_eye_path'] = "/home/ubuntu/data/left/" + df['Left-Fundus'].astype(str)
    df['right_eye_path'] = "/home/ubuntu/data/right/" + df['Right-Fundus'].astype(str)
    # 根据后面 8 列 (N, D, G, C, A, H, M, O) 得到标签，假设只有一列为 1
    df['label'] = df[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']].values.argmax(axis=1)
    # 拆分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # 统一调整为 299x299 以满足 Inception 模型要求
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FundusPairDataset(train_df, transform=transform)
    test_dataset = FundusPairDataset(test_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    print("训练集样本数：", len(train_dataset))
    print("测试集样本数：", len(test_dataset))
    return train_loader, test_loader

# 5. 模型定义：多模型融合网络
class MultiModelFusionNet(nn.Module):
    def __init__(self, model_list: list, feature_dims: list, num_classes: int):
        super(MultiModelFusionNet, self).__init__()
        self.models = nn.ModuleList(model_list)
        self.classifier = nn.Linear(sum(feature_dims), num_classes)

    def forward(self, x):
        feats = []
        for mdl in self.models:
            f = mdl(x)  # 期望每个模型输出形状 [B, feature_dim, 1, 1]
            f = f.view(f.size(0), -1)  # 转换为 [B, feature_dim]
            feats.append(f)
        fused_feat = torch.cat(feats, dim=1)
        return self.classifier(fused_feat)

# 6. Inception 分支包装器：确保加载预训练权重时 aux_logits=True，并调整输出形状
class InceptionWrapper(nn.Module):
    def __init__(self):
        super(InceptionWrapper, self).__init__()
        # 加载时 aux_logits=True 以加载官方预训练权重
        self.inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        # 替换 fc 层为 Identity，使输出为池化层后的特征向量
        self.inception.fc = nn.Identity()
        # 添加自适应池化层，将输出调整为 [B, 2048, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        out = self.inception(x)
        # 如果返回 tuple，则只取主输出
        if isinstance(out, tuple):
            out = out[0]
        # 如果输出为二维 [B, 2048]，则转换为 [B, 2048, 1, 1]
        if out.dim() == 2:
            out = out.unsqueeze(-1).unsqueeze(-1)
        else:
            out = self.avgpool(out)
        return out

# 7. 构建多模型融合网络：包含 ResNet18 和 Inception 两个分支
def build_model():
    # ResNet18 分支
    resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # 输出：[B, 512, 1, 1]

    # Inception 分支使用包装器
    inception = InceptionWrapper()  # 输出：[B, 2048, 1, 1]

    # 构造融合模型，更新 feature_dims 为 [512, 2048]，总计 2560
    model_fusion = MultiModelFusionNet(
        model_list=[resnet, inception],
        feature_dims=[512, 2048],
        num_classes=8
    )
    model_fusion.to(device)
    return model_fusion

# 8. 训练与评估流程
def train_and_evaluate():
    train_loader, test_loader = prepare_dataloaders()
    model_fusion = build_model()

    # 定义损失和优化器
    class_weights = torch.ones(8).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model_fusion.parameters(), lr=1e-4)
    num_epochs = 50

    # 训练过程
    for epoch in range(num_epochs):
        model_fusion.train()
        running_loss = 0.0
        for left_imgs, right_imgs, labels in train_loader:
            left_imgs, right_imgs = left_imgs.to(device), right_imgs.to(device)
            labels = labels.to(device)
            # 拼接左右眼图像（沿宽度方向），输入尺寸由 [B, 3, 299, 299] 变为 [B, 3, 299, 598]
            inputs = torch.cat([left_imgs, right_imgs], dim=3)
            outputs = model_fusion(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # 评估过程：单次遍历测试集，收集预测结果、标签和预测概率
    model_fusion.eval()
    y_true, y_pred, all_probs = [], [], []
    with torch.no_grad():
        for left_imgs, right_imgs, labels in test_loader:
            left_imgs, right_imgs = left_imgs.to(device), right_imgs.to(device)
            labels = labels.to(device)
            inputs = torch.cat([left_imgs, right_imgs], dim=3)
            outputs = model_fusion(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    all_probs = np.vstack(all_probs)

    # 显示混淆矩阵和分类报告
    class_names = ["Normal", "Diabetic", "Glaucoma", "Cataract", "AMD", "Hypertension", "Myopia", "Other"]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(8)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(include_values=True, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

    print(classification_report(y_true, y_pred, target_names=class_names))

    # 计算 ROC-AUC
    y_true_onehot = np.eye(8)[y_true]
    for i, cls in enumerate(class_names):
        auc = roc_auc_score(y_true_onehot[:, i], all_probs[:, i])
        print(f"{cls} AUC: {auc:.3f}")
    macro_auc = roc_auc_score(y_true_onehot, all_probs, average="macro")
    print(f"Macro-AUC: {macro_auc:.3f}")

if __name__ == '__main__':
    # Windows 下多进程 DataLoader 需放在 main 函数中启动
    train_and_evaluate()
