以下是更详细的 README 文件示例，包括 Python 版本、环境设置和使用示例：

```markdown
# CIFAR-100 图像分类

本项目使用卷积神经网络（CNN）和 Transformer 模型对 CIFAR-100 数据集进行图像分类。通过数据增强技术（如 CutMix）提高模型的性能。

## 项目结构

```
.
├── data/               # 数据集存储路径
├── models/             # 存储模型权重
├── requirements.txt    # 依赖包
└── train.py            # 训练和测试脚本
```

## 依赖项

### Python 版本

本项目使用 Python 3.8.10 进行开发。请确保您的环境与此版本兼容。

### 安装依赖

在运行项目之前，请确保安装以下依赖项。可以通过以下命令安装：

```bash
pip install -r requirements.txt
```

`requirements.txt` 文件内容：

```
torch==2.4.1+cu121
torchvision==0.19.1+cu121
numpy==1.24.4

```

## 数据集

本项目使用 [CIFAR-100 数据集](https://www.cs.toronto.edu/~kriz/cifar.html)。在首次运行时，程序将自动下载数据集并存储在 `data/` 目录中。

## 训练模型

您可以使用以下命令进行模型的训练：

```bash
python train.py
```

### 训练过程

训练过程包括以下步骤：

1. **加载数据集**：从 `data/` 目录加载 CIFAR-100 数据集，并进行预处理。
2. **定义模型**：包括 CNN 和 Transformer 模型的定义。
3. **选择优化器**：使用 Adam 优化器和交叉熵损失函数进行训练。
4. **数据增强**：实现 CutMix 数据增强技术以提升模型性能。
5. **训练与验证**：在每个 epoch 结束时验证模型并记录损失和准确率。

### 参数设置

在 `train.py` 中，您可以调整以下参数：

- `batch_size`：每次训练的样本数量（默认为 128）。
- `learning_rate`：学习率（默认为 0.001）。
- `epochs`：训练的总轮数（默认为 100）。

### 训练示例

训练完成后，模型权重将保存在 `models/cnn_model.pth` 和 `models/transformer_model.pth` 中。

## 测试模型

训练完成后，您可以使用保存的模型权重进行测试。以下是如何加载模型并进行推断的示例代码：

```python
import torch
from models import CNN, TransformerClassifier

# 加载模型
model_cnn = CNN()
model_cnn.load_state_dict(torch.load('models/cnn_model.pth'))
model_cnn.eval()

model_transformer = TransformerClassifier(num_classes=100)
model_transformer.load_state_dict(torch.load('models/transformer_model.pth'))
model_transformer.eval()

# 测试数据加载示例
from torchvision import datasets, transforms

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# 进行推断
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model_cnn(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f'CNN 测试准确率: {accuracy:.2f}%')
```

## 可视化结果

在训练过程中，使用 TensorBoard 可视化训练和验证的损失和准确率。可以使用以下命令启动 TensorBoard：

```bash
tensorboard --logdir=runs
```

在浏览器中访问 [http://localhost:6006](http://localhost:6006) 查看结果。

## 贡献

欢迎对本项目提出贡献建议或报告问题！请提交 PR 或 issue。


```

您可以根据需要进一步修改和扩展这个 README 文件，确保它包含了所有必要的信息。希望这能帮助您更好地记录和展示您的项目！