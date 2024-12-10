import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt


# 定义神经网络模型
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x

# 加载数据
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)

# 评估模型
def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28*28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total

# 训练模型并保存权重
def train_and_save_model():
    # 加载数据
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    
    # 定义模型
    net = Net()
    
    # 打印初始准确率
    print("初始准确率:", evaluate(test_data, net))
    
    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 10  # 可以根据需要调整训练轮数
    for epoch in range(num_epochs):
        for (x, y) in train_data:
            net.zero_grad()
            output = net.forward(x.view(-1, 28*28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
         
        # 每个epoch后打印准确率
        accuracy = evaluate(test_data, net)
        print(f"Epoch {epoch+1}, 准确率: {accuracy:.4f}")
    
    # 保存模型权重
    model_path = "mnist_net.pth"
    torch.save(net.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# 手写板应用程序
class HandwritingApp:
    def __init__(self, root, net):
        self.root = root
        self.net = net
        self.root.title("手写数字识别画板")
        
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()
        
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.button_clear = tk.Button(root, text="清除", command=self.clear_canvas)
        self.button_clear.pack()
        
        self.button_predict = tk.Button(root, text="识别", command=self.predict_digit)
        self.button_predict.pack()

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=20)
        self.draw.ellipse([x1, y1, x2, y2], fill="black", width=20)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        # 调整图像大小以匹配MNIST数据集的尺寸
        image_resized = self.image.resize((28, 28), Image.LANCZOS).convert('L')
        # 反转颜色
        image_inverted = ImageOps.invert(image_resized)
        # 转换为numpy数组
        image_array = np.array(image_inverted) / 255.0
        # 将图像转换为PyTorch张量
        image_tensor = torch.tensor(image_array, dtype=torch.float).view(-1, 28*28)
        # 进行预测
        with torch.no_grad():
            output = self.net.forward(image_tensor)
            prediction = torch.argmax(output, dim=1).item()
        # 显示预测结果
        print(f"预测数字: {prediction}")

if __name__ == "__main__":
    # 如果没有预训练的模型，先训练模型
    model_path = "mnist_net.pth"
    
    try:
        # 检查模型文件是否存在且不为空
        if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
            raise FileNotFoundError
        
        net = Net()
        net.load_state_dict(torch.load(model_path, weights_only=True))  # 使用weights_only=True
        print("加载预训练模型")
    except FileNotFoundError:
        print("未发现预训练模型,正在训练模型...")
        train_and_save_model()
        net = Net()
        net.load_state_dict(torch.load(model_path, weights_only=True))  # 使用weights_only=True
    
    net.eval()  # 设置模型为评估模式
    
    # 创建主窗口
    root = tk.Tk()
    app = HandwritingApp(root, net)
    root.mainloop()