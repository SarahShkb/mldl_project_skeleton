import wandb
import torch
from torch import nn
from utils import CustomNet, validate
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

def evaluation(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # todo...
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    return loss


wandb.init(project="lab3")
config = wandb.config
config.learning_rate = 0.01

model = CustomNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
transform = T.Compose([
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
tiny_imagenet_dataset_evaluation = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/evaluate', transform=transform)

train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_evaluation, batch_size=32, shuffle=True, num_workers=8)
best_acc = 0
loss = 0

# Run the training process for {num_epochs} epochs
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    loss = evaluation(epoch, model, train_loader, criterion, optimizer)



print(f'Best validation accuracy: {best_acc:.2f}%')

for i in range(10):
    wandb.log({"loss":loss})

