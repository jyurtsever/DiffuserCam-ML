from resnet import *
from classifiers import *
from torch import nn, optim
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torchvision.transforms as transforms



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main(args):
    model = resnet18(num_classes=14)
    if use_gpu:
        model = model.cuda()
    lossfn = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters())
    train(model, optimizer, lossfn, args.num_epochs)
    test(model, lossfn)

def train(model, optimizer, lossfn, num_epochs):
    losses = []
    iterations = []
    epoch = 1
    trainloader = torchdata.DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    j = 0
    for epoch in range(num_epochs):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data['image'], data['label']
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            # print(inputs.shape, labels.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            # print(outputs)
            loss = lossfn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:  # print every 2000 mini-batches
                j += 500
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 500))
                losses.append(running_loss / 500)
                iterations.append(j)
                running_loss = 0.0

    print('Finished Training')
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    # }, "saved_network.pt")
    # np.save("losses.npy", np.array(losses))
    # np.save("iterations.npy", iterations)
    return model, losses, iterations

def test(model, loss_fn):
    correct_test = 0
    total = 0
    testloader = torchdata.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    losses = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data['image'], data['label']
            if use_gpu:
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            losses.append(loss_fn(outputs, labels).item())
    np.save("labels.npy", labels.cpu().numpy())
    np.save("outputs.npy", outputs.cpu().numpy())
    print("Avg loss: {}".format(sum(losses)/len(losses)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--train_names", type=str)
    parser.add_argument("--test_names", type=str)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()
    print(use_gpu)

    trans = transforms.Compose([transforms.ToTensor()])
    train_set = DiffuserDatasetClassif(args.train_names, args.image_dir, args.gt_file, transform=trans)
    test_set = DiffuserDatasetClassif(args.test_names, args.image_dir, args.gt_file, transform=trans)

    main(args)
