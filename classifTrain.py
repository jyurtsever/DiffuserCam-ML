from resnet import *
from classifiers import *
from torch import nn, optim
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse




def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main(args):
    model = resnet18(num_classes=14)
    if use_gpu:
        model = model.cuda()
    lossfn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train(model, optimizer, lossfn, args.num_epochs)
    test(model)

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
                inputs, labels = inputs.cuda(), labels
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

def test(model):
    correct_test = 0
    total = 0
    testloader = torchdata.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if use_gpu:
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct_test / total))
    correct_train = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if use_gpu:
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_train += (predicted == labels).sum().item()

    print('Accuracy of the network on the 50000 train images: %d %%' % (
            100 * correct_train / total))

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


    train_set = DiffuserDatasetClassif(args.train_names, args.image_dir, args.gt_file)
    test_set = DiffuserDatasetClassif(args.test_names, args.image_dir, args.gt_file)

    main(args)
