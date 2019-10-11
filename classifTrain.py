from resnet import *
from classifiers import *
from torch import nn, optim
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torchvision.transforms as transforms



filenames = ['baby_r1.txt',  'bird_r1.txt', 'car_r1.txt', 'clouds_r1.txt', 'dog_r1.txt',
             'female_r1.txt',  'flower_r1.txt',  'male_r1.txt',  'night_r1.txt', 'people_r1.txt',
             'portrait_r1.txt',  'river_r1.txt',  'sea_r1.txt',  'tree_r1.txt']



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main(args):
    model = resnet18(num_classes=14)
    if use_gpu:
        model = model.cuda()
    lossfn = nn.BCELoss()
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
            if i % 50 == 49:  # print every 2000 mini-batches
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
    labels_lst = []
    outputs_lst = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data['image'], data['label']
            outputs = model(images)
            losses.append(loss_fn(outputs, labels).item())
            labels_lst.append(labels.cpu().numpy())
            outputs_lst.append(outputs.cpu().numpy())

    labels, outputs = np.concatenate(labels_lst), np.concatenate(outputs_lst)
    np.save("labels.npy", labels)
    np.save("outputs.npy", outputs) 
    cm_dict, total_acc = confusion_matrix(labels, outputs)
    print("Avg loss on test set: {}".format(sum(losses)/len(losses)))
    print("Total accuracy on test set: {}".format(total_acc))
    with open(args.save_name, 'w') as fp:
        json.dump(cm_dict, fp)

def confusion_matrix(labels, outputs):
    results = [[] for _ in range(len(filenames))]
    tp = [0 for _ in range(len(filenames))]
    tn = [0 for _ in range(len(filenames))]
    fp = [0 for _ in range(len(filenames))]
    fn = [0 for _ in range(len(filenames))]
    for i in range(len(labels)):
        ll, ol = labels[i], outputs[i]
        for j in range(len(ll)):
            if ll[j] == 1 and ol[j] >= .2:
                tp[j] += 1
                results[j].append(1)
            elif ll[j] == 0 and ol[j] < .2:
                tn[j] += 1
                results[j].append(1)
            elif ll[j] == 0 and ol[j] >= .2:
                fp[j] += 1
                results[j].append(0)
            elif ll[j] == 1 and ol[j] < .2:
                fn[j] += 1
                results[j].append(0)
    accs = [sum(r) / len(r) for r in results]
    total_acc = sum([sum(r) for r in results]) / sum([len(r) for r in results])
    cm_dict = {filenames[i] : {'tp': tp[i], 'tn': tn[i], 'fp': fp[i],
                               'fn': fn[i], 'accuracy' : accs[i]} for i in range(len(filenames))}
    return cm_dict, total_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--train_names", type=str)
    parser.add_argument("--test_names", type=str)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--save_name", type=str)
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()
    print(use_gpu)

    trans = transforms.Compose([transforms.ToTensor()])
    train_set = DiffuserDatasetClassif(args.train_names, args.image_dir, args.gt_file, transform=trans, use_gpu=use_gpu)
    test_set = DiffuserDatasetClassif(args.test_names, args.image_dir, args.gt_file, transform=trans, use_gpu=use_gpu)

    main(args)
