from denoise.resnet import *
from classif.classifiers import *
from torch import nn, optim
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torchvision.transforms as transforms



num_images = 25000



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main(args):
    model = resnet18( pretrained=args.use_pretrained, num_classes=len(filenames))
    if use_gpu:
        model = model.cuda()
    lossfn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    trainloader = torchdata.DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    testloader = torchdata.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    train(model, optimizer, lossfn, args.num_epochs, trainloader, testloader)
    
    test(model, lossfn, testloader)

def train(model, optimizer, lossfn, num_epochs, trainloader, testloader):
    losses = []
    iterations = []
    epoch = 1
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
                j += 50
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 50))
                losses.append(running_loss / 50)
                iterations.append(j)
                running_loss = 0.0
        test(model, lossfn, trainloader, set_name='training set')
        test(model, lossfn, testloader)
    print('Finished Training')
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    # }, "saved_network.pt")
    np.save(args.save_name + "_losses.npy", np.array(losses))
    np.save(args.save_name +"_iterations.npy", iterations)
    return model, losses, iterations

def test(model, loss_fn, testloader, set_name='test set'):
    correct_test = 0
    total = 0
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
    cm_dict, total_acc = confusion_matrix(labels, outputs, thresh=.1)
    print("Avg loss on {}: {}".format(set_name, sum(losses)/len(losses)))
    print("Total accuracy on {}: {}".format(set_name, total_acc))
    with open(args.save_name, 'w') as fp:
        json.dump(cm_dict, fp)

def confusion_matrix(labels, outputs, thresh=.2):
    results = [[] for _ in range(len(filenames))]
    tp = [0 for _ in range(len(filenames))]
    tn = [0 for _ in range(len(filenames))]
    fp = [0 for _ in range(len(filenames))]
    fn = [0 for _ in range(len(filenames))]
    for i in range(len(labels)):
        ll, ol = labels[i], outputs[i]
        for j in range(len(ll)):
            if ll[j] == 1 and ol[j] >= thresh:
                tp[j] += 1
                results[j].append(1)
            elif ll[j] == 0 and ol[j] < thresh:
                tn[j] += 1
                results[j].append(1)
            elif ll[j] == 0 and ol[j] >= thresh:
                fp[j] += 1
                results[j].append(0)
            elif ll[j] == 1 and ol[j] < thresh:
                fn[j] += 1
                results[j].append(0)
    accs = [sum(r) / len(r) for r in results]
    total_acc = sum([sum(r) for r in results]) / sum([len(r) for r in results])
    cm_dict = {filenames[i] : {'tp': tp[i], 'tn': tn[i], 'fp': fp[i],
                               'fn': fn[i], 'accuracy' : accs[i]} for i in range(len(filenames))}
    return cm_dict, total_acc

def get_labels(filenames):
    num_class = len(filenames)
    gt = [[0 for _ in range(num_class)] for _ in range(0, num_images + 1)]
    for i, name in enumerate(filenames):
        f = open(args.ann_dir + name, 'r')
        for line in f:
            gt[int(line.strip())][i] = 1
        f.close()
    data = []
    for im_num, class_lst in enumerate(gt):
        im_filename = "im{:05}.tiff".format(im_num)
        data.append({im_filename : class_lst})
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-gt_file", type=str)
    parser.add_argument("-image_dir", type=str)
    parser.add_argument("-train_names", type=str)
    parser.add_argument("-test_names", type=str)
    parser.add_argument("-num_epochs", type=int)
    parser.add_argument("-batch_size", type=int)
    parser.add_argument("-save_name", type=str)
    parser.add_argument("-ann_dir", type=str, default="../mirflickr25k/annotations/")
    parser.add_argument('cats', metavar='N', type=str, nargs='+',
                        help='an integer for the accumulator')
    parser.add_argument("-suffix", type=str, default=".tiff")
    parser.add_argument("-use_pretrained", type=bool, default=False)
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    filenames = args.cats
    labels = get_labels(filenames)
    print(use_gpu)

    trans = transforms.Compose([transforms.ToTensor()])
    train_set = DiffuserDatasetClassif(args.train_names, args.image_dir, labels, suffix=args.suffix, transform=trans, use_gpu=use_gpu)
    test_set = DiffuserDatasetClassif(args.test_names, args.image_dir, labels, suffix=args.suffix, transform=trans, use_gpu=use_gpu)

    main(args)
