import os
import csv
import yaml
import random
import shutil

import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, cohen_kappa_score
from sklearn.preprocessing import label_binarize

from utils.utils import *
from utils.metrics import *
from dataloaders.dataloaders_aptos import DRDataloader


def train(model, train_loader, epoch, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_index = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_index = batch_idx

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return train_loss / batch_index, 100. * correct / total


def valid(model, valid_loader, epoch, criterion):
    global best_acc, config
    model.eval()
    valid_loss = 0
    correct = 0
    total = 0
    y_true = None
    y_pred = None
    labels = []
    pre_labels = []
    auc_output = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            outputs = model(inputs)
            auc_output.extend(outputs.data.cpu().numpy())
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            labels.extend(targets.data.cpu())
            pre_labels.extend(predicted.data.cpu())
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            y_true, y_pred = assemble_labels(y_true, y_pred, targets, outputs)

            progress_bar(batch_idx, len(valid_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (valid_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        save_checkpoint(
            model,
            acc,
            epoch,
            f'{sys.path[0]}/{config["model"]["folder"]}/{config["model"]["name"]}_{config["train"]["seed"]}_{config["train"]["lr"]}.pth'
        )
    return valid_loss / batch_idx, 100. * correct / total


# Testing
def test(test_loader, criterion):
    global best_acc, log_file, config
    print("begin test")
    model = torch.load(
        f'{sys.path[0]}/{config["model"]["folder"]}/{config["model"]["name"]}_{config["train"]["seed"]}_{config["train"]["lr"]}.pth'
    )
    model = checkpoint['net']
    best_acc = checkpoint['acc']
    epoch = checkpoint['epoch']
    torch.set_rng_state(checkpoint['rng_state'])

    if use_cuda:
        model.cuda()
        print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.deterministic = True
        print('Using CUDA..')

    model.eval()
    print(epoch)
    valid_loss = 0
    correct = 0
    total = 0
    y_true = None
    y_pred = None
    labels = []
    pre_labels = []
    auc_output = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            auc_output.extend(outputs.data.cpu().numpy())
            loss = criterion(outputs, targets)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            labels.extend(targets.data.cpu())
            pre_labels.extend(predicted.data.cpu())
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            y_true, y_pred = assemble_labels(y_true, y_pred, targets, outputs)

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (valid_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    labels = np.reshape(labels, [-1, 1])
    pre_labels = np.reshape(pre_labels, [-1, 1])

    y_test_binarized = label_binarize(labels, classes=np.unique(labels))
    n_classes = y_test_binarized.shape[1]

    auc_output = np.array(auc_output)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], auc_output[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    auc_list = list(roc_auc.values())

    ece = get_ece(y_pred, y_true)
    bs = get_bs(y_pred, y_true)

    # label -->
    auc_labels = np.zeros(shape=(len(labels), config["train"]["num_classes"]))
    for i in range(len(labels)):
        auc_labels[i][labels[i]] = 1

    confusion = confusion_matrix(labels, pre_labels)

    with open(log_file, 'a') as f:
        f.write(classification_report(labels, pre_labels, digits=5))
        f.write(f"epoch: {epoch}\n")
        f.write(f"valid_acc: {best_acc}\n")
        f.write(f"kappa: {cohen_kappa_score(labels, pre_labels)}\n")
        f.write(f"auc average: {np.mean(auc_list)}\n")
        f.write(f"auc std: {np.std(auc_list)}\n")
        f.write(f"ece: {ece.item()}\n")
        f.write(f"bs: {bs}\n")
        f.write(f'confusion matrix: \n{confusion}\n')


def prepare_task(config):
    global best_acc, use_cuda, log_file, config_name
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['train']['cuda'])
    best_acc = 0
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(config["train"]["seed"])
    np.random.seed(config["train"]["seed"])
    random.seed(config["train"]["seed"])

    if not os.path.isdir(config["model"]["folder"]):
        os.makedirs(config["model"]["folder"])

    if not os.path.isdir(config["log"]["folder"]):
        os.makedirs(config["log"]["folder"])

    shutil.copyfile(
        f'{sys.path[0]}/configs/{config_name}',
        f'{sys.path[0]}/{config["log"]["folder"]}/{config["model"]["name"]}_{config["train"]["seed"]}_{config["train"]["lr"]}.log'
    )

    log_file = f'{sys.path[0]}/{config["log"]["folder"]}/{config["model"]["name"]}_{config["train"]["seed"]}_{config["train"]["lr"]}.csv'
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            log_writer = csv.writer(f, delimiter=',')
            log_writer.writerow(['epoch', 'train loss', 'train acc', 'valid loss', 'valid acc'])


def main():
    global best_acc, log_file, config, config_name
    config_name = 'config_aptos.yaml'
    with open(f'configs/{config_name}', 'rb') as f:
        config = yaml.safe_load(f.read())

    prepare_task(config)

    dataloader = DRDataloader(
        batch_size=config["train"]["batch-size"],
        num_workers=config["train"]["num_worker"],
        img_resize=config["train"]["size"],
        root_dir=config["train"]["data_path"],
    )

    (test_loader, test_dataset,) = dataloader.run("test")
    (valid_loader, valid_dataset,) = dataloader.run("valid")
    (train_loader, train_dataset,) = dataloader.run("train")

    # model initial
    exec('from models.{} import *'.format(config["model"]["file"]))
    model = eval(f'{config["model"]["caller"]}')

    model = torch.nn.DataParallel(model).cuda()
    cudnn.deterministic = True
    print('Using', torch.cuda.device_count(), 'GPUs.')
    print('Using CUDA..')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["train"]["lr"], momentum=0.9,
                          weight_decay=config["train"]["decay"])

    if config["train"]["resume"]:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(
            os.path.join(sys.path[0], config["model"]["folder"])), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(
            f'{sys.path[0]}/{config["model"]["folder"]}/{config["model"]["name"]}_{config["train"]["seed"]}_{config["train"]["lr"]}.pth')
        model = checkpoint['net']
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    else:
        print('==> Building model...')
        start_epoch = 0

    for epoch in range(start_epoch, config["train"]["epochs"]):
        adjust_learning_rate(optimizer, epoch, config["train"]["lr"])
        train_loss, train_acc = train(model, train_loader, epoch, criterion, optimizer)
        valid_loss, valid_acc = valid(model, valid_loader, epoch, criterion)
        with open(log_file, 'a') as f:
            log_writer = csv.writer(f, delimiter=',')
            log_writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc])

    test(test_loader, criterion)


if __name__ == '__main__':
    print(sys.path[0])
    main()




