import argparse
import os
from pathlib import Path
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import datetime
import wandb  # Import wandb
from models.ModifiedLDG import load_base_mLDG
from mmcls.apis.inference import init_model

from mmcv import Config
#                  0          1        2        3        4       5         6          7
CLASS_NAMES = ['Anger', 'Disgust', 'Fear', 'Sad', 'Happy', 'Surprise', 'Neutral', 'Contempt']

now = datetime.datetime.now()
time_str = now.strftime("%m-%d-%H-%M-")

parser = argparse.ArgumentParser()

# Original script arguments
parser.add_argument('--data', type=str, default=None, help='Path to the dataset, i.e. root directory with train and test folders')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model.pth.tar')
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoint/'+time_str+'model_best.pth.tar')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', dest='lr')
parser.add_argument('--factor', default=0.1, type=float, metavar='FT')
parser.add_argument('--af', '--adjust-freq', default=10, type=int, metavar='N', help='adjust learning rate frequency')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default=False, action='store_true', help='evaluate model on test set')
parser.add_argument('--gpu', default='0', type=str)

# mLDG training arguments
parser.add_argument('--apvit_config', type=str, default='./configs/apvit/AffectNet.py', help='path to APViT config')
parser.add_argument('--apvit_weights', type=str, default='./weights/apvit_7class_best.pth', help='path to APViT weights')
parser.add_argument('--image_size', type=int, default=112, help='input image size for models')

parser.add_argument('--mldg_weights', type=str, default='./weights/mLDG_no_apvit_pretrain.pth', help='path to mLDG weights')

parser.add_argument('--use_gl_modules', action='store_true', default=True, help='use mLDG configuration with local-global feature extraction modules from EfficientFace')
parser.add_argument('--no_gl_modules', dest='use_gl_modules', action='store_false', help='disable mLDG configuration with local-global feature extraction modules from EfficientFace')

parser.add_argument('--use_apvit', action='store_true', default=True, help='use APViT label distribution generator for training')
parser.add_argument('--no_apvit', dest='use_apvit', action='store_false', help='disable APViT label distribution generator and use hard labels for training')

parser.add_argument('--num_classes', type=int, default=7, choices=[7, 8], help='number of mLDG output classes')

parser.add_argument('--remap_classes', action='store_true', default=False, help='remap AffectNet classes to the ones output by APViT')
parser.add_argument('--no_remap_classes', dest='remap_classes', action='store_false', help='disable remap classes')

# WANDB related arguments
parser.add_argument('--wandb_project', type=str, default='FER-Thesis-mLDG-Training', help='wandb project name')
parser.add_argument('--wandb_entity', type=str, default='dval', help='wandb entity name')
parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
parser.add_argument('--combo_index', type=str, default=None, help='wandb experiment combination index')
parser.add_argument('--no_wandb', action='store_true', default=False, help='disable wandb logging')

args = parser.parse_args()

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    best_acc = 0

    print('Training time: ' + now.strftime("%m-%d %H:%M"))
    print(f'Using APViT label distribution generator: {args.use_apvit}')

    # Initialize wandb
    if not args.no_wandb:
        wandb_run_name = args.wandb_name if args.wandb_name else format_run_string(args)

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "adjust_freq": args.af,
                "factor": args.factor,
                "mldg_weights": args.mldg_weights,
                "use_apvit": args.use_apvit,
                "use_gl_modules": args.use_gl_modules,
                "num_classes": args.num_classes,
                "dataset": f"AffectNet-{args.num_classes}",
                "image_size": args.image_size
            }
        )
        # Log the model architecture as a graph
        wandb.run.log_code(".")

    # Create model
    ## mLDG (trainee model)
    model_cla = load_base_mLDG(
        checkpoint_path=args.mldg_weights, 
        uses_ef_modules=args.use_gl_modules,
        num_classes=args.num_classes)
    model_cla = torch.nn.DataParallel(model_cla).cuda()
    
    ## APViT (label distribution generator) - only initialize if using it
    model_dis = None
    
    if args.use_apvit:
        # Small hack to ensure APViT outputs torch tensors
        os.environ['MODEL_VIS'] = '1'

        model_dis = init_apvit_model()

    # Watch the model with wandb
    if not args.no_wandb:
        wandb.watch(model_cla, log="all", log_freq=args.print_freq)

    # define loss function (criterion) and optimizer
    criterion_val = nn.CrossEntropyLoss().cuda()
    criterion_train = cross_entropy if args.use_apvit else criterion_val

    optimizer = torch.optim.SGD(model_cla.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    recorder = RecorderMeter(args.epochs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            best_acc = best_acc.to()
            model_cla.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')

    if args.remap_classes:
        # Rename directories to match APViT's class mapping
        print("Renaming directories to match APViT's class mapping...")
        if not rename_directories(traindir):
            print("Failed to rename training directories. Exiting.")
            return
        if not rename_directories(valdir):
            print("Failed to rename validation directories. Exiting.")
            return

    # Define normalization values equivalent to APViT's preprocessing
    # These values are based on the ImageNet dataset normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 123.675/255, 116.28/255, 103.53/255
                                std=[0.229, 0.224, 0.225])    # 58.395/255, 57.12/255, 57.375/255

    train_dataset = datasets.ImageFolder(traindir,
                                         transforms.Compose([transforms.RandomResizedCrop((args.image_size, args.image_size)),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             normalize]))

    test_dataset = datasets.ImageFolder(valdir,
                                        transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                                            transforms.ToTensor(),
                                                            normalize]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    if args.evaluate:
        validate(val_loader, model_cla, criterion_val, args)
        return

    # Create log directory if it doesn't exist
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./checkpoint', exist_ok=True)

    # Log whether APViT is being used
    txt_name = './log/' + time_str + 'log.txt'
    with open(txt_name, 'a') as f:
        f.write(f'Using APViT label distribution generator: {args.use_apvit}\n')

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args)
        print('Current learning rate: ', current_learning_rate)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        # train for one epoch
        train_acc, train_los = train(train_loader, model_cla, model_dis, criterion_train, optimizer, epoch, args)

        # evaluate on validation set
        val_acc, val_los = validate(val_loader, model_cla, criterion_val, args, epoch)

        # Log metrics to wandb
        if not args.no_wandb:
            wandb.log(
                data={
                "epoch": epoch,
                "train/loss": train_los,
                "train/acc": train_acc,
                "val/loss": val_los,
                "val/acc": val_acc,
                "learning_rate": current_learning_rate})

        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        curve_name = time_str + 'log.png'

        # Plot and save the training/validation curves
        if args.no_wandb:
            recorder.plot_curve(os.path.join('./log/', curve_name))

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        print('Current best accuracy: ', best_acc.item())
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_cla.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'recorder': recorder
        }, is_best, args)
        
        # Log current checkpoints to wandb
        if not args.no_wandb:
            wandb.run.summary["best_accuracy"] = best_acc.item()
            
            print("Saving model(s) to wandb...")
            regular_checkpoint_abs_path = os.path.abspath(args.checkpoint_path)
            
            if epoch % 2 == 0:  # Save every 2 epochs to reduce overhead
                print("Saving regular checkpoint to wandb...")
                wandb.save(regular_checkpoint_abs_path, policy='now')
                print("Done!")

            if is_best:
                best_checkpoint_abs_path = os.path.abspath(args.best_checkpoint_path)
                print("Saving best checkpoint to wandb...")
                wandb.save(best_checkpoint_abs_path, policy='now')
                print("Done!")
            
        
        end_time = time.time()
        epoch_time = end_time - start_time
        print("An Epoch Time: ", epoch_time)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(str(epoch_time) + '\n')

    # Close wandb when training completes
    if not args.no_wandb:
        wandb.finish()


def train(train_loader, model_cla, model_dis, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))
    soft_max = nn.Softmax(dim=1)

    # switch mode
    model_cla.train()
    if model_dis is not None:
        model_dis.eval()  # APViT is always in eval mode if it's being used

    for i, (images, target) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda()

        # compute output from mLDG
        output = model_cla(images)
        
        # Different training modes based on whether APViT is used
        if args.use_apvit:
            output_prob = soft_max(output)
            
            # compute label distribution from APViT
            with torch.no_grad():
                # APViT already outputs softmaxed probabilities
                soft_label_prob = model_dis(images, return_loss=False)
            
            # compute loss using soft labels
            loss = criterion(output_prob, soft_label_prob)
        else:
            # standard cross-entropy with hard labels
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log batch metrics to wandb
        if not args.no_wandb and i % args.print_freq == 0:
            wandb.log({
                "train/batch_loss": losses.val,
                "train/batch_acc": top1.val,
                "train/step": epoch * len(train_loader) + i
                })

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args, epoch=0):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # For confusion matrix
    if not args.no_wandb:
        all_preds = []
        all_targets = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # Collect predictions for confusion matrix
            if not args.no_wandb:
                _, pred = output.topk(1, 1, True, True)
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())

            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        print(' *** Accuracy {top1.avg:.3f}  *** '.format(top1=top1))
        with open('./log/' + time_str + 'log.txt', 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')
        
        # Log confusion matrix to wandb
        if not args.no_wandb:
            wandb.log({
                "val/confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=np.array(all_targets),
                    preds=np.array(all_preds),
                    class_names=CLASS_NAMES[:args.num_classes]
                )
            })
            
    return top1.avg, losses.avg


def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint_path)
    print('Saved checkpoint to {}'.format(args.checkpoint_path))
    if is_best:
        shutil.copyfile(args.checkpoint_path, args.best_checkpoint_path)
        print('Saved best model to {}'.format(args.best_checkpoint_path))


def cross_entropy(predict_label, true_label):
    return torch.mean(- true_label * torch.log(predict_label + 1e-10))  # Added small epsilon to prevent log(0)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (args.factor ** (epoch // args.af))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)

def init_apvit_model():
    model_dis = None

    config = Config.fromfile(args.apvit_config)
    num_classes = args.num_classes

    # Propagate num_classes to the config
    config.num_classes = num_classes
    config.data.train.dataset.num_classes = num_classes
    config.data.val.num_classes = num_classes
    config.data.test.num_classes = num_classes
    config.model.head.num_classes = num_classes

    # Init the model with the modified configuration, move to GPU, and set to inference mode
    model_dis = init_model(
        config=config,
        checkpoint=args.apvit_weights
    )
    model_dis = model_dis.cuda()
    model_dis.eval()  # Set to evaluation mode as we don't train the distribution generator

    return model_dis


def format_run_string(args):
    combination_index = args.combo_index if args.combo_index else "default"

    base_name = os.path.splitext(os.path.basename(args.mldg_weights))[0]
    gl_str = "GL" if args.use_gl_modules else "NoGL"
    av_str = "AV" if args.use_apvit else "NoAV"
    nc_str = f"NC{args.num_classes}"

    return f"{combination_index}_{base_name}_{gl_str}_{av_str}_{nc_str}"


def rename_directories(dir_path):
    """
    Rename subdirectories (0-7) in the given directory according to the mapping:
    0→6, 1→4, 2→3, 3→5, 4→2, 5→1, 6→0, 7→7
    
    Args:
        dir_path (str): Path to the directory containing subdirectories 0-7
        
    Returns:
        bool: True if successful, False if there was an error
    """
    try:
        # Convert to Path object for easier handling
        target_dir = Path(dir_path)
        
        # Check if target directory exists
        if not target_dir.exists():
            print(f"Error: Directory '{dir_path}' does not exist.")
            return False
            
        if not target_dir.is_dir():
            print(f"Error: '{dir_path}' is not a directory.")
            return False
            
        print(f"Renaming directories in: {target_dir.absolute()}")
        
        # Define the mapping
        mapping = {
            '0': '6',
            '1': '4', 
            '2': '3',
            '3': '5',
            '4': '2',
            '5': '1',
            '6': '0',
            '7': '7'  # No change
        }
        
        # First pass: rename to temporary names to avoid conflicts
        temp_renames = []
        for old_name in mapping.keys():
            old_path = target_dir / old_name
            if old_path.exists() and old_path.is_dir():
                temp_name = f"temp_{old_name}"
                temp_path = target_dir / temp_name
                old_path.rename(temp_path)
                temp_renames.append((temp_name, old_name))
                print(f"Temporarily renamed {old_name} to {temp_name}")
        
        # Second pass: rename to final names according to mapping
        for temp_name, original_name in temp_renames:
            temp_path = target_dir / temp_name
            new_name = mapping[original_name]
            new_path = target_dir / new_name
            
            if temp_path.exists():
                temp_path.rename(new_path)
                if original_name != new_name:
                    print(f"Renamed {original_name} to {new_name}")
                else:
                    print(f"Directory {original_name} remains unchanged")
        
        print("Directory renaming completed successfully.")
        return True
        
    except Exception as e:
        print(f"Error during renaming: {e}")
        return False

if __name__ == '__main__':
    main()