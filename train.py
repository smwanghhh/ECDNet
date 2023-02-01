import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from torchvision.utils import save_image
import os, torch
import argparse
import Networks
from dataset import RafDataSet, AffectNet, CAER
import torch.nn.functional as F
import math
import cv2
import math
import random
import image_utils as util

global epoch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='rafdb', help='data.')
    parser.add_argument('-c', '--checkpoint', type=str, default= None,  help='load disentangle model')#'./2/acc_models.pth'
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Batch size for validation.')
    parser.add_argument('--img_size', type=int, default=112, help='Batch size.')
    parser.add_argument('--dz', type=int, default=128, help='latent code dimension.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--start_epochs', type=int, default=0, help='Total training epochs.')
    parser.add_argument('--end_epochs', type=int, default=500, help='Total training epochs.')
    parser.add_argument('--w1', type=float, default = 100, help='recognition weight')
    parser.add_argument('--w2', type=float, default = 10, help='classification weight')
    parser.add_argument('--w3', type=float, default = 1, help='regularization weight')
    return parser.parse_args()

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])



def run_training():
    args = parse_args()
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25))])

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if args.data == 'rafdb':
        train_dataset = RafDataSet('../datasets/raf-basic/', phase='train', transform=data_transforms, basic_aug=True)
        val_dataset = RafDataSet('../datasets/raf-basic/', phase='test', transform=data_transforms_val)
        class_num = 7
        class_name = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
        data_w = train_dataset.weight()
        data_dis = torch.cuda.FloatTensor(data_w)

    elif args.data == 'oulu':
        train_dataset = Oulu('../datasets/Oulu_Face/', phase='train', transform=data_transforms, basic_aug=True)
        val_dataset = Oulu('../datasets/Oulu_Face/', phase='test', transform=data_transforms_val)
        class_num = 7
        class_name = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
        data_w = train_dataset.weight()
        data_dis = torch.cuda.FloatTensor(data_w)
        
    elif args.data == 'affectnet':
        train_dataset = AffectNet('../datasets/AffectNet-8_Face/', phase='train', transform=data_transforms, basic_aug=True)
        val_dataset = AffectNet('../datasets/AffectNet-8_Face/', phase='test', transform=data_transforms_val)
        class_num = 7
        class_name = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
        data_w = train_dataset.weight()
        data_dis = torch.cuda.FloatTensor(data_w)

    print('training datasets:', args.data)
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)
    
    if args.w3 != 0:
      model_path = './rec+cls+regu_model'
    elif args.w2 != 0:
      model_path = './rec+cls_model'
    else:
      model_path = './rec_model'
    print(model_path)
    img_path = model_path.replace('model', 'image')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    val_num = val_dataset.__len__()
    print('Validation set size:', val_num)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=args.val_batch_size,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True)

    encoder = Networks.Encoder(img_size = args.img_size, z_app = args.dz, z_geo = args.dz, num_class = class_num)
    decoder = Networks.Decoder(args.dz, args.img_size)
    #fer_model = Networks.FER_model(img_size = args.img_size, num_class = class_num)
    encoder = encoder.cuda()
    encoder_param = encoder.get_parameters()
    decoder = decoder.cuda()
    decoder_param = decoder.get_parameters()
    percep_model = Networks.Percep_model()
    percep_model = percep_model.cuda()
    #fer_param = fer_model.get_parameters()
    #fer_model = fer_model.cuda()

    optimizer = torch.optim.Adam(decoder_param + encoder_param, args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.98)
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_rec = torch.nn.SmoothL1Loss()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])

    best_acc,  best_rec = 0., 100.
    
    neutral = torch.zeros(size = (3, args.img_size, args.img_size)).cuda()
    count = 0
    for _, (imgs, targets, indexes) in enumerate(train_loader):
        imgs = imgs.cuda()
        targets = targets.cuda()
        neutral_index = torch.where(targets == 6)[0]
        if len(neutral_index) > 0:
            neutral_imgs = torch.index_select(imgs, 0, neutral_index)
            neutral += torch.mean(neutral_imgs, 0)
            count += 1
    neutral_avg = neutral / count
    save_image(neutral_avg, 'average_face.png', nrow = 1)

    for i in range(args.start_epochs, args.end_epochs):
        train_clloss, train_regu, train_recloss = 0.0, 0.0, 0.0
        correct_sum= torch.tensor(0).cuda()
        iter_cnt = 0
        encoder.train()
        decoder.train()
        percep_model.train()
        for batch_i, (imgs, targets, indexes) in enumerate(train_loader):
            iter_cnt += 1
            imgs = imgs.cuda()
            targets = targets.cuda()
            
            gt_out = percep_model(imgs)
            # #training mixup data
            optimizer.zero_grad()
            z_app, z_geo, logit = encoder(imgs)
            app, geo = decoder(z_app, z_geo)
            gx, coord = util.warpnn(app, geo* 5, len(targets))
            gen_out = percep_model(gx)
            loss_rec = torch.tensor(0.).cuda()
            for index in range(len(gen_out)):
                loss_rec += criterion_rec(gt_out[index], gen_out[index])
            #loss_rec += criterion_rec(imgs, gx)
            loss_cls = criterion_cls(logit, targets)
            neutral = torch.unsqueeze(neutral_avg, 0).repeat(len(targets), 1, 1, 1)
            app_out = percep_model(app)
            neu_out = percep_model(neutral)
            loss_regu = torch.tensor(0.).cuda()
            for index in zip(neu_out, app_out):
                loss_regu += criterion_rec(index[0], index[1])
            smooth_loss = torch.mean(torch.abs(geo[:, 0, :-1, :] - geo[:, 0, 1:, :])+ torch.abs(geo[:, 1, :-1, :] - geo[:, 1, 1:, :]))
            smooth_loss += torch.mean(torch.abs(geo[:, 0, :, :-1] - geo[:, 0, :, 1:]) + torch.abs(geo[:, 1, :, :-1] - geo[:, 1, :, 1:]))
            loss = loss_rec * args.w1 + loss_cls * args.w2 + loss_regu * args.w3 + smooth_loss
            loss.backward()
            optimizer.step()
            _, predicts = torch.max(logit, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
            train_clloss += loss_cls
            train_recloss += loss_rec
        print(smooth_loss)
        train_acc = correct_sum.float() / float(train_dataset.__len__())
        train_clloss = train_clloss / iter_cnt
        train_recloss = train_recloss / iter_cnt
        print('[Epoch %d] Training classification acc: %.4f.   RecLoss: %.3f  ClsLoss: %.3f     LR: %.6f' %
              (i, train_acc, train_recloss,  train_clloss, optimizer.param_groups[0]["lr"]))
        scheduler.step()
        with torch.no_grad():
            val_recloss, val_clloss= 0.0, 0.0
            iter_cnt = 0
            bingo_cnt, bingo_cnt_ =  0, 0
            encoder.eval()
            decoder.eval()
            percep_model.eval()
            for batch_i, (imgs, targets, _) in enumerate(val_loader):
                imgs = imgs.cuda()
                targets = targets.cuda()
                z_app, z_geo, logit = encoder(imgs.cuda())
                apps, geo = decoder(z_app, z_geo)
                gx, coord = util.warpnn(apps, geo * 5, len(targets))
                CE_loss = criterion_cls(logit, targets)
                val_clloss += CE_loss
                Rec_loss = torch.tensor(0.).cuda()
                gt_out = percep_model(imgs)
                gen_out = percep_model(gx)
                for index in zip(gt_out, gen_out):
                    Rec_loss += criterion_rec(index[0], index[1])
                val_recloss += Rec_loss
                iter_cnt += 1
                _, predicts = torch.max(logit, 1)
                correct_or_not = torch.eq(predicts, targets.cuda())
                bingo_cnt += correct_or_not.sum().cpu()


            val_clloss = val_clloss / iter_cnt
            val_recloss = val_recloss / iter_cnt
            val_acc = bingo_cnt.float()/float(val_num)
            val_acc = np.around(val_acc.numpy(), 4)
            print('Test classification accuracy: %.4f.   RecLoss: %.3f   ClsLoss1: %.3f' %
                  (val_acc, val_recloss,  val_clloss))


            #save_image(imgs, os.path.join(img_path, 'orig.png'), nrow = int(math.sqrt(args.batch_size)), normalize = True)
            #save_image(gx, os.path.join(img_path, 'gen.png'), nrow=int(math.sqrt(args.batch_size)), normalize = True)
            #save_image(apps, os.path.join(img_path, 'app.png'), nrow=int(math.sqrt(args.batch_size)), normalize = True)
            if val_acc > best_acc:
                best_acc = val_acc
                print("best_acc %s" % (str(best_acc)))
                save = {'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict()}

                torch.save(save, os.path.join(model_path, 'acc_models' + ".pth"))
                print('Model saved.')
            if val_recloss < best_rec:
                best_rec = val_recloss
                print("best_rec %s" % (str(best_rec.item())))
                save = {'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict()}
                torch.save(save, os.path.join(model_path, 'rec_models' + ".pth"))

def test():
    args = parse_args()
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_dataset = RafDataSet('../datasets/raf-basic/', phase='test', transform=data_transforms_val)
    class_num = 7
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=args.val_batch_size,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True)
    encoder = Networks.Encoder(img_size = args.img_size, z_app = args.dz, z_geo = args.dz, num_class = class_num)
    decoder = Networks.Decoder(args.dz, args.img_size)
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        
        
    with torch.no_grad():
        for batch_i, (imgs, targets, _) in enumerate(val_loader):
            imgs = imgs.cuda()
            targets = targets.cuda()
            z_app, z_geo, _ = encoder(imgs)
            app, geo = decoder(z_app, z_geo)
            gx, coord = util.warpnn(app, geo*20, len(targets))
           # util.plot_grid(coord, app.cpu(), gx.cpu(), imgs.cpu(), targets.cpu())
        save_image(imgs, os.path.join('./images', 'orig.png'), nrow = int(math.sqrt(args.batch_size)), normalize = True)
        save_image(gx, os.path.join('./images', 'gen.png'), nrow=int(math.sqrt(args.batch_size)), normalize = True)
        save_image(app, os.path.join('./images', 'app.png'), nrow=int(math.sqrt(args.batch_size)), normalize = True)
    

            
if __name__ == "__main__":     
    class RecorderMeter():
      pass               
    run_training()
    #test()
