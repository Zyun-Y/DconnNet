from random import shuffle
import numpy as np
from connect_loss import connect_loss,Bilateral_voting
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from lr_update import get_lr
from metrics.cldice import clDice
import os
from apex import amp
import sklearn
import torchvision.utils as utils
from sklearn.metrics import precision_score
from skimage.io import imread, imsave



class Solver(object):
    def __init__(self, args,optim=torch.optim.Adam):
        self.args = args
        self.optim = optim
        self.NumClass = self.args.num_class
        self.lr = self.args.lr
        H, W = args.resize


        self.hori_translation = torch.zeros([1,self.NumClass,W,W])
        for i in range(W-1):
            self.hori_translation[:,:,i,i+1] = torch.tensor(1.0)
        self.verti_translation = torch.zeros([1,self.NumClass,H,H])
        for j in range(H-1):
            self.verti_translation[:,:,j,j+1] = torch.tensor(1.0)
        self.hori_translation = self.hori_translation.float()
        self.verti_translation = self.verti_translation.float()

    def create_exp_directory(self,exp_id):
        if not os.path.exists('models/' + str(exp_id)):
            os.makedirs('models/' + str(exp_id))

        csv = 'results_'+str(exp_id)+'.csv'
        with open(os.path.join(self.args.save, csv), 'w') as f:
            f.write('epoch, dice, Jac, clDice \n')



    def get_density(self, pos_cnt,bins = 50):
        ### only used for Retouch in this code
        val_in_bin_ = [[],[],[]]
        density_ = [[],[],[]]
        bin_wide_ = []

        ### check
        for n in range(3):
            density = []
            val_in_bin = []
            c1 = [i for i in pos_cnt[n] if i != 0]
            c1_t = torch.tensor(c1)
            bin_wide = (c1_t.max()+50)/bins
            bin_wide_.append(bin_wide)

            edges = torch.arange(bins + 1).float()*bin_wide
            for i in range(bins):
                val = [c1[j] for j in range(len(c1)) if ((c1[j] >= edges[i]) & (c1[j] < edges[i + 1]))]
                # print(val)
                val_in_bin.append(val)
                inds = (c1_t >= edges[i]) & (c1_t < edges[i + 1]) #& valid
                num_in_bin = inds.sum().item()
                # print(num_in_bin)
                density.append(num_in_bin)

            denominator = torch.tensor(density).sum()
            # print(val_in_bin)

            #### get density ####
            density = torch.tensor(density)/denominator
            density_[n]=density
            val_in_bin_[n] = val_in_bin
        print(density_)

        return density_, val_in_bin_,bin_wide_



    def train(self, model, train_loader, val_loader,exp_id, num_epochs=10):

        #### lr update schedule
        # gamma = 0.5
        # step_size = 10
        optim = self.optim(model.parameters(), lr=self.lr)
        # scheduler = lr_scheduler.MultiStepLR(optim, milestones=[12,24,35],
        #                                 gamma=gamma)  # decay LR by a factor of 0.5 every 5 epochs
        ####

        print('START TRAIN.')

        
        self.create_exp_directory(exp_id)

        if self.args.use_SDL:
            assert 'retouch' in self.args.dataset, 'Please input the calculated distribution data of your own dataset, if you are now using Retouch'
            device_name = self.args.dataset.split('retouch-')[1]
            pos_cnt = np.load(self.args.weights+device_name+'/training_positive_pixel_'+str(exp_id)+'.npy',allow_pickle=True)
            density, val_in_bin,bin_wide = self.get_density(pos_cnt)
            self.loss_func=connect_loss(self.args,self.hori_translation,self.verti_translation, density,bin_wide)
        else:
            self.loss_func=connect_loss(self.args,self.hori_translation,self.verti_translation)

        net, optimizer = amp.initialize(model, optim, opt_level='O2')


        best_p = 0
        best_epo = 0
        scheduled = ['CosineAnnealingWarmRestarts']
        if self.args.lr_update in scheduled:
            scheduled = True
            if self.args.lr_update == 'CosineAnnealingWarmRestarts':
                scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min = 0.00001)
        else:
            scheduled = False

        if self.args.test_only:
            self.test_epoch(net,val_loader,0,exp_id)
        else:
            for epoch in range(self.args.epochs):
                net.train()

                if scheduled:
                    scheduler.step()
                else:
                    curr_lr = get_lr(self.lr,self.args.lr_update, epoch, num_epochs, gamma=self.args.gamma,step=self.args.lr_step)
                    for param_group in optim.param_groups:
                        param_group['lr'] = curr_lr
                

                for i_batch, sample_batched in enumerate(train_loader):
                    X = Variable(sample_batched[0])
                    y = Variable(sample_batched[1])

                    X= X.cuda()
                    y = y.float().cuda()
                    # print(X.shape,y.shape)

                    optim.zero_grad()
                    output, aux_out = net(X)

                    loss_main = self.loss_func(output, y)
                    loss_aux = self.loss_func(aux_out, y)
                    loss =loss_main+0.3*loss_aux
                    with amp.scale_loss(loss, optimizer) as scale_loss:
                        scale_loss.backward()

                    optim.step()
                    
                    print('[epoch:'+str(epoch)+'][Iteration : ' + str(i_batch) + '/' + str(len(train_loader)) + '] Total:%.3f' %(
                        loss.item()))



                dice_p = self.test_epoch(net,val_loader,epoch,exp_id)
                if best_p<dice_p:
                    best_p = dice_p
                    best_epo = epoch
                    torch.save(model.state_dict(), 'models/' + str(exp_id) + '/best_model.pth')
                if (epoch+1) % self.args.save_per_epochs == 0:
                    torch.save(model.state_dict(), 'models/' + str(exp_id) + '/'+str(epoch+1)+'_model.pth')
                print('[Epoch :%d] total loss:%.3f ' %(epoch,loss.item()))

                # if epoch%self.args.save_per_epochs==0:
                #     torch.save(model.state_dict(), 'models/' + str(exp_id) + '/epoch' + str(epoch + 1)+'.pth')
            csv = 'results_'+str(exp_id)+'.csv'
            with open(os.path.join(self.args.save, csv), 'a') as f:
                f.write('%03d,%0.6f \n' % (
                    best_epo,
                    best_p
                ))
            # writer.close()
            print('FINISH.')
            
    def test_epoch(self,model,loader,epoch,exp_id):
        model.eval()
        self.dice_ls = []
        self.Jac_ls=[]
        self.cldc_ls = []
        with torch.no_grad(): 
            for j_batch, test_data in enumerate(loader):
                curr_dice = []
                X_test = Variable(test_data[0])
                y_test = Variable(test_data[1])
                # name = test_data[2]

                X_test= X_test.cuda()
                y_test = y_test.long().cuda()

                output_test,_ = model(X_test)
                batch,channel,H,W = X_test.shape


                hori_translation = self.hori_translation.repeat(batch,1,1,1).cuda()
                verti_translation = self.verti_translation.repeat(batch,1,1,1).cuda()

                

                if self.args.num_class == 1:  
                    output_test = F.sigmoid(output_test)
                    class_pred = output_test.view([batch,-1,8,H,W]) #(B, C, 8, H, W)
                    pred = torch.where(class_pred>0.5,1,0)
                    pred,_ = Bilateral_voting(pred.float(),hori_translation,verti_translation)

                else:
                    class_pred = output_test.view([batch,-1,8,H,W]) #(B, C, 8, H, W)
                    final_pred,_ = Bilateral_voting(class_pred,hori_translation,verti_translation)
                    pred = get_mask(final_pred)
                    pred = self.one_hot(pred, X_test.shape)
                

                dice,Jac = self.per_class_dice(pred,y_test)
                
                if self.args.num_class == 1:
                    pred_np = pred.squeeze().cpu().numpy()
                    target_np = y_test.squeeze().cpu().numpy()
                    cldc = clDice(pred_np,target_np)
                    self.cldc_ls.append(cldc)

                ###### notice: for multi-class segmentation, the self.dice_ls calculated following exclude the background (BG) class

                if self.args.num_class>1:
                    self.dice_ls += torch.mean(dice[:,1:],1).tolist() ## use self.dice_ls += torch.mean(dice,1).tolist() if you want to include BG
                    self.Jac_ls += torch.mean(Jac[:,1:],1).tolist() ## same as above
                else:
                    self.dice_ls += dice[:,0].tolist()
                    self.Jac_ls += Jac[:,0].tolist()

                if j_batch%(max(1,int(len(loader)/5)))==0:
                    print('[Iteration : ' + str(j_batch) + '/' + str(len(loader)) + '] Total DSC:%.3f ' %(
                        np.mean(self.dice_ls)))

            # print(len(self.Jac_ls))
            Jac_ls =np.array(self.Jac_ls)
            dice_ls = np.array(self.dice_ls)
            total_dice = np.mean(dice_ls)
            csv = 'results_'+str(exp_id)+'.csv'
            with open(os.path.join(self.args.save, csv), 'a') as f:
                f.write('%03d,%0.6f,%0.6f,%0.6f \n' % (
                    (epoch + 1),
                    total_dice,
                    np.mean(Jac_ls),
                    np.mean(self.cldc_ls)
                ))

            return np.mean(self.dice_ls)


    def per_class_dice(self,y_pred, y_true):
        eps = 0.0001
        y_pred = y_pred
        y_true = y_true

        FN = torch.sum((1-y_pred)*y_true,dim=(2,3)) 
        FP = torch.sum((1-y_true)*y_pred,dim=(2,3)) 
        Pred = y_pred
        GT = y_true
        inter = torch.sum(GT* Pred,dim=(2,3)) 


        union = torch.sum(GT,dim=(2,3)) + torch.sum(Pred,dim=(2,3)) 
        dice = (2*inter+eps)/(union+eps)
        Jac = (inter+eps)/(inter+FP+FN+eps)

        return dice, Jac

    def one_hot(self,target,shape):

        one_hot_mat = torch.zeros([shape[0],self.args.num_class,shape[2],shape[3]]).cuda()
        target = target.cuda()
        one_hot_mat.scatter_(1, target, 1)
        return one_hot_mat

def get_mask(output):
    output = F.softmax(output,dim=1)
    _,pred = output.topk(1, dim=1)
    #pred = pred.squeeze()
    
    return pred





