from model import *
import torch
import torch.nn.functional as F 

def test_Model():
    x = torch.zeros((3,15, 3072))  # N * L * F
    N=x.size(0)
    L=x.size(1)
    num_proposals=5
    num_classes=20
    pos_weight=0.7
    class_weight=torch.randn(num_classes)
    model = SST_AD_PreRel(num_proposals, num_classes)
    output, proposals_out, class_out, final_joint_cls_scores = model(x)
    y=np.ones((N,L))
    prop_loss=proploss(y,proposals_out,pos_weight)
    class_loss=classloss(y,class_out,class_weight,N,L )   
    print(class_loss.size()) 

def proploss(y,proposals_out,pos_weight):
    actionSum=y.copy()
    actionY=np.zeros((N,L,num_proposals))
    actionY[:,:,0]=actionSum>0
    for l in range(1,num_proposals):
        actionSum[:,l:]=actionSum[:,l:]+y[:,:-l]
        actionY[:,l:,l]=actionSum[:,l:]>((l+1)/2)
    
    actionY = torch.from_numpy(actionY).float()
    
    prop_sum=pos_weight*actionY*torch.log(proposals_out)+(1-pos_weight)*(1-actionY)*torch.log(1-proposals_out)
    #print(prop_sum.size())
    prop_loss=torch.sum(prop_sum)
    return prop_loss

def classloss(y,class_out,class_weight):   
    y=torch.from_numpy(y)
    N=y.size(0)
    L=y.size(1)
    y=y.view(N*L).long()
    closs=nn.CrossEntropyLoss(class_weight)
    y.size()
    class_loss=closs(class_out ,y )
    return class_loss
      
def train():
    #model = model.to(device=device)  # move the model parameters to CPU/GPU
    num_proposals=5
    num_classes=20
    
    pos_weight=0.7
    class_weight=torch.randn(num_classes)
    
    model = SST_AD_PreRel(num_proposals, num_classes)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            output, proposals_out, class_out, final_joint_cls_scores = model(x)
            prop_loss=proploss(y,proposals_out,pos_weight)
            class_loss=classloss(y,class_out,class_weight)  
            loss=prop_loss+class_loss
            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()
                
if __name__ == '__main__':
    #test_Model()
    learning_rate = 1e-2
    #x = torch.zeros((3,15, 3072))  # N * L * F
    #N=x.size(0)
    #L=x.size(1)
    train_part34()               