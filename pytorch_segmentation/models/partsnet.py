import torch
from torch import nn
# NEEED TO WORK vvv OUT 
# from models.layers import Conv, Hourglass, Pool
# from extensions.AE.AE_loss import AEloss
# from task.loss import HeatmapLoss

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class PartsNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128,
        tag_dim=17, **kwargs):
        super(PartsNet, self).__init__()
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=bn),
            Conv(64, 128, bn=bn),
            Pool(2, 2),
            Conv(128, 128, bn=bn),
            Conv(128, inp_dim, bn=bn)
        )
        self.features = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
            Conv(inp_dim, inp_dim, 3, bn=False),
            Conv(inp_dim, inp_dim, 3, bn=False)
        ) for i in range(nstack)] )

        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )

        self.nstack = nstack
        self.myAEloss = AEloss() #TODO: Build AE Loss Layers
        #self.heatmapLoss = HeatmapLoss() #TODO: Switch out
        def InstanceLoss():
            pass
        self.instanceLoss = InstanceLoss()
        self.tag_dim = tag_dim

    def forward(self, imgs):
        # x = imgs.permute(0, 3, 1, 2) # handled in preprocessing
        x = imgs
        x = self.pre(x)
        preds = []
        for i in range(self.nstack):
            feature = self.features[i](x)
            preds.append( self.outs[i](feature) )
            if i != self.nstack - 1:
                x = x + self.merge_preds[i](preds[-1]) + self.merge_features[i](feature)
        return torch.stack(preds, 1)

    def calc_loss(self, preds, inst_masks = None, part_masks= None):
        import inspect
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)

        d = preds.shape[-1]

        dets = preds[:,:,:17] #Actual keypoints
        tags = preds[:,:,17:34] #Corresponding tags

        #keypoints = keypoints.cpu().long()
        print(preds.size())
        batchsize = preds.size()[0]
        return 0, 0, 0  

        #tag_loss = []
        #for i in range(self.nstack):
        #    tag = tags[:,i].contiguous().view(batchsize, -1, 1)
        #    tag_loss.append( self.myAEloss(tag, keypoints) )
        #tag_loss = torch.stack(tag_loss, dim = 1).cuda(tags.get_device())

        #detection_loss = []
        #for i in range(self.nstack):
        #    detection_loss.append( self.heatmapLoss(dets[:,i], heatmaps, masks) )
        #detection_loss = torch.stack(detection_loss, dim=1)
        #return tag_loss[:,:,0], tag_loss[:,:,1], detection_loss
