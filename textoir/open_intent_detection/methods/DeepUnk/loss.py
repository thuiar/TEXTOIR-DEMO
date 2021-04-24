from open_intent_detection.utils import * 

class AMSoftmaxORI(nn.Module):
    """Computes the AM-Softmax loss with cos or arc margin"""
    margin_types = ['cos', 'arc']

    def __init__(self,
                 in_feats,
                 n_classes=10,
                 margin_type='cos', 
                 gamma=0., m=0.35, s=30, t=1.):
        super(AMSoftmaxORI, self).__init__()
        assert margin_type in AMSoftmaxORI.margin_types
        self.margin_type = margin_type
        assert gamma >= 0
        self.gamma = gamma
        assert m > 0
        self.m = m
        assert s > 0
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        assert t >= 1
        self.t = t
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)

    def forward(self, cos_theta, target):
        if self.margin_type == 'cos':
            phi_theta = cos_theta - self.m
        else:
            sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            phi_theta = cos_theta * self.cos_m - sine * self.sin_m #cos(theta+m)
            phi_theta = torch.where(cos_theta > self.th, phi_theta, cos_theta - self.sin_m * self.m)

        index = torch.zeros_like(cos_theta, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        output = torch.where(index, phi_theta, cos_theta)

        if self.gamma == 0 and self.t == 1.:
            return F.cross_entropy(self.s*output, target)

        if self.t > 1:
            h_theta = self.t - 1 + self.t*cos_theta
            support_vecs_mask = (1 - index) * \
                torch.lt(torch.masked_select(phi_theta, index).view(-1, 1).repeat(1, h_theta.shape[1]) - cos_theta, 0)
            output = torch.where(support_vecs_mask, h_theta, output)
            return F.cross_entropy(self.s*output, target)

class AMSoftmax(nn.Module):

    def __init__(self,
                 in_feats,
                 n_classes=10,
                 m=0.35,
                 s=30):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-9)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-9)
        w_norm = torch.div(self.W, w_norm)
        w_norm = w_norm.cuda()
        costh = torch.mm(x_norm, w_norm)
        phi_theta = costh - self.m
        index = torch.zeros_like(costh, dtype=torch.uint8)
        index.scatter_(1, lb.data.view(-1, 1), 1)
        output = torch.where(index, phi_theta, costh)

        return F.cross_entropy(self.s*output, lb)

        # delt_costh = torch.zeros_like(costh).scatter_(1, lb.unsqueeze(1), self.m)
        # costh_m = costh - delt_costh
        # costh_m_s = self.s * costh_m
        # loss = self.ce(costh_m_s, lb)
            
        # return loss

class AMSoftmaxLoss(nn.Module):
    """Computes the AM-Softmax loss with cos or arc margin"""
    margin_types = ['cos', 'arc']

    def __init__(self,
                 in_feats,
                 n_classes=10,
                 margin_type='cos', 
                 gamma=0., m=0.35, s=30, t=1.):
        super(AMSoftmaxLoss, self).__init__()
        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        assert gamma >= 0
        self.gamma = gamma
        assert m > 0
        self.m = m
        assert s > 0
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        assert t >= 1
        self.t = t
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)

    def forward(self, cos_theta, target):
        if self.margin_type == 'cos':
            phi_theta = cos_theta - self.m
        else:
            sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            phi_theta = cos_theta * self.cos_m - sine * self.sin_m #cos(theta+m)
            phi_theta = torch.where(cos_theta > self.th, phi_theta, cos_theta - self.sin_m * self.m)

        index = torch.zeros_like(cos_theta, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        output = torch.where(index, phi_theta, cos_theta)

        if self.gamma == 0 and self.t == 1.:
            return F.cross_entropy(self.s*output, target)

        if self.t > 1:
            h_theta = self.t - 1 + self.t*cos_theta
            support_vecs_mask = (1 - index) * \
                torch.lt(torch.masked_select(phi_theta, index).view(-1, 1).repeat(1, h_theta.shape[1]) - cos_theta, 0)
            output = torch.where(support_vecs_mask, h_theta, output)
            return F.cross_entropy(self.s*output, target)

        return focal_loss(F.cross_entropy(self.s*output, target, reduction='none'), self.gamma)
# class AMSoftmax(torch.nn.Module):
#     r"""PyTorch implementation of
#         2. Large Margin Cosine Loss / CosFase
#     Args:
#         in_features: Size of model discriptor
#         out_features: Number of classes
#         s: Input features norm
#         m: Margin value for CosFase
#         criterion: One of {'cross_entropy', 'focal', 'reduced_focal'}
#     Reference:
#         1. CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
#             https://arxiv.org/pdf/1801.07698.pdf
#     Code:
#         github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch
#         github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/master/src/modeling/metric_learning.py
#     """

#     def __init__(self, in_features, out_features, s=30.0, m=0.35, criterion="cross_entropy"):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
#         self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
#         torch.nn.init.xavier_uniform_(self.weight)

#         self.criterion = AMSoftmax

#     def forward(self, features, y_true):
#         """
#         Args:
#             features: L2 normalized logits from the model
#             y_true: Class labels, not one-hot encoded
#         """
#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         cosine = torch.nn.functional.linear(features, torch.nn.functional.normalize(self.weight))
#         phi = cosine - self.m
#         # --------------------------- convert label to one-hot ---------------------------
#         one_hot = torch.zeros(cosine.size()).to(features)
#         one_hot.scatter_(1, y_true.view(-1, 1).long(), 1)
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
#         output *= self.s

#         loss = self.criterion(output, y_true)

#         return loss