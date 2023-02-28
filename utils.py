import torch
import numpy as np
import random
import time
import scipy
import torch.nn as nn
from torch.autograd import grad


class MMDLoss(nn.Module):
    def __init__(self, args, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.device = args.device
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type
        self.eps = 0.05
        self.max_iter = 10

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def cost_matrix(self, x, y, p=2):
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum((x_col - y_lin) ** p, dim=2)
        return c

    def M(self, C, u, v):
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / self.eps

    def sinkhorn(self, x, y):
        C = self.cost_matrix(x, y)
        x_norm = (x ** 2).sum(dim=1, keepdims=True) ** 0.5
        y_norm = (y ** 2).sum(dim=1, keepdims=True) ** 0.5
        mu = (x_norm[:, 0] / x_norm.sum()).detach().to(self.device)
        nu = (y_norm[:, 0] / y_norm.sum()).detach().to(self.device)
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        actual_nits = 0
        thresh = 0.1

        for i in range(self.max_iter):
            u1 = u
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).t(), dim=-1)) + v
            err = (u - u1).abs().sum()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V))
        cost = torch.sum(pi * C)
        return cost, pi, C

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
            return loss
        elif self.kernel_type == 'sinkhorn':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            cost, pi, C = self.sinkhorn(source, target)
            pi = pi.detach()
            # _, pis, _ = self.sinkhorn(source, source)
            # _, pit, _ = self.sinkhorn(target, target)
            # pi = torch.ones(source.shape[0], target.shape[0]).to(self.device) / (source.shape[0] * target.shape[0])
            loss = torch.mean(XX) + torch.mean(YY) - torch.sum(pi*XY) - torch.sum(pi.t()*YX)
            return loss


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.alpha
        return grad_output, None


class WassersteinLoss(nn.Module):
    def __init__(self, args):
        super(WassersteinLoss, self).__init__()
        self.device = args.device
        self.discriminator = nn.Sequential(nn.Linear(args.dim, args.dim), nn.LeakyReLU(),
                                           nn.Linear(args.dim, 2), nn.LogSoftmax(1))
        self.eps = 0.05
        self.max_iter = 10

    def gradient_penalty(self, h_s, h_t):
        idx = np.arange(h_s.shape[0])
        np.random.shuffle(idx)
        h_s = h_s[idx]
        h_s = h_s[:h_t.size(0)]
        differences = h_t - h_s
        alpha = torch.rand(h_t.size(0), 1).to(self.device)
        interpolates = h_s + (alpha * differences)
        interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
        preds = self.discriminator(interpolates)
        gradients = grad(preds, interpolates,
                         grad_outputs=torch.ones_like(preds),
                         retain_graph=True, create_graph=True)[0]
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty

    def sort_rows(self, M, ins):
        MT = M.t()
        sorted_MT, _ = torch.topk(MT, ins)
        return sorted_MT.t()

    def random_distance(self, source, target):
        torch.autograd.set_detect_anomaly(True)
        proj = torch.randn(source.shape[1], source.shape[1]).detach().to(self.device)
        p1 = torch.matmul(source, proj)
        p2 = torch.matmul(target, proj)
        ppx = torch.mean(p1, dim=0)
        ppy = torch.mean(p2, dim=0)
        wd = torch.sum((ppx - ppy)**2)**0.5
        return wd

    def forward(self, source, target, alpha):
        source = GradReverse.apply(source, alpha)
        target = GradReverse.apply(target, alpha)
        # gp = self.gradient_penalty(source, target)
        sd = self.discriminator(source)
        td = self.discriminator(target)
        # wd = sd.mean() - td.mean()
        # return wd, gp
        return sd, td

    def cost_matrix(self, x, y, p=2):
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum((x_col - y_lin)**p, dim=2)
#        x_norm = (x ** 2).sum(dim=1, keepdims=True) ** 0.5
#        y_norm = (y ** 2).sum(dim=1, keepdims=True) ** 0.5
#        c = 1 - torch.matmul(x / x_norm, (y / y_norm).t())
        return c

    def sinkhorn(self, x, y):
        C = self.cost_matrix(x, y)
        # mu = (torch.ones(x.shape[0]) / x.shape[0]).detach().to(self.device)
        # nu = (torch.ones(y.shape[0]) / y.shape[0]).detach().to(self.device)
        x_norm = (x ** 2).sum(dim=1, keepdims=True) ** 0.5
        y_norm = (y ** 2).sum(dim=1, keepdims=True) ** 0.5
        mu = (x_norm[:, 0] / x_norm.sum()).detach().to(self.device)
        nu = (y_norm[:, 0] / y_norm.sum()).detach().to(self.device)
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        actual_nits = 0
        thresh = 0.1

        for i in range(self.max_iter):
            u1 = u
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).t(), dim=-1)) + v
            err = (u - u1).abs().sum()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V))
        cost = torch.sum(pi * C)
        return cost, pi, C

    def M(self, C, u, v):
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / self.eps

    def sinkhorn_loss(self, x, y):
        """
        Solve the entropic regularization balanced optimal transport problem

        Parameters:
        param: a(tensor (I, )) sample weights for source measure
        param: b(tensor (J, )) sample weights for target measure
        param: M(tensor (I, J)) distance matrix between source and target measure
        param: reg(float64) regularization factor > 0
        param: numItermax(int) max number of iterations
        param: stopThr(float64) stop threshol
        param: verbose(bool) print information along iterations

        Return:
        P(tensor (I, J)) the final transport plan
        loss(float) the wasserstein distance between source and target measure
        """
        M = self.cost_matrix(x, y)
        a, b = torch.ones(x.shape[0]).detach().to(self.device), torch.ones(y.shape[0]).detach().to(self.device)
        u = (torch.ones(a.shape[0], 1) / a.shape[0]).detach().to(self.device)
        v = (torch.ones(b.shape[0], 1) / b.shape[0]).detach().to(self.device)
        K = torch.exp(-M / self.eps)

        Kp = (1 / a).reshape(-1, 1) * K

        cpt, err = 0, 1
        while (err > 1e-3 and cpt < self.max_iter):
            uprev, vprev = u, v
            KtranposeU = torch.mm(K.t(), u)
            v = b.reshape(-1, 1) / KtranposeU
            u = 1. / Kp.mm(v)

            if (torch.any(KtranposeU == 0)
                    or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                    or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))):
                u, v = uprev, vprev
                break

            if cpt % 10 == 0:
                tmp2 = torch.einsum('ia,ij,jb->j', u, K, v)
                err = torch.norm(tmp2 - b)
            cpt += 1
        P = u.reshape(-1, 1) * K * v.reshape(1, -1)
        return torch.sum(P * M), P, M

    def ot_1d(self, source, target):
        res = torch.zeros_like(source).float().to(self.device)
        source_n = source.shape[0]
        target_n = target.shape[0]

        source_value = source
        _, source = torch.sort(source)
        _, order = torch.sort(target)

        cur_value = 0
        wdistance = 0
        cur_capacity = 1 / target_n
        for p in source:
            stock = 1 / source_n
            value = 0
            while stock > 0:
                if stock >= cur_capacity:
                    stock -= cur_capacity
                    value += target[order[cur_value]] * cur_capacity * source_n
                    wdistance += ((target[order[cur_value]] - source_value[p])**2) * cur_capacity * source_n
                    cur_value = min([cur_value + 1, target_n - 1])
                    cur_capacity = 1 / target_n
                else:
                    cur_capacity -= stock
                    value += target[order[cur_value]] * stock * source_n
                    wdistance += ((target[order[cur_value]] - source_value[p])**2) * stock * source_n
                    stock = 0
            res[p] = value
        return res, wdistance


class OT(nn.Module):
    def __init__(self, args):
        super(OT, self).__init__()
        self.device = args.device
        self.eps = 0.05
        self.max_iter = 10

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def cost_matrix(self, x, y, p=2):
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum((x_col - y_lin) ** p, dim=2)
        return c

    def M(self, C, u, v):
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / self.eps

    def sinkhorn(self, x, y):
        C = self.cost_matrix(x, y)
        x_norm = (x ** 2).sum(dim=1, keepdims=True) ** 0.5
        y_norm = (y ** 2).sum(dim=1, keepdims=True) ** 0.5
        mu = (x_norm[:, 0] / x_norm.sum()).detach().to(self.device)
        nu = (y_norm[:, 0] / y_norm.sum()).detach().to(self.device)
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        actual_nits = 0
        thresh = 0.1

        for i in range(self.max_iter):
            u1 = u
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).t(), dim=-1)) + v
            err = (u - u1).abs().sum()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V))
        cost = torch.sum(pi * C)
        return cost, pi, C

    def forward(self, source, target):
        # cost, pi, C = self.sinkhorn(source, target)
        # pi = pi.detach()
        #
        # kernels = self.guassian_kernel(source, target)
        # loss = torch.mean(kernels[:source.shape[0], :source.shape[0]]) \
        #        + torch.mean(kernels[source.shape[0]:, source.shape[0]:])\
        #        - 2 * torch.mean(kernels[:source.shape[0], source.shape[0]:])

        # loss = torch.sum((source.mean(0) - target.mean(0))**2)

        # kernels = self.guassian_kernel(source, target)
        # # C = kernels[:source.shape[0], :source.shape[0]]
        # loss = torch.mean(kernels[:source.shape[0], :source.shape[0]]) \
        #        + torch.mean(kernels[source.shape[0]:, source.shape[0]:])\
        #        - 2 * torch.mean(kernels[:source.shape[0], source.shape[0]:]) #torch.sum(pi*C)

        # kernels = self.guassian_kernel(source, target)
        # loss = torch.mean(kernels[:source.shape[0], :source.shape[0]]) \
        #        + torch.mean(kernels[source.shape[0]:, source.shape[0]:])\
        #        - 2 * torch.mean(kernels[:source.shape[0], source.shape[0]:])

        # Instance-level GWD
        # Cs, Ct = (source.unsqueeze(1) - source.unsqueeze(0))**2, (target.unsqueeze(1) - target.unsqueeze(0))**2
        # loss = torch.sum((Cs.mean(dim=1).mean(dim=0) - Ct.mean(dim=1).mean(dim=0))**2)

        # Feature-level GWD
        # Cs, Ct = (source.transpose(0, 1).unsqueeze(1) - source.transpose(0, 1).unsqueeze(1))**2, \
        #          (target.transpose(0, 1).unsqueeze(1) - target.transpose(0, 1).unsqueeze(1))**2
        # loss = torch.norm(Cs.sum(2) - Ct.sum(2), p=2) / (4 * source.shape[1] * source.shape[1])

        # Instance-level Mean GWD
        Cs, Ct = (source.unsqueeze(1) - source.unsqueeze(0)) ** 2, (target.unsqueeze(1) - target.unsqueeze(0)) ** 2
        loss = torch.norm(Cs.sum(2) - Ct.sum(2), p=2) / (source.shape[0] * source.shape[0])

        return loss


class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()

    def forward(self, source, target):
        d = source.shape[1]
        ns, nt = source.shape[0], target.shape[0]

        D_s = torch.sum(source, dim=0, keepdim=True)
        C_s = (torch.matmul(source.t(), source) - torch.matmul(D_s.t(), D_s) / ns) / (ns - 1)

        D_t = torch.sum(target, dim=0, keepdim=True)
        C_t = (torch.matmul(target.t(), target) - torch.matmul(D_t.t(), D_t) / nt) / (nt - 1)

        loss = torch.norm(C_s - C_t, p=2)
        loss = loss / (4 * d * d)
        return loss


def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)


def sgw_gpu(xs, xt, device, nproj=200, tolog=False, P=None):
    """ Returns SGW between xs and xt eq (4) in [1]. Only implemented with the 0 padding operator Delta
    Parameters
    ----------
    xs : tensor, shape (n, p)
         Source samples
    xt : tensor, shape (n, q)
         Target samples
    device :  torch device
    nproj : integer
            Number of projections. Ignore if P is not None
    P : tensor, shape (max(p,q),n_proj)
        Projection matrix. If None creates a new projection matrix
    tolog : bool
            Wether to return timings or not
    Returns
    -------
    C : tensor, shape (n_proj,1)
           Cost for each projection
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    Example
    ----------
    import numpy as np
    import torch
    from sgw_pytorch import sgw

    n_samples=300
    Xs=np.random.rand(n_samples,2)
    Xt=np.random.rand(n_samples,1)
    xs=torch.from_numpy(Xs).to(torch.float32)
    xt=torch.from_numpy(Xt).to(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P=np.random.randn(2,500)
    sgw_gpu(xs,xt,device,P=torch.from_numpy(P).to(torch.float32))
    """
    if tolog:
        log = {}

    if tolog:
        st = time.time()
        xsp, xtp = sink_(xs, xt, device, nproj, P)
        ed = time.time()
        log['time_sink_'] = ed - st
    else:
        xsp, xtp = sink_(xs, xt, device, nproj, P)
    if tolog:
        st = time.time()
        d, log_gw1d = gromov_1d(xsp, xtp, tolog=True)
        ed = time.time()
        log['time_gw_1D'] = ed - st
        log['gw_1d_details'] = log_gw1d
    else:
        d = gromov_1d(xsp, xtp, tolog=False)

    if tolog:
        return d, log
    else:
        return d


def _cost(xsp, xtp, tolog=False):
    """ Returns the GM cost eq (3) in [1]
    Parameters
    ----------
    xsp : tensor, shape (n, n_proj)
         1D sorted samples (after finding sigma opt) for each proj in the source
    xtp : tensor, shape (n, n_proj)
         1D sorted samples (after finding sigma opt) for each proj in the target
    tolog : bool
            Wether to return timings or not
    Returns
    -------
    C : tensor, shape (n_proj,1)
           Cost for each projection
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    """
    st = time.time()

    xs = xsp
    xt = xtp

    xs2 = xs * xs
    xs3 = xs2 * xs
    xs4 = xs2 * xs2

    xt2 = xt * xt
    xt3 = xt2 * xt
    xt4 = xt2 * xt2

    X = torch.sum(xs, 0)
    X2 = torch.sum(xs2, 0)
    X3 = torch.sum(xs3, 0)
    X4 = torch.sum(xs4, 0)

    Y = torch.sum(xt, 0)
    Y2 = torch.sum(xt2, 0)
    Y3 = torch.sum(xt3, 0)
    Y4 = torch.sum(xt4, 0)

    xxyy_ = torch.sum((xs2) * (xt2), 0)
    xxy_ = torch.sum((xs2) * (xt), 0)
    xyy_ = torch.sum((xs) * (xt2), 0)
    xy_ = torch.sum((xs) * (xt), 0)

    n = xs.shape[0]

    C2 = 2 * X2 * Y2 + 2 * (n * xxyy_ - 2 * Y * xxy_ - 2 * X * xyy_ + 2 * xy_ * xy_)

    power4_x = 2 * n * X4 - 8 * X3 * X + 6 * X2 * X2
    power4_y = 2 * n * Y4 - 8 * Y3 * Y + 6 * Y2 * Y2

    C = (1 / (n ** 2)) * (power4_x + power4_y - 2 * C2)

    ed = time.time()

    if not tolog:
        return C
    else:
        return C, ed - st


def gromov_1d(xs, xt, tolog=False):
    """ Solves the Gromov in 1D (eq (2) in [1] for each proj
    Parameters
    ----------
    xsp : tensor, shape (n, n_proj)
         1D sorted samples for each proj in the source
    xtp : tensor, shape (n, n_proj)
         1D sorted samples for each proj in the target
    tolog : bool
            Wether to return timings or not
    fast: use the O(nlog(n)) cost or not
    Returns
    -------
    toreturn : tensor, shape (n_proj,1)
           The SGW cost for each proj
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    """

    if tolog:
        log = {}

    st = time.time()
    xs2, i_s = torch.sort(xs, dim=0)

    if tolog:
        xt_asc, i_t = torch.sort(xt, dim=0)  # sort increase
        xt_desc, i_t = torch.sort(xt, dim=0, descending=True)  # sort deacrese
        l1, t1 = _cost(xs2, xt_asc, tolog=tolog)
        l2, t2 = _cost(xs2, xt_desc, tolog=tolog)
    else:
        xt_asc, i_t = torch.sort(xt, dim=0)
        xt_desc, i_t = torch.sort(xt, dim=0, descending=True)
        l1 = _cost(xs2, xt_asc, tolog=tolog)
        l2 = _cost(xs2, xt_desc, tolog=tolog)
    toreturn = torch.mean(torch.min(l1, l2))
    ed = time.time()

    if tolog:
        log['g1d'] = ed - st
        log['t1'] = t1
        log['t2'] = t2

    if tolog:
        return toreturn, log
    else:
        return toreturn


def sink_(xs, xt, device, nproj=200, P=None):  # Delta operator (here just padding)
    """ Sinks the points of the measure in the lowest dimension onto the highest dimension and applies the projections.
    Only implemented with the 0 padding Delta=Delta_pad operator (see [1])
    Parameters
    ----------
    xs : tensor, shape (n, p)
         Source samples
    xt : tensor, shape (n, q)
         Target samples
    device :  torch device
    nproj : integer
            Number of projections. Ignored if P is not None
    P : tensor, shape (max(p,q),n_proj)
        Projection matrix
    Returns
    -------
    xsp : tensor, shape (n,n_proj)
           Projected source samples
    xtp : tensor, shape (n,n_proj)
           Projected target samples
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    """
    dim_d = xs.shape[1]
    dim_p = xt.shape[1]

    if dim_d < dim_p:
        random_projection_dim = dim_p
        xs2 = torch.cat((xs, torch.zeros((xs.shape[0], dim_p - dim_d)).to(device)), dim=1)
        xt2 = xt
    else:
        random_projection_dim = dim_d
        xt2 = torch.cat((xt, torch.zeros((xt.shape[0], dim_d - dim_p)).to(device)), dim=1)
        xs2 = xs

    if P is None:
        P = torch.randn(random_projection_dim, nproj)
    p = P / torch.sqrt(torch.sum(P ** 2, 0, True))

    try:
        xsp = torch.matmul(xs2, p.to(device))
        xtp = torch.matmul(xt2, p.to(device))
    except RuntimeError as error:
        print('----------------------------------------')
        print('xs origi dim :', xs.shape)
        print('xt origi dim :', xt.shape)
        print('dim_p :', dim_p)
        print('dim_d :', dim_d)
        print('random_projection_dim : ', random_projection_dim)
        print('projector dimension : ', p.shape)
        print('xs2 dim :', xs2.shape)
        print('xt2 dim :', xt2.shape)
        print('xs_tmp dim :', xs2.shape)
        print('xt_tmp dim :', xt2.shape)
        print('----------------------------------------')
        print(error)

    return xsp, xtp
