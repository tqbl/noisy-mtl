import torch
import torch.nn.functional as F

from .baseline import Baseline, cce


class MultiTaskSystem(Baseline):
    def __init__(self, model, lr, lr_scheduler, model_args=None):
        super().__init__(model, lr, lr_scheduler, model_args)

        # Use the MTL loss function
        self.criterion = MultiTaskLoss()
        self.add_hyperparameters(criterion=self.criterion)

    def forward(self, x, logits=False, training=False):
        if isinstance(x, tuple):
            x, indexes = x
        else:
            indexes = None

        y = self.model(x, training)
        if not logits and self.activation is not None:
            y = self.activation(y)

        if indexes is None:
            return y

        return {'y_pred': y, 'indexes': indexes}

    def training_step(self, batch):
        batch_clean, batch_noisy = batch
        output_clean = self(batch_clean[0], logits=True, training=True)
        output_noisy = self(batch_noisy[0], logits=True, training=True)
        output = (torch.cat([output_clean[0], output_noisy[0]]),
                  torch.cat([output_clean[1], output_noisy[1]]))
        batch_y = torch.cat([batch_clean[1], batch_noisy[1]])
        weight = torch.cat([torch.ones_like(output_clean[0][:, 0]),
                            torch.zeros_like(output_noisy[0][:, 0])])

        loss, loss_cce, loss_js = self.criterion(output, batch_y, weight)
        if loss is not None:
            self.log('loss', loss.item())
            self.log('loss_cce', loss_cce.item())
            self.log('loss_js', loss_js.item())
        return loss


class MultiTaskDataLoader:
    def __init__(self, loader_clean, loader_noisy):
        self.loader_clean = loader_clean
        self.loader_noisy = loader_noisy

    def to(self, device):
        self.loader_clean.device = device
        self.loader_noisy.device = device

    def __iter__(self):
        iter_clean = self.loader_clean.__iter__()
        for batch_noisy in self.loader_noisy:
            try:
                batch_clean = next(iter_clean)
            except StopIteration:
                iter_clean = self.loader_clean.__iter__()
                batch_clean = next(iter_clean)

            yield batch_clean, batch_noisy

    def __len__(self):
        return len(self.loader_noisy)


class MultiTaskLoss:
    def __init__(self, omega=1., gamma=10.):
        self.omega = omega
        self.gamma = gamma

    def __call__(self, y_pred, y_true, weight):
        if not isinstance(y_pred, tuple):
            return cce(y_pred, y_true)

        weight = weight.unsqueeze(1)

        y_pred_clean, y_pred_noisy = y_pred
        loss_clean = -y_true * y_pred_clean.log_softmax(dim=1)
        loss_noisy = -y_true * y_pred_noisy.log_softmax(dim=1)
        loss_cce = (weight * loss_clean).mean() \
            + self.omega * loss_noisy.mean()

        p_clean = weight * y_pred_clean.softmax(dim=1)
        p_noisy = weight * y_pred_noisy.softmax(dim=1)
        loss_js = self.gamma * js_divergence(p_clean, p_noisy)

        loss = loss_cce + loss_js

        return loss, loss_cce, loss_js


def js_divergence(p_clean, p_noisy):
    p_mixture = ((p_clean + p_noisy) / 2).clamp(1e-7, 1).log()
    dist = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
            F.kl_div(p_mixture, p_noisy, reduction='batchmean')) / 2
    return dist
