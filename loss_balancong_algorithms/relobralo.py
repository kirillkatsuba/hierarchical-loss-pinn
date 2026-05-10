import torch
import torch.nn.functional as F

class ReLoBRaLo:
    """
    Relative Loss Balancing with Random Lookback (ReLoBRaLo).
    Dynamically adjusts the weights of different loss components during training.
    """
    
    def __init__(self, num_losses, alpha=0.999, temperature=0.1, rho_prob=0.99):
        """
        Initializes the ReLoBRaLo loss balancing algorithm.
        
        Args:
            num_losses (int): Number of components in the loss function.
            alpha (float, optional): Exponential moving average decay rate. Defaults to 0.999.
            temperature (float, optional): Temperature scaling for the softmax function. Defaults to 0.1.
            rho_prob (float, optional): Probability for the Bernoulli random variable to be 1 (random lookback). Defaults to 0.99.
        """
        self.num_losses = num_losses
        self.alpha = alpha
        self.temperature = temperature
        self.rho_prob = rho_prob
        
        self.lambdas = torch.ones(num_losses)
        self.init_losses = None
        self.prev_losses = None

    def compute_weights(self, current_losses):
        """
        Computes and updates the balancing weights for the current training iteration.
        
        Args:
            current_losses (torch.Tensor): A 1D tensor containing the current values of the loss components.
        
        Returns:
            torch.Tensor: A tensor containing the updated weights for each loss component.
        """
        L_curr = current_losses.clone().detach()
        
        if self.init_losses is None:
            self.init_losses = L_curr
            self.prev_losses = L_curr
            self.lambdas = self.lambdas.to(L_curr.device)
            return self.lambdas
        
        eps = 1e-8 
        
        rel_loss_init = L_curr / (self.init_losses + eps)
        rel_loss_prev = L_curr / (self.prev_losses + eps)
        
        lambda_hat_init = self.num_losses * F.softmax(rel_loss_init / self.temperature, dim=0)
        lambda_hat_prev = self.num_losses * F.softmax(rel_loss_prev / self.temperature, dim=0)
        
        rho = torch.bernoulli(torch.tensor(self.rho_prob, device=L_curr.device))
        
        new_lambdas = self.alpha * (rho * self.lambdas + (1 - rho) * lambda_hat_init) + \
                      (1 - self.alpha) * lambda_hat_prev
                      
        self.lambdas = new_lambdas
        self.prev_losses = L_curr
        
        return self.lambdas
        