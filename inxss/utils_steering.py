import torch

class OnlineVariance:
    """
    A class to calculate variance in an online manner with device compatibility.

    Online Variance Calculation using Welford's method
    Author: ChatGPT at OpenAI
    
    Attributes:
        n (int): The number of samples seen so far.
        mean (torch.Tensor): The current mean.
        M2 (torch.Tensor): Intermediate calculation tensor for variance.
        
    Methods:
        update(x): Update variance with new sample x.
        variance(sample=True): Return the variance.
        std_dev(sample=True): Return the standard deviation.
        reset(): Reset accumulated data for a fresh start.
    """
    
    def __init__(self, shape, device=torch.device("cpu")):
        """Initialize with the given shape for tensors and device."""
        self.shape = shape
        self.device = device
        self.reset()

    def to(self, device):
        """Move the tensors to the specified device."""
        self.mean = self.mean.to(device)
        self.M2 = self.M2.to(device)
        self.device = device

    def update_single(self, x):
        """
        Update variance calculator with a new sample.
        
        Args:
            x (torch.Tensor): New sample tensor.
        """
        x = x.to(self.device)  # Ensure x is on the correct device before calculation
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        
    def update(self, x_batch):
        """
        Update variance calculator with a batch of samples.
        
        Args:
            x_batch (torch.Tensor): New batch of sample tensors. Should have shape (batch_size, *self.shape)
        """
        x_batch = x_batch.to(self.device)  # Ensure x_batch is on the correct device before calculation
        
        if x_batch.ndim == len(self.shape):
            x_batch = x_batch.unsqueeze(0)
            
        batch_size = x_batch.size(0)
        
        if batch_size == 1:
            self.update_single(x_batch.squeeze(0))
        else:
            # Compute mean and variance for the new batch
            batch_mean = torch.mean(x_batch, dim=0)
            batch_var = torch.var(x_batch, dim=0, unbiased=False) * (batch_size / (batch_size - 1))
            
            # Update overall count
            new_count = self.n + batch_size
            
            # Compute pooled mean
            pooled_mean = (self.mean * self.n + batch_mean * batch_size) / new_count
            
            # Update M2 using the new batch's variance and difference in means
            self.M2 += batch_var * (batch_size - 1) + (self.mean - batch_mean) ** 2 * self.n * batch_size / new_count
            
            # Update mean and count
            self.mean = pooled_mean
            self.n = new_count

    def variance(self, sample=True):
        """
        Calculate the variance.
        
        Args:
            sample (bool): If True, use sample variance formula. Default is True.
        
        Returns:
            torch.Tensor: Variance tensor.
        """
        if self.n < 2:
            return float('nan')
        if sample:
            return self.M2 / (self.n - 1)
        return self.M2 / self.n

    def std_dev(self, sample=True):
        """
        Calculate the standard deviation.
        
        Args:
            sample (bool): If True, use sample standard deviation formula. Default is True.
        
        Returns:
            torch.Tensor: Standard deviation tensor.
        """
        return torch.sqrt(self.variance(sample))
    
    def reset(self):
        """Reset accumulated data for a fresh start."""
        self.n = 0
        self.mean = torch.zeros(self.shape, device=self.device)
        self.M2 = torch.zeros(self.shape, device=self.device)

# # Example usage:

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Initialize the online variance calculator
# var_calculator = OnlineVariance((N1, N2, N3), device=device)

# # For each input tensor2 and the resulting output, update the calculator
# for tensor2 in your_tensors:
#     output = your_model(tensor2.to(device))
#     var_calculator.update(output)

# # Get the final standard deviation
# std_dev = var_calculator.std_dev()

# # Reset for a fresh start
# var_calculator.reset()
