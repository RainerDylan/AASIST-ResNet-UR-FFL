import torch

class UncertaintySensor:
    def __init__(self, mc_passes=10):
        self.mc_passes = mc_passes

    def measure(self, model, waveforms):
        model.eval()
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                
        with torch.no_grad():
            outputs = []
            for _ in range(self.mc_passes):
                logits = model(waveforms)
                probs = torch.softmax(logits, dim=1)[:, 1] 
                outputs.append(probs.unsqueeze(0))
                
            outputs = torch.cat(outputs, dim=0) 
            
            # Standard Deviation is used instead of Variance to prevent Softmax crushing
            epistemic_std = outputs.std(dim=0, unbiased=False)
            
            batch_mu = epistemic_std.mean()
            batch_std_dev = epistemic_std.std() + 1e-8
            
            z_u = (epistemic_std - batch_mu) / batch_std_dev
            
        return z_u, batch_mu.item()