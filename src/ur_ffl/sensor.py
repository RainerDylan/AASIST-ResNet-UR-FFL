import torch

class UncertaintySensor:
    def __init__(self, mc_passes=50):
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
            
            mu = outputs.mean(dim=0)
            sigma_sq = outputs.var(dim=0, unbiased=False)
            
            aleatoric = mu * (1.0 - mu) + 1e-8
            u_epistemic = sigma_sq / aleatoric
            
            batch_mu = u_epistemic.mean()
            batch_std = u_epistemic.std() + 1e-8
            z_u = (u_epistemic - batch_mu) / batch_std
            
            # FIXED: Output the raw mean epistemic uncertainty for the PD Controller
            mean_raw_uncertainty = batch_mu.item()
            
        return z_u, mean_raw_uncertainty