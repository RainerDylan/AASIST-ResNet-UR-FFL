class PDController:
    def __init__(self):
        # Target an exact 8% increase in model Standard Deviation caused by the noise
        self.target_delta = 0.08 
        
        # High proportional gain forces immediate response. No integral memory to get stuck.
        self.kp = 6.0
        self.kd = 2.0
        
        self.min_alpha = 0.1
        self.max_alpha = 0.9 
        
        self.base_alpha = 0.5
        self.prev_error = 0.0
        self.alpha = 0.5 

    def compute_severity(self, clean_std, deg_std):
        actual_delta = deg_std - clean_std
        error = self.target_delta - actual_delta
        
        p_term = self.kp * error
        d_term = self.kd * (error - self.prev_error)
        
        # Pure spring logic anchored at 0.5. 
        # Mathematically guarantees smooth traversal of the alpha spectrum without locking.
        alpha_raw = self.base_alpha + p_term + d_term
        
        self.alpha = max(self.min_alpha, min(alpha_raw, self.max_alpha))
        self.prev_error = error
        
        return self.alpha