class PDController:
    def __init__(self):
        # Restored to exact thesis specification
        self.target = 0.15 
        
        # Standard, stable PID tuning
        self.kp = 0.3
        self.kd = 0.1
        
        self.min_alpha = 0.1
        self.max_alpha = 0.9 
        
        self.prev_error = 0.0
        self.alpha = 0.3 # Start at a low/moderate severity

    def compute_severity(self, mean_deg_uncertainty):
        # Error is now properly calculated against the model's reaction to the noise
        error = self.target - mean_deg_uncertainty
        
        p_term = self.kp * error
        d_term = self.kd * (error - self.prev_error)
        
        # True mathematical accumulator for closed-loop dynamic surfing
        self.alpha += (p_term + d_term)
        
        # Strict constraints
        self.alpha = max(self.min_alpha, min(self.alpha, self.max_alpha))
        self.prev_error = error
        
        return self.alpha