import numpy as np

class NP:
    def __init__(self, design_params):
        self.size = design_params["size"]
        self.zeta = design_params["zeta"]
        self.tf_ligand = design_params["tf_ligand"]
        self.color = design_params["color"]
        self.pos = np.array([np.random.uniform(-50,50), 
                           np.random.uniform(-50,50), 
                           np.random.uniform(1.5, 2.5)])
        self.velocity = np.zeros(3)
        self.crossed = False
        
    def AMT_rate(self, rates):
        return rates["amt_base"] * (self.zeta ** 2) * np.exp(-self.size/50)
    
    def RMT_rate(self, rates):
        return rates["rmt_base"] * self.tf_ligand
    
    def step(self, rates, dt=0.1):
        if self.crossed: return
            
        z = self.pos[2]
        
        # Endothelium interaction (z=0.5-1.0)
        if 0.5 <= z <= 1.0:
            amt_rate = self.AMT_rate(rates)
            rmt_rate = self.RMT_rate(rates)
            total_rate = (amt_rate + rmt_rate + 
                         rates["synergy_factor"] * np.sqrt(amt_rate * rmt_rate))
            
            if np.random.random() < total_rate * dt:
                self.velocity[2] = -0.1
                self.crossed = True
                
        # Physics
        self.velocity += np.random.normal(0, 0.01, 3)
        self.velocity[2] -= 0.005
        self.pos += self.velocity * dt

def run_simulation(design, steps=500, N=1000):
    from params import NP_DESIGNS, RATES
    model = []
    for i in range(N):
        model.append(NP(NP_DESIGNS[design]))
    
    for step in range(steps):
        for np_agent in model:
            np_agent.step(RATES)
    
    return np.array([np_agent.pos for np_agent in model])
