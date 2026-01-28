import numpy as np

# PASTE params directly (no import needed)
NP_DESIGNS = {
    "AMT_Gold": {"size": 15, "zeta": 25, "tf_ligand": 0, "color": "red"},
    "RMT_PBCA": {"size": 80, "zeta": 0, "tf_ligand": 1, "color": "blue"}, 
    "DUAL_GoldTf": {"size": 15, "zeta": 15, "tf_ligand": 1, "color": "gold"}
}

RATES = {"amt_base": 1e-4, "rmt_base": 5e-5, "synergy_factor": 0.4}

class NP:
    def __init__(self, design):
        params = NP_DESIGNS[design]
        self.size, self.zeta, self.tf_ligand = params["size"], params["zeta"], params["tf_ligand"]
        self.color = params["color"]
        self.pos = np.array([np.random.uniform(-50,50), np.random.uniform(-50,50), np.random.uniform(1.5, 2.5)])
        self.velocity = np.zeros(3)
        self.crossed = False
        
    def AMT_rate(self):
        return RATES["amt_base"] * (self.zeta ** 2) * np.exp(-self.size/50)
    
    def RMT_rate(self):
        return RATES["rmt_base"] * self.tf_ligand
    
    def step(self, dt=0.1):
        if self.crossed: return
        z = self.pos[2]
        if 0.5 <= z <= 1.0:
            amt, rmt = self.AMT_rate(), self.RMT_rate()
            rate = amt + rmt + RATES["synergy_factor"] * np.sqrt(amt * rmt)
            if np.random.random() < rate * dt:
                self.velocity[2] = -0.1
                self.crossed = True
        self.velocity += np.random.normal(0, 0.01, 3)
        self.velocity[2] -= 0.005
        self.pos += self.velocity * dt

def run_simulation(design, steps=300, N=500):
    agents = [NP(design) for _ in range(N)]
    for _ in range(steps):
        for agent in agents:
            agent.step()
    return np.array([agent.pos for agent in agents])
