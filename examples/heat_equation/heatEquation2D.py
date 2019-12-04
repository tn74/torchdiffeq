import numpy as np
import matplotlib.pyplot as plt
import time
import random



class heatequation2D():
    data = None
    def __init__(self, w=10, h=10, dx=0.1, dy=0.1, D=4):
        """
        Construct a new 'headequation1D' object.

        :param w: plate width
        :param h: plate height
        :param dx: intervals in x direction
        :param dy: intervals in y direction. 
        :param D: thermal diffusivity of material
        :return: returns nothing
        """
        # plate size, mm
        self.D = D
        self.dx = dx
        self.dy = dy
        self.nx = int(w/dx)
        self.ny = int(w/dy)
        self.dx2 = dx*dx
        self.dy2 = dy*dy
        self.dt = self.dx2 * self.dy2 / (2 * D * (self.dx2 + self.dy2))
        self.u0 = np.zeros((self.nx, self.ny,2))
        self.u = np.zeros((self.nx, self.ny, 2))
    
    def initializeRing(self, r=2,cx=2,cy=2):
        r2 = r**2
        for i in range(self.nx):
            for j in range(self.ny):
                p2 = (i*self.dx-cx)**2 + (j*self.dy-cy)**2
                if p2 < r2:
                    self.u0[i,j,0] = 700
        print(np.shape(self.u0))
    
    def initializeRing_random(self, r=2,cx=2,cy=2):
        r2 = r**2
        for i in range(self.nx):
            for j in range(self.ny):
                p2 = (i*self.dx-cx)**2 + (j*self.dy-cy)**2
                if p2 < r2:
                    self.u0[i,j,0] = float(random.randint(300,1000))
                    # Heat sources are initialized here. 
                    self.u0[i,j,1] = 1 if random.randint(1,3) == 1 else 0

    def initializeBox_random(self, r=2,cx=2,cy=2):
        for i in range(self.nx):
            for j in range(self.ny):
                if abs(cx - i) < r and abs(cy - j) < r:
                    self.u0[i,j,0] = float(random.randint(300,1000))
                    # Heat sources are initialized here. 
                    self.u0[i,j,1] = 1 if random.randint(1,3) == 1 else 0

    def initializeRandom(self):
        for i in range(self.nx):
            for j in range(self.ny):
                if random.randint(1,3) == 1:
                    self.u0[i,j,0] = float(random.randint(300,1000))
                    # Heat sources are initialized here.
                    self.u0[i,j,1] = 1 if random.randint(1,3) == 1 else 0
    
    def do_timestep(self,u0, u):
        # Propagate with forward-difference in time, central-difference in space
        u[1:-1, 1:-1, 0] = u0[1:-1, 1:-1, 0] + (1-u0[1:-1, 1:-1, 1])*(self.D * self.dt * (
              (u0[2:, 1:-1, 0] - 2*u0[1:-1, 1:-1, 0] + u0[:-2, 1:-1, 0])/self.dx2
              + (u0[1:-1, 2:, 0] - 2*u0[1:-1, 1:-1, 0] + u0[1:-1, :-2, 0])/self.dy2))
        u0 = u.copy()
        return u0, u

    def generate_data(self, nsteps=10):
        ans = []
        ans.append(self.u0)
        seconds = time.time()
        for m in range(nsteps):
            self.u0, self.u = self.do_timestep(self.u0, self.u)
            ans.append(self.u.copy())
        print("Seconds since epoch =", time.time()-seconds)
        self.data = np.array(ans)
        return self.data

# heatequation2D_inst = heatequation2D(w=10, h=10, dx=0.1, dy=0.1, D=4)
# heatequation2D_inst.initializeRing_random(r=2,cx=5,cy=5);
# data = heatequation2D_inst.generate_data(nsteps=1000);

def generate_truth_sampler_he2D(size = 10, num_init_states=5, dx = 0.1, D=4):
    ans = []
    for i in range(num_init_states):
        print(i)
        # Instantiate heat equation model. 
        heatequation2D_inst = heatequation2D(w=size, h=size, dx=dx, dy=dx, D=4)

        # Setup initial State 
        curr_init_state = random.randint(1,3)
        
        # Initialize Rings
        if (curr_init_state == 1):
            numRings = random.randint(1,3)
            for i in range(numRings):
                heatequation2D_inst.initializeRing_random(r=random.randint(1,4),cx=random.randint(4,6),cy=random.randint(4,6))
        
        # Initialize Boxes
        if (curr_init_state == 2):
            numBoxes = random.randint(1,3)
            for i in range(numBoxes):
                heatequation2D_inst.initializeBox_random(r=random.randint(1,4),cx=random.randint(4,6),cy=random.randint(4,6))   

        # Initialize Random
        if (curr_init_state == 3):
            heatequation2D_inst.initializeRandom()

        # Generate data and append to list
        data = heatequation2D_inst.generate_data(nsteps=1000)
        ans.append(data)

    # Stack list
    res = np.stack(ans)

    return res
        



data = generate_truth_sampler_he2D(size = 10, num_init_states=5, dx = 0.1, D=4)
print(np.shape(data))