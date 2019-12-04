import numpy as np
import matplotlib.pyplot as plt



class heatequation2D():
    data = None
    def __init__(self, w=10, h=10, dx=0.1, dy=0.1, D=4, Tcool=300, Thot=700):
        """
        Construct a new 'headequation1D' object.

        :param w: plate width
        :param h: plate height
        :param dx: intervals in x direction
        :param dy: intervals in y direction. 
        :param D: thermal diffusivity of material
        :param Tcool: 
        :param Thot: 
        :return: returns nothing
        """
        # plate size, mm
        self.D = D
        self.Tcool = Tcool
        self.dx = dx
        self.dy = dy
        self.Thot = Thot
        self.nx = int(w/dx)
        self.ny = int(w/dy)
        self.dx2 = dx*dx
        self.dy2 = dy*dy
        self.dt = self.dx2 * self.dy2 / (2 * D * (self.dx2 + self.dy2))
        self.u0 = Tcool * np.ones((self.nx, self.ny))
        self.u = np.empty((self.nx, self.ny))
    
    def initializeRing(self, r=2,cx=2,cy=2):
        r2 = r**2
        for i in range(self.nx):
            for j in range(self.ny):
                p2 = (i*self.dx-cx)**2 + (j*self.dy-cy)**2
                if p2 < r2:
                    self.u0[i,j] = self.Thot

    def customInitialization(self, heatIndexSet):
        for i in range(self.nx):
            for j in range(self.ny):
                if (i,j) in heatIndexMap:
                    self.u0[i,j] = heatIndexMap[(i,j)]
    
    def do_timestep(self,u0, u):
        # Propagate with forward-difference in time, central-difference in space
        u[1:-1, 1:-1] = u0[1:-1, 1:-1] + self.D * self.dt * (
              (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/self.dx2
              + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/self.dy2)
        u0 = u.copy()
        return u0, u


    def generate_data(self, nsteps=10):
        ans = []
        ans.append(self.u0)
        for m in range(nsteps):
            self.u0, self.u = self.do_timestep(self.u0, self.u)
            ans.append(self.u.copy())
        self.data = np.array(ans)
        return self.data

heatequation2D_inst = heatequation2D(w=10, h=10, dx=0.1, dy=0.1, D=4, Tcool=300, Thot=700);
heatequation2D_inst.initializeRing(r=2,cx=5,cy=5);
data = heatequation2D_inst.generate_data(nsteps=1000);
print(data.shape)
