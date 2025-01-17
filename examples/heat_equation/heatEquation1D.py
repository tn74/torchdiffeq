import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import random


class heatequation1D():
	data = None
	def __init__(self, L=0.1, T0=[40,0,30], dx=0.01, alpha=0.0001, t_final=50, dt=0.1):
		"""
        Construct a new 'headequation1D' object.

        :param L: Length of rod
        :param T0: Initial temperatures.
        :param dx: Distance between two discrete points on rod.
        :param alpha: Temperature transfer constant of rod (Based on material)
        :param t_final: Final time
        :param dt: Difference between two discrete times when temperature is measured.
        :return: returns nothing
        """
		self.L = L
		self.T0 = T0
		self.room_temp = 0
		self.dx = dx
		self.alpha = alpha
		self.t_final = t_final
		self.dt = dt
		self.n = len(T0[0])

	def singleDimDelta(self,t0,t1,t2):
		return self.alpha*(-(t1-t0)/self.dx**2 + (t2 - t1)/self.dx**2)

	def generate_data(self):
		ans = []
		self.x = np.linspace(self.dx/2, self.L-self.dx/2, self.n)
		t = np.arange(0, self.t_final, self.dt)
		T = np.asarray(self.T0)
		ans.append(T)
		for j in range(1, len(t)):
			dTdt = np.zeros((2,self.n))
			for i in range(self.n-1):
				# If rod's position is heat-source, temperature is constant.
				if (self.T0[1][i] == 1):
					dTdt[0][i] = 0
				# If rod's position isn't heat-source, temperature will change.
				else:
					dTdt[0][i] = self.singleDimDelta(T[0][i-1], T[0][i], T[0][i+1])
			# If left bound of rod isn't heat source, use room temperature as left bound.
			if self.T0[1][0] == 0:
				dTdt[0][0] = self.singleDimDelta(self.room_temp, T[0][0], T[0][1])
			# If right bound of rod isn't heat source, use room temperature as right bound.
			if self.T0[1][self.n-1] == 0:
				dTdt[0][self.n-1] = self.singleDimDelta(T[0][self.n-2], T[0][self.n-1], self.room_temp)
			T = T + dTdt*self.dt
			ans.append(T)
		self.data = np.array(ans)
		return self.data

# T0 = [[40, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 20], [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
# heat_eqn_inst1_2 = heatequation1D(L=0.1, T0=T0, dx=0.01, alpha=0.0001, t_final=100, dt=0.1)
# heat_eqn_inst1_y1_2 = heat_eqn_inst1_2.generate_data()


def generate_truth_sampler_he1D(rod_size=12, num_init_states=100, L=0.1, dx=0.01, alpha=0.0001, t_final=100, dt=0.1):
	ans = []
	for i in range(num_init_states):
		T0 = np.zeros((2,rod_size))
		numHeatSources = random.randint(0,int(rod_size/3)) + 1
		heatSourceIndexes = np.random.choice(np.arange(0,rod_size), numHeatSources, replace=False)
		for index in np.nditer(heatSourceIndexes):
			T0[1][index] = 1
		heatValues = np.random.uniform(low=0,high=273, size=rod_size)
		heatequation1D_obj = heatequation1D(L=L, T0=T0, dx=dx, alpha=alpha, t_final=t_final, dt=dt)
		data = heatequation1D_obj.generate_data()
		ans.append(data)
	res = np.stack(ans)
	return res

if __name__ == "__main__":
    print(np.shape(generate_truth_sampler_he1D(rod_size=12, num_init_states=100)))









