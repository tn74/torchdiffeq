import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


class heatequation1D():
	data = None
	def __init__(self, L, hsi, T0, room_temp, dx, alpha, t_final, dt):
		"""
        Construct a new 'headequation1D' object.

        :param L: Length of rod
        :param hsi: the indexes of the heat sources on the rod.
        :param T0: Initial temperatures.
        :param dx: Distance between two discrete points on rod.
        :param alpha: Temperature transfer constant of rod (Based on material)
        :param t_final: Final time
        :param dt: Difference between two discrete times when temperature is measured.
        :return: returns nothing
        """
		self.L = L
		self.hsi = heat_source_idxs
		self.T0 = T0
		self.room_temp = 0
		self.dx = dx
		self.alpha = alpha
		self.t_final = t_final
		self.dt = dt
		self.n = len(T0)


	def singleDimDelta(self,t0,t1,t2):
		return self.alpha*(-(t1-t0)/self.dx**2 + (t2 - t1)/self.dx**2)

	def generate_data(self):
		ans = []
		self.x = np.linspace(self.dx/2, self.L-self.dx/2, self.n)
		dTdt = np.empty(self.n)
		t = np.arange(0, self.t_final, self.dt)
		T = np.asarray(self.T0)
		ans.append(T)
		for j in range(1, len(t)):
			for i in range(0,self.n):
				# If rod's position is heat-source, temperature is constant.
				if (i in self.hsi):
					dTdt[i] = 0
				# If rod's position isn't heat-source, temperature will change.
				else:
					dTdt[i] = self.singleDimDelta(T[i-1], T[i], T[i+1])
			# If left bound of rod isn't heat source, use room temperature as left bound.
			if 0 not in self.hsi:
				dTdt[0] = self.singleDimDelta(self.room_temp, T[0], T[1])
			# If right bound of rod isn't heat source, use room temperature as right bound.
			if n-1 not in self.hsi:
				dTdt[n-1] = self.singleDimDelta(T[n-2], T[n-1], self.room_temp)
			T = T + dTdt*self.dt
			ans.append(T)
		self.data = np.array(ans)
		return self.data

room_temp = 0
T0 = [40, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 20]
heat_source_idxs = set([0, 2, 11])
heat_eqn_inst1_2 = heatequation1D(L=0.1, T0=T0, room_temp=0, hsi=heat_source_idxs, dx=0.01, alpha=0.0001, t_final=100, dt=0.1)
heat_eqn_inst1_y1_2 = heat_eqn_inst1_2.generate_data()
