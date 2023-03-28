import numpy as np
import pandas as pd


imu = pd.read_fwf('imu.txt')
trj = pd.read_fwf('trj.txt')
print(imu.columns.tolist())
t = imu['time[s]']
t_data = trj['time[s]']


a = float(6378137)   # большая полуось эллипса
b = float(6356752)   # малая полуось эллипса
e = float((a*a-b*b)/a**2)  # квадрат эксцентриситета

U = float(7.292115e-5)  # угловая скорость земли в рад/c
e2 = e
R = float(1/((1/a**2+3/b**2)**(1/2)))

# начальная выставка - > берем данные до 2ух минут и осредняем, превращая в вектора
f_z_start = (imu[['fz1[mps2]', 'fz2[mps2]' , 'fz3[mps2]']][:12002].mean()).to_numpy()
w_z_start = (imu[['wz1[dps]', 'wz2[dps]', 'wz3[dps]']][:12002].mean()).to_numpy()

l_start = np.array([np.cross(
    w_z_start, f_z_start)/np.linalg.norm(np.cross(w_z_start, f_z_start)),
    np.cross(np.cross(
     w_z_start, f_z_start), f_z_start)/np.linalg.norm(np.cross(np.cross(
      w_z_start, f_z_start), f_z_start)), f_z_start/np.linalg.norm(f_z_start)]).transpose()

print("")
