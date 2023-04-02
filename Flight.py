import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


def gravity(phi, h, a) -> np.float64:
    """
        Модельное значение ускореня силы тяжести с учетом его зависимости от широты и высоты
        Формула Гельмерта с поправкой
    """
    c2 = 0.005302   # beta1
    c3 = 0.000007   # beta2
    c4 = 0.00014    # delta g
    ge = 9.78030
    # np.divide(2.0, 4.0)= 0.5 - деление
    return np.float64(ge * (1 + c2 * np.power((np.sin(phi)), 2) - c3 * np.power((np.sin(2 * phi)), 2) -
                            2 * np.divide(h, a)) - c4)


def l_orientation(theta, psi, gamma) -> np.array:
    """
        Получение матрицы ориентации
    """

    ans = np.array([[np.cos(theta)*np.sin(psi), np.cos(theta)*np.cos(psi), np.sin(theta)],
                   [-np.sin(theta)*np.sin(psi)*np.cos(gamma)+np.cos(psi)*np.sin(gamma),
                   -np.sin(theta)*np.cos(psi)*np.cos(gamma)-np.sin(psi)*np.sin(gamma),
                   np.cos(theta)*np.cos(gamma)],
                   [np.sin(theta)*np.sin(psi)*np.sin(gamma)+np.cos(psi)*np.cos(gamma),
                   np.sin(theta)*np.cos(psi)*np.sin(gamma)-np.sin(psi)*np.cos(gamma),
                   -np.cos(theta)*np.sin(gamma)]])
    return ans


def l_reverse(matrx):
    """
        Возвращает углы по матрицы L
        Функция arctan2() вычисляет arctan(x1/x2) и
         возвращает значение угла в правильном квадранте (четверти координатной плоскости).
    """
    psi = np.arctan2(matrx[0, 0], matrx[0, 1])
    theta = np.arctan2(matrx[0, 2], np.sqrt(matrx[0, 0] ** 2 + matrx[0, 1] ** 2))
    gamma = np.arctan2(-matrx[2, 2], matrx[1, 2])
    return psi, theta, gamma


def angular_velocity_matrix(omega) -> np.array:

    """
    Получает на вход вектор угловой скорости и составляет по нему матрицу омега с крышкой
    """
    return np.array([[0, omega[2], -omega[1]],
                     [-omega[2], 0, omega[0]],
                     [omega[1], -omega[0], 0]])


def radius_of_curvature(a, e, phi):
    """
    Вычисление радиуса кривизны сечений модельного эллипсоида
    :param phi: широта в градусах
    :param a: большая полуось Земли в метрах
    :param e: эксцентриситет Земли
    :return: R_E, R_N - радиусы кривизны Земли в направлении оси вращения и нормали к поверхности соответственно
Радиус кривизны в направлении экватора (Equatorial Radius of Curvature) обозначается как $R_E$.
 Он определяется как радиус кривизны земной поверхности в точке на экваторе, то есть
 как расстояние от центра Земли до поверхности Земли в точке на экваторе.
 Этот радиус кривизны используется для описания формы земной поверхности в направлении,
  перпендикулярном оси вращения Земли.
Радиус кривизны в направлении полюса (Polar Radius of Curvature) обозначается как $R_N$.
 Он определяется как радиус кривизны земной поверхности в точке на полюсе,
  то есть как расстояние от центра Земли до поверхности Земли в точке на полюсе.
  Этот радиус кривизны используется для описания формы земной поверхности в направлении, параллельном оси вращения Земли.
Значения $R_E$ и $R_N$ зависят от геодезической широты (latitude) $\varphi$ в данной точке и эллипсоида,
который используется для описания формы Земли.
    """

    r_e = np.float64(np.divide(a, np.sqrt(1 - np.power(e * np.sin(phi),2))))
    r_n = np.float64(np.divide(a * (1 - np.power(e,2)) , np.power((np.sqrt(1 - np.power(e,2) * np.power(np.sin(phi),2))),3)))
    return r_e, r_n


def integrable(a, omega, dt):
    omega_norm = np.linalg.norm(omega)
    omega_matrix = angular_velocity_matrix(omega)
    sin_omega_dt = np.sin(omega_norm * dt)
    cos_omega_dt = np.cos(omega_norm * dt)
    omega_norm_sq = omega_norm * omega_norm
    a_new = np.eye(3) + (sin_omega_dt / omega_norm) * omega_matrix + ((1 - cos_omega_dt) / omega_norm_sq) \
                 * omega_matrix @ omega_matrix
    return a_new.dot(a)


imu = pd.read_fwf('imu.txt')
trj = pd.read_fwf('trj.txt')
#print(imu.columns.tolist())

#Мировые константы
a = np.float64(6378137)   # большая полуось эллипса
b = np.float64(6356752)   # малая полуось эллипса
e = np.float64(0.006694478197993)  # квадрат эксцентриситета
#U = np.float64(0.00007292115)  # угловая скорость земли в рад/c
U = np.float64(0.00007272205)

# начальная выставка - > берем данные до 2ух минут и осредняем, превращая в вектора
force_z = (imu[['fz1[mps2]', 'fz2[mps2]', 'fz3[mps2]']][:12000].mean()).to_numpy()
omega_z = (imu[['wz1[dps]', 'wz2[dps]', 'wz3[dps]']][:12000].mean()).to_numpy()

L = np.array([np.cross(
    omega_z, force_z) / np.linalg.norm(np.cross(omega_z, force_z)),
              np.cross(force_z, np.cross(
     omega_z, force_z)) / np.linalg.norm(np.cross(force_z, np.cross(
      omega_z, force_z))), force_z / np.linalg.norm(force_z)]).transpose()

# kurs = np.deg2rad((trj['heading[deg]'])[:200].mean())
# kren = np.deg2rad((trj['roll[deg]'])[:200].mean())
# tangag = np.deg2rad((trj['pitch[deg]'])[:200].mean())
# L = l_orientation(tangag, kurs,kren)

# Конец начальной выставки

# Нужны начальные условия для уравнения движения, их можно взять прям из файла trj как я понял
# Достаточно прям первые данные брать или осреднять? я просто первые взял, с осреднением закомментил
# phi_0, lambda_0, h_0 = (trj[[ 'lat[deg]', 'lon[deg]', 'alt[m]']][:12000].mean()).to_numpy()
# x =(trj[['lat[deg]', 'lon[deg]', 'alt[m]']][:1]).to_numpy()

# Инициализация массивов, как научил Дмитрий

n = trj.shape[0]

phi = np.zeros(n+1)
lmbda = np.zeros(n+1)
h = np.zeros(n+1)
gr = np.zeros(n+1)
V = np.zeros((3, n+2))

heading = np.zeros(n+1)  # курс
pitch = np.zeros(n+1)  # тангаж
roll = np.zeros(n+1)  # крен

# Начальные условия phi_0,lambda_0,h_0 а так же для матриц и
h[0] = trj['alt[m]'][:100].mean()
lmbda[0] = np.deg2rad(trj['lon[deg]'][:100].mean())
phi[0] = np.deg2rad(trj['lat[deg]'][:100].mean())
gr[0] = gravity(phi[0], h[0], a)
A_z = L
A_x = np.eye(3)
forces_z = (imu[['fz1[mps2]', 'fz2[mps2]', 'fz3[mps2]']]).to_numpy()
omegas_z = np.deg2rad((imu[['wz1[dps]', 'wz2[dps]', 'wz3[dps]']]).to_numpy())

# прирост времени постоянный
dt = 0.01

start = time.time()


for i in range(n):
    Re, Rn = radius_of_curvature(a, e, phi[i])

    # угловая скорость географического трехгранника относительно земли
    omega_x = np.array([-V[1, i] / (Rn + h[i]), V[0, i] / (Re + h[i]), V[0, i] * np.tan(phi[i]) / (Re + h[i])])
    # угловая скорость земли
    ux = np.array([0, U * np.cos(phi[i]), U * np.sin(phi[i])])

    phi[i+1] = phi[i] + V[1, i] / (Rn + h[i]) * dt
    lmbda[i+1] = lmbda[i] + V[0, i] / ((Re + h[i]) * np.cos(phi[i])) * dt
    h[i+1] = h[i] + V[2, i] * dt
    gr[i] = gravity(phi[i], h[i], a)
    V[:, i + 1] = V[:, i] + ((angular_velocity_matrix(omega_x) + 2 * angular_velocity_matrix(ux)) @ V[:, i] + np.array(
        [0, 0, -gr[i]]) + L.T @ forces_z[i, :].T) * dt

    heading[i], pitch[i], roll[i] = l_reverse(L)

    A_x = integrable(A_x, omega_x + ux, dt)
    A_z = integrable(A_z, omegas_z[i, :], dt)
    L = np.matmul(A_z, np.transpose(A_x))


# Построение графиков
phi = np.array(phi)
phi_data = np.deg2rad(trj['lat[deg]'])
lmbda_data = np.deg2rad(trj['lon[deg]'])
h_data = trj['alt[m]']

lmbda = np.array(lmbda)
t = imu['time[s]'].to_numpy()/10
t_data = imu['time[s]'].to_numpy()/10
h = np.array(h)
end = time.time() - start
print("Ужасно долгое время работы = ", end)

fig1 = plt.figure(1)
axes1 = fig1.subplots(1, 1)
axes1.plot(phi, label='phi')
axes1.plot(phi_data, label='phi_data')

# Создаем новые оси и бурагозим
fig2 = plt.figure(2)
axes2 = fig2.subplots()
axes2.plot(lmbda, label='lmbd')
axes2.plot(lmbda_data, label='lmbd_data')
axes2.legend()



plt.show()

