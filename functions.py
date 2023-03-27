import numpy as np
import pandas as pd


def gravity(phi, h, a) -> np.float:
    """
        Модельное значение ускореня силы тяжести с учетом его зависимости от широты и высоты
        Формула Гельмерта с поправкой
    """
    c2 = 0.005302   # beta1
    c3 = 0.000007   # beta2
    c4 = 0.00014    # delta g
    ge = 9.78030
    # np.divide(2.0, 4.0)= 0.5 - деление
    return np.float(ge * (1 + c2 * (np.sin(phi))**2 - c3 * (np.sin(2 * phi))**2 - 2 * np.divide(h, a)) - c4)


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
    return theta, psi, gamma


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

    r_e = a / np.sqrt(1 - e**2 * np.sin(phi)**2)
    r_n = a * (1 - e**2) / (np.sqrt(1 - e**2 * np.sin(phi)**2))**3
    return r_e, r_n


