import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def vis(C):
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.set_xlabel('Столбец')
    ax1.set_ylabel('Строка')
    kku = ax1.imshow(C)
    fig.colorbar(kku, ax=ax1)

    ax2 = fig.add_subplot(132, projection='3d')
    X, Y = np.meshgrid(np.linspace(1, len(C), len(C)), np.linspace(1, len(C), len(C)))
    ax2.plot_surface(X, Y, C, cmap=plt.cm.YlGnBu_r)
    ax2.set_zlim(min([min(i) for i in C]), max([max(i) for i in C]))
    ax2.set_xlabel('Столбец')
    ax2.set_ylabel('Строка')
    ax2.set_zlabel('Значение')

    ax3 = fig.add_subplot(133)
    ax3.scatter(C[0:], C[::-1], alpha=0.5)
    ax3.grid(True)

    ax1.set_title('Тепловая карта')
    ax2.set_title('Объёмная тепловая карта')
    ax3.set_title('Разброс значений')
    fig.suptitle('Matplotlib')

    fig.set_figwidth(14)
    fig.set_figheight(5)
    plt.subplots_adjust(wspace=0.4)
    plt.show()

    fig2 = plt.figure()
    ax = fig2.add_subplot(131)
    kkh = sns.ecdfplot(data=pd.DataFrame(np.transpose(C)), palette="deep", ax=ax)

    ax2 = fig2.add_subplot(132)
    sns.set_theme(style="ticks")
    sns.despine(fig2)
    kku = sns.histplot(C, palette="deep", edgecolor=".3", linewidth=.5, ax=ax2)

    ax3=fig2.add_subplot(133)
    kkd = sns.lineplot(data=np.transpose(C), palette="deep", linewidth=2.5, ax=ax3)

    fig2.suptitle('Seaborn')
    kkh.set_title('Градация значений по строкам')
    kku.set_title('Количество значений в диапазонах\nпо строкам')
    kkd.set_title('График значений по строкам')
    fig2.set_figwidth(15)
    fig2.set_figheight(6)
    plt.subplots_adjust(wspace=0.4)
    plt.show()


try:
    n = int(input("Введите количество строк и столбцов > 3: "))
    while n < 4:
        n = int(input("Введите количество строк и столбцов > 3: "))
    k = int(input("Введите значение коэффициента k: "))
    A = np.random.randint(-10.0, 10.0, (n, n), dtype='int64')
    F = np.copy(A)
    np.set_printoptions(precision=2, linewidth=200)
    print(f"----A----\n{A}\n\n----F----\n{F}\n\n")
    cond_e, cond_lines = 0, 1
    for i in range(n):
        for j in range(n):
            if i % 2 == 0:
                cond_lines *= int(A[i][j])
            if i > (n // 2 - (n - 1) % 2) and j > (n // 2 - (n - 1) % 2) and j % 2 == 0 and A[i][j] == 0:
                cond_e += 1
    print(f"Количество нулей в нечетных столбцах Е = {cond_e}\nПроизведение чисел в нечетных строках = {cond_lines}\n")
    if cond_e * k > cond_lines:
        for i in range(n // 2):
            F[i] = F[i][::-1]
    else:
        for i in range(n // 2):
            for j in range(n // 2):
                F[i][j], F[n // 2 + n % 2 + i][n // 2 + n % 2 + j] = F[n // 2 + n % 2 + i][n // 2 + n % 2 + j], F[i][j]
    print(f"----F----\n{F}\n")
    if np.linalg.det(A) > sum(np.diagonal(F)):
        if np.linalg.det(F) == 0:
            print("Матрица F - вырожденная, невозможно провести вычисления")
        else:
            print(f"----A*AT – K * F-1----\n{np.matmul(A, np.transpose(A)) - np.linalg.inv(F) * k}")
            vis(np.matmul(A, np.transpose(A)) - np.linalg.inv(F) * k)
    else:
        if np.linalg.det(A) == 0:
            print("Матрица A - вырожденная, невозможно провести вычисления")
        else:
            print(f"----(A-1 +G-FТ)*K----\n{(np.linalg.inv(A) + np.tril(A) - np.transpose(F)) * k}")
            vis((np.linalg.inv(A) + np.tril(A) - np.transpose(F)) * k)
    print(f"Время выполнения: {time.process_time()}")

except ValueError:
    print("Ввод не являются числом")