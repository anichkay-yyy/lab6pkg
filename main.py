#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

# -----------------------
# Геометрия буквы "V" (Воронович) как каркас, экструзия 2D-контур -> 3D призма
# -----------------------

def build_letter_V(depth: float = 1.0, scale_xy: float = 1.0):
    """
    Возвращает список вершин (Nx4, однородные координаты) и список рёбер (индексы).
    2D контур буквы "V" задан в плоскости z=0, затем экструзия вдоль z в обе стороны.
    """
    contour2d = np.array([
        [-1.0,  2.0],
        [ 0.0, -2.0],
        [ 1.0,  2.0],
        [ 0.6,  2.0],
        [ 0.0, -0.8],
        [-0.6,  2.0],
    ]) * scale_xy

    z_front = -depth / 2
    z_back  =  depth / 2

    verts_front = np.hstack([contour2d, np.full((len(contour2d), 1), z_front)])
    verts_back  = np.hstack([contour2d, np.full((len(contour2d), 1), z_back)])

    verts = np.vstack([verts_front, verts_back])
    ones = np.ones((verts.shape[0], 1))
    verts_h = np.hstack([verts, ones])  # однородные

    edges = []
    n = len(contour2d)
    # контуры
    for i in range(n):
        edges.append((i, (i + 1) % n))         # фронт
        edges.append((i + n, (i + 1) % n + n)) # тыл
        edges.append((i, i + n))               # боковые
    return verts_h, edges

# -----------------------
# Преобразования
# -----------------------

def translation_matrix(dx, dy, dz):
    M = np.eye(4)
    M[:3, 3] = [dx, dy, dz]
    return M

def scaling_matrix(sx, sy, sz):
    M = np.eye(4)
    M[0, 0], M[1, 1], M[2, 2] = sx, sy, sz
    return M

def rotation_matrix(axis, angle_deg):
    """
    Вращение вокруг произвольной оси (проходящей через начало координат).
    axis: iterable (ux, uy, uz)
    angle_deg: угол в градусах
    """
    ux, uy, uz = axis
    v = np.array([ux, uy, uz], dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.eye(4)
    ux, uy, uz = v / norm
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    C = 1 - c
    R = np.array([
        [ux*ux*C + c,    ux*uy*C - uz*s, ux*uz*C + uy*s],
        [uy*ux*C + uz*s, uy*uy*C + c,    uy*uz*C - ux*s],
        [uz*ux*C - uy*s, uz*uy*C + ux*s, uz*uz*C + c   ]
    ])
    M = np.eye(4)
    M[:3, :3] = R
    return M

def apply_transform(verts_h, M):
    return (M @ verts_h.T).T

# -----------------------
# Проекции
# -----------------------

def ortho_proj_Oxy(verts_h):
    return verts_h[:, 0], verts_h[:, 1]

def ortho_proj_Oxz(verts_h):
    return verts_h[:, 0], verts_h[:, 2]

def ortho_proj_Oyz(verts_h):
    return verts_h[:, 1], verts_h[:, 2]

# -----------------------
# Рисование
# -----------------------

def plot_scene(ax3d, ax_xy, ax_xz, ax_yz, verts_h, edges):
    ax3d.cla(); ax_xy.cla(); ax_xz.cla(); ax_yz.cla()

    # 3D оси
    ax3d.set_title("3D каркас")
    ax3d.set_box_aspect([1, 1, 1])
    for e in edges:
        p, q = verts_h[e[0], :3], verts_h[e[1], :3]
        ax3d.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color='blue')
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.grid(True, linestyle=':')

    # Оси проекций
    ax_xy.set_title("Ортографическая проекция на Oxy")
    ax_xy.axhline(0, color='gray', linewidth=0.5)
    ax_xy.axvline(0, color='gray', linewidth=0.5)
    ax_xy.set_aspect('equal', adjustable='box')

    ax_xz.set_title("Ортографическая проекция на Oxz")
    ax_xz.axhline(0, color='gray', linewidth=0.5)
    ax_xz.axvline(0, color='gray', linewidth=0.5)
    ax_xz.set_aspect('equal', adjustable='box')

    ax_yz.set_title("Ортографическая проекция на Oyz")
    ax_yz.axhline(0, color='gray', linewidth=0.5)
    ax_yz.axvline(0, color='gray', linewidth=0.5)
    ax_yz.set_aspect('equal', adjustable='box')

    # Рёбра проекций
    px, py = ortho_proj_Oxy(verts_h)
    for e in edges:
        i, j = e
        ax_xy.plot([px[i], px[j]], [py[i], py[j]], color='green')
    px, pz = ortho_proj_Oxz(verts_h)
    for e in edges:
        i, j = e
        ax_xz.plot([px[i], px[j]], [pz[i], pz[j]], color='red')
    py, pz = ortho_proj_Oyz(verts_h)
    for e in edges:
        i, j = e
        ax_yz.plot([py[i], py[j]], [pz[i], pz[j]], color='purple')

# -----------------------
# GUI
# -----------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ЛР 6a/6b/6c — 3D каркас буквы V (Воронович)")
        self.geometry("1200x800")

        self.base_verts, self.edges = build_letter_V(depth=1.0, scale_xy=1.0)
        self.verts = self.base_verts.copy()
        self.M_total = np.eye(4)

        self._build_controls()
        self._build_plot()
        self.update_plot()

    def _build_controls(self):
        frame = ttk.Frame(self)
        frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        ttk.Label(frame, text="Масштабирование (sx, sy, sz):").pack(anchor='w')
        self.sx = tk.DoubleVar(value=1.0)
        self.sy = tk.DoubleVar(value=1.0)
        self.sz = tk.DoubleVar(value=1.0)
        self._triple_entry(frame, self.sx, self.sy, self.sz)

        ttk.Label(frame, text="Перенос (dx, dy, dz):").pack(anchor='w', pady=(8, 0))
        self.dx = tk.DoubleVar(value=0.0)
        self.dy = tk.DoubleVar(value=0.0)
        self.dz = tk.DoubleVar(value=0.0)
        self._triple_entry(frame, self.dx, self.dy, self.dz)

        ttk.Label(frame, text="Вращение: угол (°) и ось (ux, uy, uz):").pack(anchor='w', pady=(8, 0))
        self.angle = tk.DoubleVar(value=0.0)
        self.ux = tk.DoubleVar(value=0.0)
        self.uy = tk.DoubleVar(value=0.0)
        self.uz = tk.DoubleVar(value=1.0)
        self._single_entry(frame, self.angle, label="Угол, °")
        self._triple_entry(frame, self.ux, self.uy, self.uz)

        ttk.Button(frame, text="Применить все", command=self.apply_all).pack(fill=tk.X, pady=8)
        ttk.Button(frame, text="Сброс", command=self.reset).pack(fill=tk.X)

        ttk.Label(frame, text="Итоговая матрица (4x4):").pack(anchor='w', pady=(12, 4))
        self.matrix_text = tk.Text(frame, width=35, height=8, font=("Courier", 10))
        self.matrix_text.pack(fill=tk.X)

    def _triple_entry(self, parent, v1, v2, v3):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)
        ttk.Entry(row, textvariable=v1, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Entry(row, textvariable=v2, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Entry(row, textvariable=v3, width=8).pack(side=tk.LEFT, padx=2)

    def _single_entry(self, parent, var, label=None):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)
        if label:
            ttk.Label(row, text=label, width=8).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var, width=10).pack(side=tk.LEFT, padx=2)

    def _build_plot(self):
        fig = plt.Figure(figsize=(9, 7), dpi=100)
        gs = fig.add_gridspec(2, 2)
        self.ax3d = fig.add_subplot(gs[:, 0], projection='3d')
        self.ax_xy = fig.add_subplot(gs[0, 1])
        self.ax_xz = fig.add_subplot(gs[1, 1])

        # дополнительное окно для Oyz (вписываем в ax_xz через inset)
        self.ax_yz = self.ax_xz.inset_axes([1.05, 0.05, 0.45, 0.45])
        self.ax_yz.set_facecolor("#f8f8f8")

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = canvas

    def update_plot(self):
        plot_scene(self.ax3d, self.ax_xy, self.ax_xz, self.ax_yz, self.verts, self.edges)
        self.canvas.draw()
        self._update_matrix_text()

    def _update_matrix_text(self):
        self.matrix_text.delete("1.0", tk.END)
        mat = self.M_total
        for row in mat:
            self.matrix_text.insert(tk.END, " ".join(f"{v:8.4f}" for v in row) + "\n")

    def apply_all(self):
        try:
            sx, sy, sz = self.sx.get(), self.sy.get(), self.sz.get()
            dx, dy, dz = self.dx.get(), self.dy.get(), self.dz.get()
            angle = self.angle.get()
            ux, uy, uz = self.ux.get(), self.uy.get(), self.uz.get()
        except tk.TclError:
            messagebox.showerror("Ошибка", "Некорректный ввод чисел.")
            return

        Ms = scaling_matrix(sx, sy, sz)
        Mt = translation_matrix(dx, dy, dz)
        Mr = rotation_matrix((ux, uy, uz), angle)

        M = Mt @ Mr @ Ms  # порядок: сначала масштаб, затем вращение, затем перенос
        self.M_total = M @ self.M_total  # накапливаем
        self.verts = apply_transform(self.base_verts, self.M_total)
        self.update_plot()

    def reset(self):
        self.M_total = np.eye(4)
        self.verts = self.base_verts.copy()
        self.update_plot()

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
