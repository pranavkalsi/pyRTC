import sys
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from pyRTC.Pipeline import ImageSHM
from pyRTC.utils import *
import os
import math


class RealTimeView(QMainWindow):
    def __init__(self, shm_names, fps, static_vmin=None, static_vmax=None):
        super().__init__()

        self.shm_names = shm_names
        self.n = len(shm_names)

        self.static_vmin = static_vmin
        self.static_vmax = static_vmax

        self.shm = []
        self.metadata = []
        self.im = []
        self.cbar = []
        self.fpsText = []
        self.old_count = []
        self.old_time = []
        self.linear_norm = []

        self.figure = Figure(figsize=(8, 8), tight_layout=True)

        ncols = math.ceil(math.sqrt(self.n))
        nrows = math.ceil(self.n / ncols)

        for i, shm_name in enumerate(self.shm_names):
            self._init_subplot(i, shm_name, nrows, ncols)

        self.canvas = FigureCanvas(self.figure)

        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central_widget)

        self.setWindowTitle("PyRTC Multi-SHM Viewer")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_view)
        self.timer.start(1000 // fps)

    def _init_subplot(self, i, shm_name, nrows, ncols):
        metadata = ImageSHM(
            shm_name + "_meta",
            (ImageSHM.METADATA_SIZE,),
            np.float64
        )
        meta = metadata.read_noblock()

        width = max(1, int(meta[4]))
        height = max(1, int(meta[5]))
        dtype = float_to_dtype(meta[3])

        shm = ImageSHM(shm_name, (width, height), dtype)
        frame = shm.read_noblock()

        ax = self.figure.add_subplot(nrows, ncols, i + 1)
        ax.set_title(shm_name)

        vmin, vmax = np.min(frame), np.max(frame)
        if self.static_vmin is not None:
            vmin = self.static_vmin
        if self.static_vmax is not None:
            vmax = self.static_vmax

        im = ax.imshow(
            frame,
            cmap="inferno",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest"
        )

        cbar = self.figure.colorbar(im, ax=ax)

        fps_text = ax.text(
            width // 2,
            int(1.15 * height),
            "PAUSED",
            fontsize=12,
            ha="center",
            va="bottom",
            color="g"
        )

        self.metadata.append(metadata)
        self.shm.append(shm)
        self.im.append(im)
        self.cbar.append(cbar)
        self.fpsText.append(fps_text)
        self.old_count.append(0)
        self.old_time.append(0)
        self.linear_norm.append(im.norm)

    def update_view(self):
        redraw = False

        for i in range(self.n):
            frame = self.shm[i].read_noblock()
            meta = self.metadata[i].read_noblock()

            new_count = meta[0]
            new_time = meta[1]

            if new_time > self.old_time[i]:
                fps = (new_count - self.old_count[i]) / (new_time - self.old_time[i])
                fps_str = f"{fps:.2f} FPS"
            else:
                fps_str = "PAUSED"

            self.old_count[i] = new_count
            self.old_time[i] = new_time

            if isinstance(frame, np.ndarray):
                vmin, vmax = np.min(frame), np.max(frame)

                if self.static_vmin is not None:
                    vmin = self.static_vmin
                if self.static_vmax is not None:
                    vmax = self.static_vmax

                self.im[i].set_data(frame)
                self.im[i].set_clim(vmin, vmax)
                self.cbar[i].update_normal(self.im[i])
                self.fpsText[i].set_text(fps_str)

                redraw = True

        if redraw:
            self.canvas.draw_idle()

    def closeEvent(self, event):
        for shm in self.shm:
            shm.close()
        event.accept()


if __name__ == "__main__":
    set_affinity(0)

    app = QApplication(sys.argv)

    if len(sys.argv) < 2:
        print("Usage: python pyRTCView.py shm1 [shm2 shm3 ...]")
        sys.exit(1)

    shm_names = sys.argv[1:]

    view = RealTimeView(
        shm_names,
        fps=30,
        static_vmin=None,
        static_vmax=None
    )

    view.show()
    sys.exit(app.exec_())
