import numpy as np
from matplotlib.backends.qt_compat import QtWidgets, QtCore
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import torch
from PyQt5.QtWidgets import QApplication
import sys
import mano

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, num_pose_params=9, num_shape_params=10, num_transl_params=3):
        super().__init__()
        self.num_pose_params = num_pose_params
        self.num_shape_params = num_shape_params
        self.num_transl_params = num_transl_params

        # Create layout
        layout = QtWidgets.QVBoxLayout()
        sliders = []
        for i in range(num_pose_params + num_shape_params + num_transl_params):
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setMaximum(100)
            slider.setMinimum(-100)
            
            slider.setSingleStep(5)
            slider.valueChanged.connect(self.update)  # Connect slider to update function
            sliders.append(slider)
        
        # use_pca = True if num_pose_params < 48 else False
        self.mano_layer = mano.load(model_path='../mano_v1_2/models',
                     is_rhand=True,
                     num_pca_comps=num_pose_params-3,
                     batch_size=1,
                     flat_hand_mean=True
                     )

        self.faces = self.mano_layer.faces.astype(np.int32)#[np.newaxis, ...]
        self.pose = torch.zeros(1, self.num_pose_params)
        self.shape = torch.zeros(1, self.num_shape_params)
        self.transl = torch.zeros(1, self.num_transl_params)
        output = self.mano_layer(betas=self.shape,
                  global_orient=self.pose[:, :3],
                  hand_pose=self.pose[:, 3:],
                  transl=self.transl,
                  return_verts=True,
                  return_tips=True)
        
        self.init_mesh = output.vertices[0].detach().numpy()
        # Create figure canvas
        self.figure = Figure(figsize=(8, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111, projection='3d')

        # Store 3D mesh data
        self.X = self.init_mesh[:, 0]
        self.Y = self.init_mesh[:, 1]
        self.Z = self.init_mesh[:, 2]

        x_range = self.X.max() - self.X.min()
        y_range = self.Y.max() - self.Y.min()
        z_range = self.Z.max() - self.Z.min()
        max_range = max(x_range, y_range, z_range)
        self.axes.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])

        self.axes.plot_trisurf(self.X, self.Y, self.Z, triangles=self.faces, cmap='coolwarm')

        # Add sliders and canvas to layout
        slider_layout = QtWidgets.QGridLayout()
        slider_layout.setSpacing(10) 
        for i in range(num_pose_params + num_shape_params + num_transl_params):
            if i < 3:
                text = "rotation"
            elif i < num_pose_params:
                text = "pose"
            elif i >= num_pose_params and i < num_pose_params + num_shape_params:
                text = "shape"
            else:
                text = "translation"
            slider_layout.addWidget(QtWidgets.QLabel("{} Parameter {}".format(text, i+1)), (i * 2) // 8, (i * 2) % 8)
            slider_layout.addWidget(sliders[i], (i*2) // 8, (i*2) % 8 + 1)

        layout.addLayout(slider_layout)
        layout.addWidget(self.canvas)

        # Set layout
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Store sliders in a list for easy access
        self.sliders = sliders


    def update(self):
        # Get slider values
        values = [slider.value() / 100 for slider in self.sliders]

        # Update 3D mesh data
        pose = torch.tensor(values[:self.num_pose_params]).unsqueeze(0)
        shape = self.shape if self.num_shape_params == 0 else torch.tensor(values[self.num_pose_params:self.num_pose_params+self.num_shape_params]).unsqueeze(0)
        transl = self.transl if self.num_transl_params == 0 else torch.tensor(values[self.num_pose_params+self.num_shape_params:]).unsqueeze(0)
        output = self.mano_layer(betas=shape,
                  global_orient=pose[:, :3],
                  hand_pose=pose[:, 3:],
                  transl=transl,
                  return_verts=True,
                  return_tips=True)
        mesh = output.vertices[0].detach().numpy()
        
        self.X, self.Y, self.Z = mesh[:, 0], mesh[:, 1], mesh[:, 2]

        # Update plot
        self.axes.clear()
        x_range = self.X.max() - self.X.min()
        y_range = self.Y.max() - self.Y.min()
        z_range = self.Z.max() - self.Z.min()
        max_range = max(x_range, y_range, z_range)
        self.axes.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])
        self.axes.plot_trisurf(self.X, self.Y, self.Z, triangles=self.faces, cmap='coolwarm')
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    num_pose_params = int(sys.argv[1])
    main_window = MainWindow(num_pose_params=num_pose_params + 3, num_shape_params=10, num_transl_params=3)    
    main_window.show()
    sys.exit(app.exec_())