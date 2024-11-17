import sys
import numpy as np
import torchvision
from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import torch
from torchvision import transforms
from data_aug.view_generator import ContrastiveLearningViewGenerator
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_simclr_pipeline_transform(size, s=1):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])
    return data_transforms

class Sim_model:
    def __init__(self):
        self.model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
        self.checkpoint_path = None

    def load_checkpoint(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = checkpoint['state_dict']

        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    state_dict[k[len("backbone."):]] = state_dict[k]

            elif k.startswith('res_backbone.'):
                if k.startswith('res_backbone') and not k.startswith('res_backbone.fc'):
                    state_dict[k[len("res_backbone."):]] = state_dict[k]

            del state_dict[k]

        log = self.model.load_state_dict(state_dict, strict=False)
        assert log.missing_keys == ['fc.weight', 'fc.bias']

        # set eval
        self.model.eval()

    def predict(self, image):
        return self.model(image)

    def convert(self, numpy_array):
        return torch.from_numpy(numpy_array).float().permute(2,0,1).unsqueeze(0).to(device)

    def compute_similarity(self, image1, image2):
        image1 = self.convert(image1)
        image2 = self.convert(image2)
        feature1 = self.predict(image1)
        feature2 = self.predict(image2)

        norm1 = torch.norm(feature1)
        norm2 = torch.norm(feature2)
        sim = torch.matmul(feature1, feature2.T) / norm1 / norm2
        sim = sim.flatten()
        return sim.detach().cpu().numpy()


class ImageWindow(QWidget):
    def __init__(self, sim_model):
        super().__init__()
        self.setWindowTitle('Visualize')
        self.setFixedSize(1000, 800)
        self.initUI()

        # Augmentation
        self.Aug_Generator = ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(32), n_views=2)
        self.sim_model = sim_model

    def initUI(self):
        self.layout = QHBoxLayout()

        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        self.button_layout = QHBoxLayout()

        self.upload_button = QPushButton('Upload Image')
        self.upload_button.clicked.connect(self.openimage)
        self.upload_button.setFixedSize(165, 40)

        self.checkpoint_button = QPushButton('Upload Checkpoint')
        self.checkpoint_button.clicked.connect(self.upload_checkpoint)
        self.checkpoint_button.setFixedSize(165, 40)

        self.sim_label = QLabel(self)
        self.sim_label.setGeometry(50, 50, 280, 50)
        self.sim_label.move(120, 250)
        self.sim_label.setStyleSheet("QLabel{font-size:25px;font-weight:bold}")
        self.set_sim(None)


        self.label = QLabel(self)
        self.set_label_area(self.label, "Origin")

        self.augmented_label1 = QLabel(self)
        self.set_label_area(self.augmented_label1, "Augmented Image1")

        self.augmented_label2 = QLabel(self)
        self.set_label_area(self.augmented_label2, "Augmented Image2")

        self.status_label = QLabel(self)  # Add status label
        self.status_label.setGeometry(50, 100, 200, 50)
        self.status_label.move(170, 200)

        self.button_layout.addWidget(self.upload_button)
        self.button_layout.addWidget(self.checkpoint_button)

        self.left_layout.addLayout(self.button_layout)
        # self.left_layout.addWidget(self.sim_label)
        self.left_layout.addWidget(self.label)

        # self.augmented_label1 = QLabel('Augmented Image 1')
        # self.augmented_label2 = QLabel('Augmented Image 2')

        self.right_layout.addWidget(self.augmented_label1)
        self.right_layout.addWidget(self.augmented_label2)

        self.layout.addLayout(self.left_layout)
        self.layout.addLayout(self.right_layout)

        self.setLayout(self.layout)

    def set_label_area(self, label, name):
        label.setText(name)
        label.setFixedSize(420, 320)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("QLabel{background:white;font-size:28px;}")

    def set_sim(self, sim):
        if sim:
            self.sim_label.setText("Similarity is {:.2f}%".format(100*sim[0]))
        else:
            self.sim_label.setText("Similarity is  ")


    def show_img_InUI(self, imgName, label):
        img = QtGui.QPixmap(imgName).scaled(label.width(), label.height())
        if img:
            label.setPixmap(img)
            return True
        return False


    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "Upload Image", "", "*.jpg;;*.png;;All Files(*)")
        success = self.show_img_InUI(imgName, self.label)
        if success:
            self.update_imgs(imgName)

    def upload_checkpoint(self):
        checkpoint_path, _ = QFileDialog.getOpenFileName(self, "Upload Checkpoint", "", "*.pth;;*.tar;;All Files(*)")
        if checkpoint_path:
            self.sim_model.load_checkpoint(checkpoint_path)
            self.status_label.setText("Load success!")
        else:
            self.status_label.setText("Load failed!")


    def update_imgs(self, imgName):
        print(imgName)
        try:
            original_image = Image.open(imgName)
        except:
            print("cannot find image")

        augmented_images = self.augment_image(original_image)


        self.show_image(augmented_images[0], self.augmented_label1)
        self.show_image(augmented_images[1], self.augmented_label2)

        similarity = self.sim_model.compute_similarity(augmented_images[0],augmented_images[1])
        self.set_sim(similarity)

        # self.show_image(augmented_images[0], self.augmented_label1)
        # self.show_image(augmented_images[1], self.augmented_label2)


    def augment_image(self, image):
        augmented_images = self.Aug_Generator(image)
        return [np.array(transforms.ToPILImage()(img)) for img in augmented_images]

    def show_image(self, image, label):
        qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        qimage = qimage.scaled(420, 320, Qt.IgnoreAspectRatio)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap)

if __name__ == '__main__':
    SIMmodel = Sim_model()
    SIMmodel.load_checkpoint("runs/Nov15_b256_e100/checkpoint_0100.pth.tar")
    app = QApplication(sys.argv)
    window = ImageWindow(SIMmodel)
    window.show()
    sys.exit(app.exec_())
