# coding=utf-8
import dlib
import sys, os, argparse
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

sys.path.append('code/')
import datasets, hopenet, utils

join = os.path.join

class HeadPose:
    def __init__(self):
        cudnn.enabled = True
        batch_size = 1
        self.gpu = 0
        snapshot_path = '/home/xiangmingcan/notespace/deep-head-pose/hopenet_robust_alpha1.pkl'
        input_path = '/home/xiangmingcan/notespace/cvpr_data/celeba/'
        output = 'output/celeba.txt'
        face_model = '/home/xiangmingcan/notespace/deep-head-pose/mmod_human_face_detector.dat'

        out_dir = os.path.split(output)[0]
        name = os.path.split(output)[1]

        write_path = join(out_dir, "images_" + name[:-4])
        if not os.path.exists(write_path):
            os.makedirs(write_path)

        if not os.path.exists(input_path):
            sys.exit('Folder does not exist')

        # ResNet50 structure
        self.model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

        # Dlib face detection model
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(face_model)

        print 'Loading snapshot.'
        # Load snapshot
        saved_state_dict = torch.load(snapshot_path)
        self.model.load_state_dict(saved_state_dict)

        print 'Loading data.'

        self.transformations = transforms.Compose([transforms.Scale(224),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.model.cuda(self.gpu)

        print 'Ready to test network.'

        # Test the Model
        self.model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        total = 0

        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).cuda(self.gpu)

        # -------------- for image operation ------------------
    def estimate(self, image):
        # image 是完整的路径
        image = cv2.imread(image)
        cv2_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Dlib detect
        dets = self.cnn_face_detector(cv2_frame, 1)

        yaw_predicted, pitch_predicted, roll_predicted = None, None, None
        for idx, det in enumerate(dets):
            # Get x_min, y_min, x_max, y_max, conf
            x_min = det.rect.left()
            y_min = det.rect.top()
            x_max = det.rect.right()
            y_max = det.rect.bottom()
            conf = det.confidence

            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)
            x_min -= 2 * bbox_width / 4
            x_max += 2 * bbox_width / 4
            y_min -= 3 * bbox_height / 4
            y_max += bbox_height / 4
            x_min = max(x_min, 0); y_min = max(y_min, 0)
            x_max = min(image.shape[1], x_max); y_max = min(image.shape[0], y_max)
            # Crop image
            img = cv2_frame[y_min:y_max,x_min:x_max]
            img = Image.fromarray(img)

            # Transform
            img = self.transformations(img)
            img_shape = img.size()
            img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
            img = Variable(img).cuda(self.gpu)

            yaw, pitch, roll = self.model(img)

            yaw_predicted = F.softmax(yaw)
            pitch_predicted = F.softmax(pitch)
            roll_predicted = F.softmax(roll)
            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99

            # # Print new frame with cube and axis
            # drawed_img = utils.draw_axis(image, yaw_predicted, pitch_predicted, roll_predicted, tdx =(x_min + x_max) / 2, tdy=(y_min + y_max) / 2, size =bbox_height / 2)

        return [yaw_predicted, pitch_predicted, roll_predicted]


if __name__ == '__main__':
    dataset = sys.argv[1]
    method = sys.argv[2]

    flag = 0
    if dataset == "ffhq":
        flag = 1

    src = "/home/xiangmingcan/notespace/cvpr_data/" + dataset
    tgt = "/home/xiangmingcan/notespace/cvpr_result/" + dataset + '/' + method

    save_log = os.path.join("headPose/", dataset, method + ".txt")

    path = os.path.join("headPose/", dataset)
    if not os.path.exists(path):
        os.makedirs(path)

    logFile = open(save_log, 'w')

    img_list = os.listdir(tgt)
    sorted(img_list)

    headpose = HeadPose()

    for input_img in img_list:
        if '_mask' in input_img:
            continue
        eular_angles_result = headpose.estimate(os.path.join(tgt, input_img))
        print(eular_angles_result)

        # reference image 的欧拉角
        refer_img = input_img.split('-')[1]

        if flag:
            refer_img = refer_img[:-3] + 'png'

        eular_angles_refer = headpose.estimate(os.path.join(src, refer_img))
        print(eular_angles_refer)

        vec1 = np.array(eular_angles_result)
        vec2 = np.array(eular_angles_refer)

        if (None in vec1) or (None in vec2):
            continue

        distance = np.linalg.norm(vec1 - vec2)
        print(distance)
        print('\n')

        logFile.write(str(distance))
        logFile.write('\n')

    logFile.close()