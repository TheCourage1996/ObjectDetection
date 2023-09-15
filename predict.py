from model import PSPNet
import torch
from PIL import Image
from  dataloader import VOCDataset
from torchvision import transforms
import torch.nn.functional as F
import os
import numpy as np
weight_path = r'D:\1Apython\Pycharm_pojie\Semantic_segmentation\PSPNet\weights\checkpoint--epoch4.pth'
img_file = r'D:\1Apython\Pycharm_pojie\data_set\VOCdevkit\VOC2007\JPEGImages\2007_000027.jpg'
output = r'./weights'
model = PSPNet(num_classes=21)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(weight_path)
model.load_state_dict(checkpoint['state_dict'])
MEAN = [0.45734706, 0.43338275, 0.40058118] # 数据集的均值和方差
STD = [0.23965294, 0.23532275, 0.2398498]
normalize = transforms.Normalize(MEAN, STD) #拿出训练时候dataloader 里面的设置
to_tensor = transforms.ToTensor()
model.to(device)
model.eval()

palette = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                    (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                    (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                        (128, 64, 12)]

def save_images(image, mask, output_path, image_file, palette,num_classes):
	# Saves the image, the model output and the results after the post processing
    w, h = image.size
    image_file = os.path.basename(image_file).split('.')[0] # basenmae,返回图片名字
    colorized_mask = cam_mask(mask,palette,num_classes)
    colorized_mask.save(os.path.join(output_path, image_file+'.png'))

def cam_mask(mask,palette,n):
    seg_img = np.zeros((np.shape(mask)[0], np.shape(mask)[1], 3))
    for c in range(n):
        seg_img[:, :, 0] += ((mask[:, :] == c) * (palette[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((mask[:, :] == c) * (palette[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((mask[:, :] == c) * (palette[c][2])).astype('uint8')
    colorized_mask = Image.fromarray(np.uint8(seg_img))
    return colorized_mask

with torch.no_grad():
    image = Image.open(img_file).convert('RGB')
    input = normalize(to_tensor(image)).unsqueeze(0)
    prediction = model(input.to(device))
    prediction = prediction.squeeze(0).cpu().numpy()
    prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
    # =================================================
    save_images(image, prediction, output, img_file, palette,  num_classes=21)
