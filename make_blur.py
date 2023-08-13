import os
import numpy as np
from PIL import Image
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
# parser.add_argument('--input_dir', default = './input/2011_09_26_drive_0035_sync/image02/data')
# parser.add_argument('--output_dir', default = './output/2011_09_26_drive_0035_sync/')
# parser.add_argument('--result_dir', default = './result/2011_09_26_drive_0035_sync/')
parser.add_argument('--exp', default=5, type=int)
parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')

args = parser.parse_args()

def read_images_from_directory(directory):
    image_list = []
    if not os.path.exists(directory):
        os.makedirs(directory)
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image_path = os.path.join(directory, filename)
            try:
                image = Image.open(image_path).convert('RGB')
                
                image = np.array(image)
                image_list.append(image)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return image_list


def composite_blur(images):
    
    result = np.zeros_like(images[0] ,dtype=float)
    for i, img in enumerate(images): 
            result += img

    return result   

def make_noise(image):
    
    gaussian_noise = np.random.normal(0, 1, image.shape )
    image += gaussian_noise
    image = np.clip(image, 0, 255)
    
    return image





try:
    try:
        try:
            from model.RIFE_HDv2 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v2.x HD model.")
        except:
            from train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v3.x HD model.")
    except:
        from model.RIFE_HD import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v1.x HD model")
except:
    from model.RIFE import Model
    model = Model()
    model.load_model(args.modelDir, -1)
    print("Loaded ArXiv-RIFE model")
model.eval()
model.device()

input_dir = './input'
input_list = [os.path.join(input_dir, f, "image_03/data").replace('\\', '/') for f in os.listdir(input_dir)]
for image_dir in input_list:
    # image_lists = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)]
    image_lists = [os.path.join(image_dir, f).replace('\\', '/') for f in os.listdir(image_dir)]

    for k in range(1, len(image_lists)):
        img = [image_lists[k-1], image_lists[k]]

        if img[0].endswith('.exr') and img[1].endswith('.exr'):
            img0 = cv2.imread(img[0], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            img1 = cv2.imread(img[1], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device)).unsqueeze(0)
            img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device)).unsqueeze(0)

        else:

            img0 = cv2.imread(img[0], cv2.IMREAD_UNCHANGED)
            img1 = cv2.imread(img[1], cv2.IMREAD_UNCHANGED)
            img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
            img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

        n, c, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)


        if args.ratio:
            img_list = [img0]
            img0_ratio = 0.0
            img1_ratio = 1.0
            if args.ratio <= img0_ratio + args.rthreshold / 2:
                middle = img0
            elif args.ratio >= img1_ratio - args.rthreshold / 2:
                middle = img1
            else:
                tmp_img0 = img0
                tmp_img1 = img1
                for inference_cycle in range(args.rmaxcycles):
                    middle = model.inference(tmp_img0, tmp_img1)
                    middle_ratio = ( img0_ratio + img1_ratio ) / 2
                    if args.ratio - (args.rthreshold / 2) <= middle_ratio <= args.ratio + (args.rthreshold / 2):
                        break
                    if args.ratio > middle_ratio:
                        tmp_img0 = middle
                        img0_ratio = middle_ratio
                    else:
                        tmp_img1 = middle
                        img1_ratio = middle_ratio
            img_list.append(middle)
            img_list.append(img1)
        else:
            img_list = [img0, img1]
            for i in range(args.exp):
                tmp = []
                for j in range(len(img_list) - 1):
                    mid = model.inference(img_list[j], img_list[j + 1])
                    tmp.append(img_list[j])
                    tmp.append(mid)
                tmp.append(img1)
                img_list = tmp
                
                
        output_directory = image_dir.replace('input', 'output').split('/')[:-2]
        output_directory = '/'.join(output_directory)
        # output_directory = args.output_dir

        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        for i in range(len(img_list)):
            if img[0].endswith('.exr') and img[1].endswith('.exr'):
                cv2.imwrite('output/img{}.exr'.format(i), (img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
            else:
                cv2.imwrite(os.path.join(output_directory,'img{}.png'.format(i).replace('\\', '/') ), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])





        images = read_images_from_directory(output_directory)
        print(images)







        result = composite_blur(images)

        result /= len(images)





        # result = make_noise(result)

        result = np.clip(result, 0, 255)
        result_normalized = (result - np.min(result)) * (255.0 / (np.max(result) - np.min(result)))
        result_normalized = result_normalized.astype(np.uint8)

        # Save the normalized 'result' array as an image
        save_result = Image.fromarray(result_normalized)


        result_directory = image_dir.replace('input', 'result').split('/')[:-1]
        result_directory = ('/').join(result_directory)
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)

        file_path =  os.path.join(result_directory, '{:010d}.jpg'.format(k))  # Replace with the desired file path
        save_result.save(file_path)
        print(f"save_{result_directory.split('/')[2]}_{k} image")


