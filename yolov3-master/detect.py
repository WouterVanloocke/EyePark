import argparse
import time
from sys import platform
from models import *
from utils.datasets import *
from utils.utils import *
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, Region
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
import os
import cv2
from pathlib import Path
import shutil
import stat
import csv
import io
import ast

ENDPOINT = "https://westeurope.api.cognitive.microsoft.com"
# Plaats hier de keys van je subscription
training_key = "f0dda4df7d7d4cf7a929e471a8aacbe3"
prediction_key = "085fb3c741924d09927420ad56be35d4"
prediction_resource_id = "/subscriptions/3d325e01-eb54-447a-b415-75dc9f70c03f/resourceGroups/customVision/providers/Microsoft.CognitiveServices/accounts/customVision_prediction"

# Plaats hier de juiste naam van je iteratie
publish_iteration_name = "Iteration1"

trainer = CustomVisionTrainingClient(training_key, endpoint=ENDPOINT)

# Zoek het object detection domain
obj_detection_domain = next(domain for domain in trainer.get_domains() if domain.type == "ObjectDetection" and domain.name == "General")

# Zet hier de key van je project
project = trainer.get_project("97181c71-c821-4f8f-8571-3d7ff56ff334")

def detect(cfg,
           data,
           weights,
           images='data/samples',  # input folder
           output='output',  # output folder
           fourcc='mp4v',  # video codec
           img_size=416,
           conf_thres=0.8,
           nms_thres=0.8,
           save_txt=False,
           save_images=True):
    # Initialize
    device = torch_utils.select_device(force_cpu=ONNX_EXPORT)
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    if ONNX_EXPORT:
        s = (320, 192)  # (320, 192) or (416, 256) or (608, 352) onnx model image size (height, width)
        model = Darknet(cfg, s)
    else:
        model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3, s[0], s[1]))
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Half precision
    opt.half = opt.half and device.type != 'cpu'  # half precision only supported on CUDA
    if opt.half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if opt.webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size, half=opt.half)
    else:
        dataloader = LoadImages(images, img_size=img_size, half=opt.half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    auto_tag = next(filter(lambda t:t.name == "auto", trainer.get_tags(project.id)), None)
    # Run inference
    t0 = time.time()
    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        t = time.time()
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred, _ = model(img)
        det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0]
        #

        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results to screen
            print('%gx%g ' % img.shape[2:], end='')  # print image size
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')
            with open('C:/Studentenjob/yolov3-master/coordinaten.txt', 'w+') as file:
                file.write("" + auto_tag.id  + ": [")
            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in det:
                #if save_txt:  # Write to file
                if int(cls) == 2:  
                    x1 = float(('{0}').format(*xyxy,))/1920
                    y1 = float(('{1}').format(*xyxy,))/1080
                    x2 = float(('{2}').format(*xyxy,))/1920
                    y2 = float(('{3}').format(*xyxy,))/1080
                    breedte = (x2 -x1)
                    hoogte = (y2-y1)

                    with open('C:/Studentenjob/yolov3-master/coordinaten.txt', 'a') as file:
                        file.write("(\"" + os.path.basename(path) + "\"" + ',[ ' + str(x1)+ ', ' + str(y1) + ', ' + str(breedte) + ', ' + str(hoogte) + ' ]),')                        
    
            with open('C:/Studentenjob/yolov3-master/coordinaten.txt', 'rb+') as file:
                file.seek(-1,os.SEEK_END)
                file.truncate()
            with open('C:/Studentenjob/yolov3-master/coordinaten.txt', 'a') as file:    
                file.write("]}")
            base_image_url = Path("C:/Studentenjob/yolov3-master/output/")

            # Go through the data table above and create the images
            print ("Adding images...")
            tagged_images_with_regions = []
            f = open('C:/Studentenjob/yolov3-master/coordinaten.txt', "r")
            auto_image_regions = {}
            for line in f.readlines():
                # Now we split the file on `x`, since the part before the x will be
                # the key and the part after the value
                line = line.split(':')
                # Take the line parts and strip out the spaces, assigning them to the variables
                # Once you get a bit more comfortable, this works as well:
                # key, value = [x.strip() for x in line] 
                key = line[0].strip()
                value = line[1].strip()
                # Now we check if the dictionary contains the key; if so, append the new value,
                # and if not, make a new list that contains the current value
                # (For future reference, this is a great place for a defaultdict :)
                #if key in auto_image_regions:
                auto_image_regions[key] = [value]
                # else:
                #     auto_image_regions[key] = [value]
                print(auto_image_regions)
            
            #lijst = f.readlines()

            #auto_image_regions = {}
            
            # auto_image_regions = {
            #     auto_tag.id : [
            #         lines
            #     ]
            # }              
                
            print(auto_image_regions)
            for tag_id in auto_image_regions:
                for filename in auto_image_regions[tag_id]:
                    x,y,w,h = auto_image_regions[filename]
                    regions = [ Region(tag_id=auto_tag.id, left=x,top=y,width=w,height=h) ]
                    
                    with open(str(base_image_url) + "/" + filename, mode="rb") as image_contents:
                        tagged_images_with_regions.append(ImageFileCreateEntry(name=filename, contents=image_contents.read(), regions=regions))

            upload_result = trainer.create_images_from_files(project.id, images=tagged_images_with_regions)

            if not upload_result.is_batch_successful:
                print("Image batch upload failed.")
                for image in upload_result.images:
                    print("Image status: ", image.status)
                exit(-1)
        
                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        print('Done. (%.3fs)' % (time.time() - t))

        if opt.webcam:  # Show live webcam
            cv2.imshow(weights, im0)

        if save_images:  # Save image with detections
            if dataloader.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
                vid_writer.write(im0)

    if save_images:
        print('Results saved to %s' % os.getcwd() + os.sep + output)
        if platform == 'darwin':  # macos
            os.system('open ' + output + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='fourcc output video codec (verify ffmpeg support)')
    parser.add_argument('--output', type=str, default='output', help='specifies the output path for images and videos')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--webcam', action='store_true', help='use webcam')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt.cfg,
               opt.data,
               opt.weights,
               images=opt.images,
               img_size=opt.img_size,
               conf_thres=opt.conf_thres,
               nms_thres=opt.nms_thres,
               fourcc=opt.fourcc,
               output=opt.output)
