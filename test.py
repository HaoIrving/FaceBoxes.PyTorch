from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox, PriorBox_sar
from utils.nms_wrapper import nms
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.faceboxes import FaceBoxes, FaceBoxes_sar
from utils.box_utils import decode
from utils.timer import Timer

from data import load_sar_ship_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import (Boxes, Instances)
from detectron2.evaluation import COCOEvaluator

from detectron2.utils.visualizer import Visualizer


parser = argparse.ArgumentParser(description='FaceBoxes')

parser.add_argument('-m', '--trained_model', default='weights/Final_FaceBoxes.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset', default='SSDD_test', type=str, choices=['AFW', 'PASCAL', 'FDDB', 'SAR_SHIP_test'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--show_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    # args.cpu = True
    # args.show_image = True
    args.nms_threshold = 0.3
    args.vis_thres = 0.1
    # args.trained_model = 'weights/rfe/Final_FaceBoxes.pth'

    torch.set_grad_enabled(False)
    # net and model
    net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
    # net = FaceBoxes_sar(phase='test', size=None, num_classes=2) 
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)


    # save file
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    fw = open(os.path.join(args.save_folder, args.dataset + '_dets.txt'), 'w')

    # testing dataset
    testset_folder = os.path.join('data', 'SSDD', args.dataset, 'images/')
    testset_list = os.path.join('data', 'SSDD', args.dataset, 'img_list.txt')
    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    # testing scale
    if args.dataset == "FDDB":
        resize = 3
    elif args.dataset == "PASCAL":
        resize = 2.5
    elif args.dataset == "AFW":
        resize = 1
    elif args.dataset == "SSDD_test":
        resize = 1.5

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # coco eval 
    dataset_name = 'dummy_dataset'
    DatasetCatalog.register(dataset_name, lambda: load_sar_ship_instances('data/SSDD/SSDD_test', ['ship',]))
    MetadataCatalog.get(dataset_name).set(thing_classes=['ship',])
    evaluator = COCOEvaluator(dataset_name, distributed=False, output_dir=args.save_folder)
    evaluator.reset()
    
    dataset_dicts = load_sar_ship_instances('data/SSDD/SSDD_test', ['ship',])
    sar_metadata = MetadataCatalog.get("dummy_dataset")

    # testing begin
    for i, d in enumerate(dataset_dicts):
        img_name = d['image_id']
        image_path = testset_folder + img_name + '.jpg'
        img_raw = cv2.imread(image_path)
        
        # pixel_max = img_raw.max()
        # # # pixel_min = img.min()
        # k = pixel_max ** (1 / 255)
        # img_raw = np.clip(img_raw, 1, None)
        # img_raw = np.log(img_raw) / np.log(k)

        # img_raw = img_raw[:, :, np.newaxis]
        # img_raw = np.concatenate((img_raw, img_raw, img_raw), axis=2)

        img_draw = img_raw.astype(np.uint8)
        img = np.float32(img_raw)

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (98.13131, 98.13131, 98.13131)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        # priorbox = PriorBox_sar(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        #keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        _t['misc'].toc()

        # save dets
        if args.dataset == "FDDB":
            fw.write('{:s}\n'.format(img_name))
            fw.write('{:.1f}\n'.format(dets.shape[0]))
            for k in range(dets.shape[0]):
                xmin = dets[k, 0]
                ymin = dets[k, 1]
                xmax = dets[k, 2]
                ymax = dets[k, 3]
                score = dets[k, 4]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                fw.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'.format(xmin, ymin, w, h, score))
        else:
            for k in range(dets.shape[0]):
                xmin = dets[k, 0]
                ymin = dets[k, 1]
                xmax = dets[k, 2]
                ymax = dets[k, 3]
                ymin += 0.2 * (ymax - ymin + 1)
                score = dets[k, 4]
                fw.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(img_name, score, xmin, ymin, xmax, ymax))
        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

        # coco eval 
        inputs = {"image_id": img_name}
        h, w, _ = img_raw.shape
        outputs = Instances((h, w))
        outputs.pred_boxes = Boxes(dets[:, :4])
        classes = [0 for _ in dets]
        classes = torch.tensor(classes, dtype=torch.int64)
        outputs.pred_classes = classes
        outputs.scores = torch.tensor(dets[:, 4])
        evaluator.process([inputs], [{'instances': outputs}])

        # show image
        if args.show_image:
            visualizer = Visualizer(img_draw, metadata=sar_metadata, scale=1)
            out = visualizer.draw_dataset_dict(d)
            img_gt = out.get_image()
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_gt, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_gt, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.imshow('res', img_gt)
            cv2.waitKey(0)

    fw.close()

    # coco eval
    results = evaluator.evaluate()
    for task, res in results.items():
        # Don't print "AP-category" metrics since they are usually not tracked.
        important_res = [(k, v) for k, v in res.items() if "-" not in k]
        print("copypaste: Task: {}".format(task))
        print("copypaste: " + ",".join([k[0] for k in important_res]))
        print("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))
    print("AP50: {}, FPS: {}".format(results['bbox']['AP50'], 1 / _t['forward_pass'].average_time))