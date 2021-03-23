from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import json

from data import cfg
from layers.functions.prior_box import PriorBox, PriorBox_sar, PriorBox_sar_old
from utils.nms_wrapper import nms, soft_nms
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.faceboxes import FaceBoxes
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
parser.add_argument('--prefix', default='weights/lr_1e3_plain', type=str, help='File path to save results')
parser.add_argument('-nl', '--non_local', action="store_true", default=False, help=' ')
parser.add_argument('-se', '--se', action="store_true", default=False, help=' ')
parser.add_argument('-cbam', '--cbam', action="store_true", default=False, help=' ')
parser.add_argument('-gcb', '--gcb', action="store_true", default=False, help=' ')
parser.add_argument('-cda', '--coordatt', action="store_true", default=False, help=' ')
parser.add_argument('-x', '--xception', action="store_true", default=False, help=' ')
parser.add_argument('-mb', '--mobile', action="store_true", default=False, help=' ')
parser.add_argument('-mbv1', '--mobilev1', action="store_true", default=False, help=' ')
parser.add_argument('-shf', '--shuffle', action="store_true", default=False, help=' ')
parser.add_argument('-dcr', '--dsc_crelu', action="store_true", default=False, help=' ')

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

    prefix = args.prefix
    # prefix = 'weights/tmp'
    # prefix = 'weights/lr_1e3_plain'
    # prefix = 'weights/at_4e3'
    # prefix = 'weights/at1_4e3_01'
    # prefix = 'weights/at1_4e3_05'
    # prefix = 'weights/at1_mh_4e3_1'
    # prefix = 'weights/at1_mh_4e3_01'  # sigma 0.2
    # prefix = 'weights/at1_mh_4e3_01_5125vggbn'  # sigma 0.2
    # prefix = 'weights/at1_mh_4e3_01_sigma1'
    # prefix = 'weights/at1_mh_4e3_1_ce_sigma1'
    # prefix = 'weights/at1_mh_4e3_1_ce_sigma02'
    # prefix = 'weights/at1_mh2_4e3_1'
    # prefix = 'weights/at2_mh_4e3_03'
    # prefix = 'weights/at2_mh_4e3_01'
    # prefix = 'weights/at2_4e3_03'
    # prefix = 'weights/at2_4e3_01'
    save_folder = os.path.join(args.save_folder, prefix.split('/')[-1])
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    from utils.logger import Logger
    import sys
    sys.stdout = Logger(os.path.join(save_folder, 'eval.txt'))

    # args.dsc_crelu=True
    dsc_crelu = args.dsc_crelu
    # args.shuffle=True
    shuffle = args.shuffle
    # args.mobile=True
    mobile = args.mobile
    # args.mobilev1=True
    mobilev1 = args.mobilev1
    # args.xception=True
    xception = args.xception

    # args.coordatt=True
    coordatt = args.coordatt
    # args.gcb=True
    gcb = args.gcb
    # args.cbam=True
    cbam = args.cbam
    # args.se=True
    se = args.se
    # args.non_local=True
    non_local = args.non_local
    if non_local:
        from models_light.faceboxes_xception_mbv2_nonlocal import FaceBoxes
    if se:
        from models_light.faceboxes_xception_mbv2_se import FaceBoxes
    if cbam:
        from models_light.faceboxes_xception_mbv2_cbam import FaceBoxes
    if gcb:
        from models_light.faceboxes_xception_mbv2_gcb import FaceBoxes
    if coordatt:
        from models_light.faceboxes_xception_mbv2_coordatt import FaceBoxes

    if xception:
        from models_light.faceboxes_xception import FaceBoxes
    if mobile:
        from models_light.faceboxes_xception_mbv2 import FaceBoxes
    if mobilev1:
        from models_light.faceboxes_xception_mbv1 import FaceBoxes
    if shuffle:
        from models_light.faceboxes_xception_shfv2 import FaceBoxes
    if dsc_crelu:
        from models_light.faceboxes_xception_DscCRelu import FaceBoxes


    # args.cpu = True
    # args.show_image = True
    args.nms_threshold = 0.6  # nms
    # args.nms_threshold = 0.45  # snms
    # args.trained_model = 'weights/rfe/Final_FaceBoxes.pth'
    MEANS = (98.13131, 98.13131, 98.13131)

    # save file
    # fw = open(os.path.join(args.save_folder, args.dataset + '_dets.txt'), 'w')

    # testing dataset
    testset_folder = os.path.join('data', 'SSDD', args.dataset, 'images/')
    testset_list = os.path.join('data', 'SSDD', args.dataset, 'img_list.txt')
    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    # testing scale
    target_size = 1024
    # if args.dataset == "FDDB":
    #     resize = 3
    # elif args.dataset == "PASCAL":
    #     resize = 2.5
    # elif args.dataset == "AFW":
    #     resize = 1
    # elif args.dataset == "SSDD_test":
    #     resize = 2

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # coco eval 
    dataset_name = 'dummy_dataset'
    DatasetCatalog.register(dataset_name, lambda: load_sar_ship_instances('data/SSDD/SSDD_test', ['ship',]))
    MetadataCatalog.get(dataset_name).set(thing_classes=['ship',])
    evaluator = COCOEvaluator(dataset_name, tasks=['bbox'], distributed=False, output_dir=args.save_folder)
    
    dataset_dicts = load_sar_ship_instances('data/SSDD/SSDD_test', ['ship',])
    sar_metadata = MetadataCatalog.get("dummy_dataset")
    
    torch.set_grad_enabled(False)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    # net and model
    net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
    # net = FaceBoxes_sar(phase='test', size=None, num_classes=2) 

    ap_stats = {"ap":[], "ap50": [], "ap75": [], "ap_small": [], "ap_medium": [], "ap_large": [], "epoch": []}

    start_epoch = 10; step = 10
    start_epoch = 200; step = 5
    ToBeTested = []
    ToBeTested = [prefix + f'/FaceBoxes_epoch_{epoch}.pth' for epoch in range(start_epoch, 300, step)]
    ToBeTested.append(prefix + '/Final_FaceBoxes.pth') # 68.5
    # ToBeTested.append(prefix + '/FaceBoxes_epoch_220.pth') # 68.5
    # ToBeTested = [prefix + f'FaceBoxes_epoch_{epoch}.pth' for epoch in range(start_epoch, 300, step)]
    # ToBeTested.append(prefix + 'Final_FaceBoxes.pth') # 68.5
    # ToBeTested.append(prefix + 'Final_FaceBoxes.pth') # 68.5
    for index, model_path in enumerate(ToBeTested):
        args.trained_model = model_path
        net = load_model(net, args.trained_model, args.cpu)
        net.eval()
        print('Finished loading model!')
        # print(net)
        net = net.to(device)

        ap_stats['epoch'].append(start_epoch + index * step)
        print("evaluating epoch: {}".format(ap_stats['epoch'][-1]))

        evaluator.reset()

        # testing begin
        for i, d in enumerate(dataset_dicts):
            im_name = d['image_id']
            image_path = testset_folder + im_name + '.jpg'
            im_raw = cv2.imread(image_path, -1)
            im_draw = im_raw.astype(np.uint8)
            
            h, w, _ = im_raw.shape
            scale = torch.Tensor([w, h, w, h])
            scale = scale.to(device)
            im = np.float32(im_raw)
            im = cv2.resize(im, None, None, fx=target_size/w, fy=target_size/h, interpolation=cv2.INTER_LINEAR)
            im_height, im_width, _ = im.shape
            im -= MEANS
            im = im.transpose(2, 0, 1)
            im = torch.from_numpy(im).unsqueeze(0)
            im = im.to(device)

            _t['forward_pass'].tic()
            loc, conf = net(im)  # forward pass
            _t['forward_pass'].toc()
            _t['misc'].tic()
            # priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priorbox = PriorBox_sar_old(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale
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
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            keep = soft_nms(dets, sigma=0.5, Nt=args.nms_threshold, threshold=args.confidence_threshold, method=1)  # higher performance
            dets = dets[keep, :]

            # keep top-K faster NMS
            dets = dets[:args.keep_top_k, :]
            _t['misc'].toc()

            # save dets
            for k in range(dets.shape[0]):
                xmin = dets[k, 0]
                ymin = dets[k, 1]
                xmax = dets[k, 2]
                ymax = dets[k, 3]
                ymin += 0.2 * (ymax - ymin + 1)
                score = dets[k, 4]
                # fw.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(im_name, score, xmin, ymin, xmax, ymax))
            # print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

            # coco eval 
            inputs = {"image_id": im_name}
            h, w, _ = im_raw.shape
            outputs = Instances((h, w))
            outputs.pred_boxes = Boxes(dets[:, :4])
            classes = [0 for _ in dets]
            classes = torch.tensor(classes, dtype=torch.int64)
            outputs.pred_classes = classes
            outputs.scores = torch.tensor(dets[:, 4])
            evaluator.process([inputs], [{'instances': outputs}])

            # show image
            if args.show_image:
                visualizer = Visualizer(im_draw, metadata=sar_metadata, scale=1)
                out = visualizer.draw_dataset_dict(d)
                im_gt = out.get_image()
                for b in dets:
                    if b[4] < args.vis_thres:
                        continue
                    text = "{:.4f}".format(b[4])
                    b = list(map(int, b))
                    cv2.rectangle(im_gt, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cx = b[0]
                    cy = b[1] + 12
                    cv2.putText(im_gt, text, (cx, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                cv2.imshow('res', im_gt)
                cv2.waitKey(0)

        # fw.close()

        # coco eval
        results = evaluator.evaluate()
        for task, res in results.items():
            # Don't print "AP-category" metrics since they are usually not tracked.
            important_res = [(k, v) for k, v in res.items() if "-" not in k]
            print("copypaste: Task: {}".format(task))
            print("copypaste: " + ",".join([k[0] for k in important_res]))
            print("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))
        print("\nFPS: {}".format(1 / _t['forward_pass'].average_time))
        
        ap_stats['ap'].append(results['bbox']['AP'])
        ap_stats['ap50'].append(results['bbox']['AP50'])
        ap_stats['ap75'].append(results['bbox']['AP75'])
        ap_stats['ap_small'].append(results['bbox']['APs'])
        ap_stats['ap_medium'].append(results['bbox']['APm'])
        ap_stats['ap_large'].append(results['bbox']['APl'])

    # print the best model.
    max_idx = np.argmax(np.asarray(ap_stats['ap50']))
    print('Best ap50: {:.4f} at epoch {}'.format(ap_stats['ap50'][max_idx], ap_stats['epoch'][max_idx]))
    print('ap: {:.4f}, ap50: {:.4f}, ap75: {:.4f}, ap_s: {:.4f}, ap_m: {:.4f}, ap_l: {:.4f}'.\
        format(ap_stats['ap'][max_idx], ap_stats['ap50'][max_idx], ap_stats['ap75'][max_idx], ap_stats['ap_small'][max_idx], ap_stats['ap_medium'][max_idx], ap_stats['ap_large'][max_idx]))
    max_idx = np.argmax(np.asarray(ap_stats['ap']))
    print('Best ap  : {:.4f} at epoch {}'.format(ap_stats['ap'][max_idx], ap_stats['epoch'][max_idx]))
    print('ap: {:.4f}, ap50: {:.4f}, ap75: {:.4f}, ap_s: {:.4f}, ap_m: {:.4f}, ap_l: {:.4f}'.\
        format(ap_stats['ap'][max_idx], ap_stats['ap50'][max_idx], ap_stats['ap75'][max_idx], ap_stats['ap_small'][max_idx], ap_stats['ap_medium'][max_idx], ap_stats['ap_large'][max_idx]))
    res_file = os.path.join(save_folder, 'ap_stats.json')
    print('Writing ap stats json to {}'.format(res_file))
    with open(res_file, 'w') as fid:
        json.dump(ap_stats, fid)
    
    from plot_curve import plot_map, plot_loss
    metrics = ['ap', 'ap50', 'ap_small', 'ap_medium', 'ap_large']
    legend  = ['ap', 'ap50', 'ap_small', 'ap_medium', 'ap_large']
    plot_map(save_folder, ap_stats, metrics, legend)

    txt_log = prefix + '/log.txt'
    plot_loss(save_folder, txt_log)
