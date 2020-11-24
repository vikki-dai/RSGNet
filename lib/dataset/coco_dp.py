# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
from collections import defaultdict
from collections import OrderedDict

import json_tricks as json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset.DensePoseDataset import DensePoseDataset
from dataset.densepose_cocoeval import denseposeCOCOeval
from dataset.densepose_cocoeval_challenge import denseposeCOCOeval as challenge_eval
import pycocotools.mask as mask_util
import cv2
from nms.nms import oks_nms

color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255

logger = logging.getLogger(__name__)


class COCODPDataset(DensePoseDataset):
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''
    def __init__(self, cfg, root, image_set, is_train, image_set_name, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.evalDataDir = cfg.TEST.EVAL_DATA_DIR
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE

        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.image_set_name = image_set_name
        self.PIXEL_MEANS = np.asarray([[[0.485, 0.456, 0.406]]])
        self.STD = np.asarray([[[0.229, 0.224, 0.225]]])
        # if is_train and cfg.DATASET.RPN_AUG:
        #     self.coco = COCO(self._get_ann_file_aug_densepose())
        # else:
        self.coco = COCO(self._get_ann_file_densepose())


        # deal with class names
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls],
                                             self._class_to_ind[cls])
                                            for cls in self.classes[1:]])

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))

        self.num_joints = 17
        self.num_surfaces = 24
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.parent_ids = None

        self.db = self._get_db()

        # if is_train and cfg.DATASET.SELECT_DATA:
        #     self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_ann_file_aug_densepose(self):
        """ self.root / annotations / person_keypoints_train2017.json """
        prefix = 'densepose' \
            if 'test' not in self.image_set else 'image_info'
        return os.path.join(self.root, 'annotations',
                            prefix + '_coco_rpn_aug_' + self.image_set_name + '.json')
    def _get_ann_file_densepose(self):
        """ self.root / annotations / person_keypoints_train2017.json """
        prefix = 'densepose' \
            if 'test' not in self.image_set else 'image_info'
        return os.path.join(self.root, 'annotations',
                            prefix + '_coco_' + self.image_set_name + '.json')

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        if self.is_train or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_dp_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_dp_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_dp_annotation_kernal(index))
        return gt_db

    def _load_coco_dp_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id', u'num_keypoints',
                   u'dp_masks', u'dp_I', u'dp_U', u'dp_V', u'dp_x', u'dp_y']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if isinstance(obj['segmentation'], list):
                # Valid polygons have >= 3 points, so require >= 6 coordinates
                obj['segmentation'] = [
                    p for p in obj['segmentation'] if len(p) >= 6
                ]
            if 'ignore' in obj and obj['ignore'] == 1:
                continue

            if 'iscrowd' in obj and obj['iscrowd'] == 1:
                continue
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            if 'dp_masks' not in obj:
                continue
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                # obj['clean_bbox'] = [x1, y1, x2, y2]
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)

        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]

            if cls != 1:
                continue

            # ignore objs without dp annotation
            # if 'dp_masks' not in obj.keys():
            #     continue
            # if len(obj['dp_x'])==0:
            #     continue
            # keypoints
            # joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            # joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            # for ipt in range(self.num_joints):
            #     joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
            #     joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
            #     joints_3d[ipt, 2] = 0
            #     t_vis = obj['keypoints'][ipt * 3 + 2]
            #     if t_vis > 1:
            #         t_vis = 1
            #     joints_3d_vis[ipt, 0] = t_vis
            #     joints_3d_vis[ipt, 1] = t_vis
            #     joints_3d_vis[ipt, 2] = 0
            # person full masks
            valid_segm_polys = obj['segmentation']
            proposal = None
            # person landmarks
            if 'dp_masks' in obj.keys():
                GT_I = np.array(obj['dp_I'])
                GT_U = np.array(obj['dp_U'])
                GT_V = np.array(obj['dp_V'])
                GT_x = np.array(obj['dp_x'])
                GT_y = np.array(obj['dp_y'])
                dp_mask = obj['dp_masks']
                dp_tag = True
                if 'proposal' in obj.keys():
                    proposal = np.round(np.asarray(obj['proposal'][:4])).astype((np.float32))
            else:
                GT_I = np.array([])
                GT_U = np.array([])
                GT_V = np.array([])
                GT_x = np.array([])
                GT_y = np.array([])
                dp_mask = []
                dp_tag = False
            # bbox
            bbox = np.round(np.asarray(obj['clean_bbox'][:4])).astype((np.float32))
            center, scale = self._box2cs(obj['clean_bbox'][:4])
            rec.append({
                'image': self.image_path_from_index(index),
                'bbox': bbox,
                'center': center,
                'scale': scale,
                'full_masks':valid_segm_polys,
                # 'joints_3d': joints_3d,
                # 'joints_3d_vis': joints_3d_vis,
                'dp_I': GT_I,
                'dp_U': GT_U,
                'dp_V': GT_V,
                'dp_x': GT_x,
                'dp_y': GT_y,
                'dp_tag':dp_tag,
                'dp_masks': dp_mask,
                'filename': '',
                'imgnum': 0,
                'proposal': proposal,
            })

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            # scale = scale * 1.25
            scale = scale * 1.

        return center, scale

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = '%012d.jpg' % index
        if '2014' in self.image_set:
            file_name = 'COCO_%s_' % self.image_set + file_name

        prefix = 'test2017' if 'test' in self.image_set else self.image_set

        data_name = prefix + '.zip@' if self.data_format == 'zip' else prefix

        image_path = os.path.join(
            self.root, 'images', data_name, file_name)

        return image_path

    def _load_coco_person_detection_results(self):
        all_boxes = None
        coco_im_ids = set(self.coco.getImgIds())
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        logger.info('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['image_id'] not in coco_im_ids:
                continue
            if det_res['category_id'] != 1:
                continue
            img_name = self.image_path_from_index(det_res['image_id'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue
            if box[2] < 10 or box[3] < 10:
                 continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            box = np.asarray(box,dtype=np.int32)
            kpt_db.append({
                'image': img_name,
                'center': center,
                'scale': scale,
                'score': score,
                'bbox': box
            })

        logger.info('=> Total boxes after fliter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db

    # need double check this API and classes field
    def evaluate(self, cfg, preds, output_dir, img_path,
                 *args, **kwargs):
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        bbox_res_file = os.path.join(
            res_folder, 'detection_%s_results.json' % self.image_set)
        segms_res_file = os.path.join(
            res_folder, 'segmentation_%s_results.json' % self.image_set)
        dp_res_file = os.path.join(
            res_folder, 'densepose_%s_results.pkl' % self.image_set)
        # person x (keypoints)
        results = []
        for idx, im_path in enumerate(img_path):
            results.append({
                'bbox': preds['all_boxes'][idx][:4],
                'body_uv': preds['all_bodys'][idx][:3],
                'body_parts':preds['all_bodys'][idx][3],
                'input_image':preds['all_bodys'][idx][4:7],
                'body_mask': preds['all_bodys'][idx][7],
                'body_full_mask':preds['all_bodys'][idx][8],
                'score': preds['all_boxes'][idx][4],
                'image': int(im_path[-16:-4]),
                'image_path': im_path
            })
        # image x person x (keypoints)
        all_results = defaultdict(list)
        for res in results:
            all_results[res['image']].append(res)

        for im_id in all_results.keys():
            res = all_results[im_id]
            count = 0
            for det in res:
                uv_dets = det['body_uv']
                part_dets = det['body_parts']
                input_img = det['input_image'].transpose((1,2,0))*self.STD+self.PIXEL_MEANS
                save_dir = os.path.join(output_dir, 'vis', str(im_id) + '_%d' % count + '_im.jpg')
                cv2.imwrite(save_dir, input_img)
                body_mask = det['body_mask']
                cv2.imwrite(os.path.join(output_dir, 'vis', str(im_id) + '_%d' % count + '_mask.jpg'),body_mask*255)
                body_full_mask = det['body_full_mask']
                cv2.imwrite(os.path.join(output_dir, 'vis', str(im_id) + '_%d' % count + '_full_mask.jpg'), body_full_mask * 255)
                vis_im = np.zeros_like(uv_dets).transpose((1,2,0))
                for surface in range(1,25):
                    vis_im[uv_dets[0]==surface, :] = color_list[surface]
                save_dir = os.path.join(output_dir,'vis',str(im_id)+'_%d'%count+'_uv.jpg')
                cv2.imwrite(save_dir,vis_im)
                vis_im = np.zeros_like(uv_dets).transpose((1, 2, 0))
                for surface in range(1, 15):
                    vis_im[part_dets == surface, :] = color_list[surface]
                save_dir = os.path.join(output_dir, 'vis', str(im_id) + '_%d' % count + '_part.jpg')
                cv2.imwrite(save_dir, vis_im)

        for im_id in all_results.keys():
            res = all_results[im_id]
            for det in res:
                im_path = det['image_path']
                ori_image = cv2.imread(im_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                im_h, im_w = ori_image.shape[:2]
                box = det['bbox'].astype(np.int32)
                full_mask = det['body_full_mask']
                im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
                im_mask[box[1]:box[1]+full_mask.shape[0], box[0]:box[0]+full_mask.shape[1]] = full_mask
                rle = mask_util.encode(
                    np.array(im_mask[:, :, np.newaxis], order='F')
                )[0]
                rle['counts'] = rle['counts'].decode('utf-8')
                det['body_full_mask'] = rle


        # write full segms results
        # self._write_segms_bbox_results_file(all_results, segms_res_file)
        # write bbox results
        # self._write_coco_bbox_results_file(all_results, bbox_res_file)
        # write densepose results
        self._write_coco_body_uv_results_file(all_results, dp_res_file)
        if 'test' not in self.image_set:
            # info_str = self._do_segms_eval(segms_res_file, res_folder)
            # info_str = self._do_detection_eval(bbox_res_file, res_folder)
            # bbox_name_value = OrderedDict(info_str)
            info_str = self._do_body_uv_eval(dp_res_file, res_folder)
            dp_name_value = OrderedDict(info_str)
            return dp_name_value, dp_name_value['AP']
        else:
            return {'Null': 0}, 0

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [{'cat_id': self._class_to_coco_ind[cls],
                      'cls_ind': cls_ind,
                      'cls': cls,
                      'ann_type': 'keypoints',
                      'keypoints': keypoints
                      }
                     for cls_ind, cls in enumerate(self.classes) if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> Writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.num_joints * 3), dtype=np.float)

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [{'image_id': img_kpts[k]['image'],
                       'category_id': cat_id,
                       'keypoints': list(key_points[k]),
                       'score': img_kpts[k]['score'],
                       'center': list(img_kpts[k]['center']),
                       'scale': list(img_kpts[k]['scale'])
                       } for k in range(len(img_kpts))]
            cat_results.extend(result)

        return cat_results

    def _do_segms_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = denseposeCOCOeval(self.coco, coco_dt, 'segm')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        eval_file = os.path.join(
            res_folder, 'segmentation_%s_results.pkl' % self.image_set)

        with open(eval_file, 'wb') as f:
            pickle.dump(coco_eval, f, pickle.HIGHEST_PROTOCOL)
        logger.info('=> coco eval results saved to %s' % eval_file)

        return info_str

    def _write_segms_bbox_results_file(self, all_results, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]

        data_pack = [{'cat_id': self._class_to_coco_ind[cls],
                      'cls_ind': cls_ind,
                      'cls': cls,
                      'ann_type': 'bbox',
                      'results': all_results
                      }
                     for cls_ind, cls in enumerate(self.classes) if not cls == '__background__']

        results = self._coco_segms_results_one_category(data_pack[0])
        logger.info('=> Writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_segms_results_one_category(self, data_pack):
        cat_id = data_pack['cat_id']
        all_results = data_pack['results']
        cat_results = []

        for img in all_results.keys():
            im_results = all_results[img]
            for det in im_results:
                cat_results.extend(
                    [{'image_id': img,
                      'segmentation': det['body_full_mask'],
                      'category_id': cat_id,
                      'bbox': [det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3]],
                      'score': det['score']}]
                )
        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        eval_file = os.path.join(
            res_folder, 'keypoints_%s_results.pkl' % self.image_set)

        with open(eval_file, 'wb') as f:
            pickle.dump(coco_eval, f, pickle.HIGHEST_PROTOCOL)
        logger.info('=> coco eval results saved to %s' % eval_file)

        return info_str
    def _write_coco_bbox_results_file(self, all_results, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]

        data_pack = [{'cat_id': self._class_to_coco_ind[cls],
                      'cls_ind': cls_ind,
                      'cls': cls,
                      'ann_type': 'bbox',
                      'results': all_results
                      }
                     for cls_ind, cls in enumerate(self.classes) if not cls == '__background__']

        results = self._coco_bbox_results_one_category(data_pack[0])
        logger.info('=> Writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_bbox_results_one_category(self, data_pack):
        cat_id = data_pack['cat_id']
        all_results = data_pack['results']
        cat_results = []

        for img in all_results.keys():
            im_results = all_results[img]
            for det in im_results:
                cat_results.extend(
                    [{'image_id': img,
                      'category_id': cat_id,
                      'bbox': [det['bbox'][0],det['bbox'][1],det['bbox'][2],det['bbox'][3]],
                      'score': det['score']}]
                )

        return cat_results

    def _do_detection_eval(self, res_file, output_dir):
        coco_dt = self.coco.loadRes(str(res_file))

        coco_eval = denseposeCOCOeval(self.coco, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
        info_str = []
        self._log_detection_eval_metrics(coco_eval)
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))
        eval_file = os.path.join(output_dir, 'detection_results.pkl')
        self.save_object(coco_eval, eval_file)
        logger.info('Wrote json eval results to: {}'.format(eval_file))
        return info_str


    def _log_detection_eval_metrics(self, coco_eval):
        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95
        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        logger.info(
            '~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] ~~~~'.format(
                IoU_lo_thresh, IoU_hi_thresh))
        logger.info('{:.1f}'.format(100 * ap_default))
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][
                        ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            logger.info('{:.1f}'.format(100 * ap))
        logger.info('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()

    def _write_coco_body_uv_results_file(self, all_results, res_file):

        data_pack = [{'cat_id': self._class_to_coco_ind[cls],
                      'cls_ind': cls_ind,
                      'cls': cls,
                      'ann_type': 'body_uv',
                      'results': all_results
                      }
                     for cls_ind, cls in enumerate(self.classes) if not cls == '__background__']

        results = self._coco_body_uv_results_one_category(data_pack[0])
        logger.info('=> Writing results pkl to %s' % res_file)
        with open(res_file, 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


    def _coco_body_uv_results_one_category(self, data_pack):
        cat_id = data_pack['cat_id']
        all_results = data_pack['results']
        cat_results = []

        for img in all_results.keys():
            im_results = all_results[img]
            for det in im_results:

                uv_dets = det['body_uv']
                part_dets = det['body_parts']
                uv_dets[1:3, :, :] = uv_dets[1:3, :, :] * 255
                cat_results.extend(
                    [{'image_id': img,
                      'category_id': cat_id,
                      'bbox': [float(det['bbox'][0]),float(det['bbox'][1]),float(det['bbox'][2]),float(det['bbox'][3])],
                      'uv': uv_dets,
                      'parts':part_dets,
                      'score': det['score']}]
                )

        return cat_results

    def _do_body_uv_eval(self, res_file, output_dir):

        ann_type = 'uv'
        imgIds = self.coco.getImgIds()
        imgIds.sort()
        if res_file.endswith('.json'):
            res = res_file
        else:
            with open(res_file, 'rb') as f:
                res = pickle.load(f, encoding='iso-8859-1')
                # res = pickle.load(f)
        coco_dt = self.coco.loadRes(res)
        test_sigma = 0.255
        coco_eval = denseposeCOCOeval(self.coco, coco_dt, ann_type, test_sigma)
        # coco_eval = challenge_eval(self.coco, coco_dt, ann_type, test_sigma, self.evalDataDir)
        coco_eval.params.imgIds = imgIds
        # print('evaluate IOU')
        # coco_eval.evaluate('IOU')
        # coco_eval.accumulate()
        # coco_eval.summarize()
        print('evaluate GPS')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str
    def save_object(self, obj, file_name):
        """Save a Python object by pickling it."""
        file_name = os.path.abspath(file_name)
        with open(file_name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
