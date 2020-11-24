from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch
from config.models import get_model_name
from core.evaluate import accuracy
from core.loss import UDPLosses
# from core.inference import get_final_preds, get_final_preds_with_target_relation_score, get_final_preds_with_offsets, \
#     get_final_preds_with_dark
from core.inference import get_final_preds
# from utils.transforms import flip_back, flip_dp_back, flip_back_offset
from utils.transforms import flip_back, flip_dp_back
from utils.vis import save_debug_images
from torch.nn import functional as F
from collections import defaultdict
import pickle

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.8f} ({loss.avg:.8f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

def train_initial(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    crit = torch.nn.MSELoss().cuda()
    crite = torch.nn.BCELoss().cuda()

    for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, tight_bbox_regression, target_mask, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs, bboxes_regression, relation_scores = model(input)

        target = target.cuda(non_blocking=True)  # ([64,17,64,48])
        target_weight = target_weight.cuda(non_blocking=True)
        target_joints = meta['joints']
        # all_ins_target = all_ins_target.cuda(non_blocking=True)
        # all_ins_target_weight = all_ins_target_weight.cuda(non_blocking=True)
        tight_bbox_regression = tight_bbox_regression.cuda(non_blocking=True)
        target_mask = target_mask.cuda(non_blocking=True)
        # target_num_joints = target_num_joints.cuda(non_blocking=True)


        # 0.5*interference + 1*target
        # if isinstance(outputs, list):
        #     mt_loss = criterion(outputs[0], all_ins_target, all_ins_target_weight)
        #     for output in outputs[1:]:
        #         mt_loss += criterion(output, all_ins_target, all_ins_target_weight)
        # else:
        #     output = outputs
        #     mt_loss = criterion(output, all_ins_target, all_ins_target_weight)

        # only target
        if isinstance(outputs, list):
            mt_loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                mt_loss += criterion(output, target, target_weight)
        else:
            output = outputs
            mt_loss = criterion(output, target, target_weight)


        # mask features similarity loss
        bbox_loss = crit(bboxes_regression, tight_bbox_regression)  # MSELoss
        relation_loss = crite(relation_scores, target_mask.float())  # BCELoss

        loss = 0.01*bbox_loss + mt_loss + 0.001 *relation_loss
        # loss = mt_loss + 0.01 * relation_loss
        # loss = bbox_loss + mt_loss

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            print('bbox_loss', bbox_loss)
            print('mt_loss', mt_loss)
            print('relation_loss', relation_loss)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images_with_bbox(config, input, meta, target, pred*4, output, bboxes_regression,
                              prefix)


def validate_initial(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, tight_bbox_regression, target_mask, meta) in enumerate(val_loader):
            # compute output
            outputs, bboxes_regression, relation_scores = model(input)


            # target tight bbox check
            # target_joints = meta['joints']
            # target_joints_vis = meta['joints_vis']
            # image_width = 192
            # image_height = 256
            # batch_size = target.shape[0]
            # num_joints = target.shape[1]
            # bbox_in = 0
            # bbox_out = 0
            # for k in range(batch_size):  # target:([64,17,64,48]), vis_features:([64,32,64,48]), for one image
            #     gt_joints = target_joints[k]
            #     gt_joints_vis = target_joints_vis[k]
            #     pre_bbox = bboxes_regression[k]
            #     xmin = pre_bbox[0] * image_width - 12
            #     ymin = pre_bbox[1] * image_height - 20
            #     xmax = pre_bbox[0] * image_width + pre_bbox[2] * image_width + 12
            #     ymax = pre_bbox[1] * image_height + pre_bbox[3] * image_height + 20
            #
            #     # boundary check
            #     xmin = xmin if xmin > 0 else 0
            #     xmax = xmax if xmax < image_width else image_width - 1
            #     ymin = ymin if ymin > 0 else 0
            #     ymax = ymax if ymax < image_height else image_height - 1
            #
            #     for i in range(num_joints):
            #         if gt_joints_vis[i][0] == 0:
            #             continue
            #         if gt_joints[i][0] < xmin or gt_joints[i][0] > xmax or gt_joints[i][1] < ymin or gt_joints[i][1] > ymax:
            #             bbox_out = bbox_out + 1
            #         else:
            #             bbox_in = bbox_in + 1
            #
            # print('out/all', bbox_out / (bbox_out + bbox_in))
            # print('in/all', bbox_in / (bbox_out + bbox_in))


            if isinstance(outputs, list):
                # output = outputs[-1]
                output = outputs[0]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped, bboxes_regression_flipped,  relation_scores_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    # output_flipped = outputs_flipped[-1]
                    output_flipped = outputs_flipped[0]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            # test gt_heatmap
            # output = output
            # output += 0.1*target
            # output += 0.3 * target
            # output += 0.5 * target
            # output += 0.7 * target
            # output += 0.9 * target
            # output += 1.0 * target
            # output += 1.1 * target
            # output += 1.2 * target
            # output += 1.3 * target


            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            preds, maxvals = get_final_preds_with_gt(
                config, output.clone().cpu().numpy(), target.clone().cpu().numpy(), c, s)

            # preds, maxvals = get_final_preds_with_bboxes(
            #     config, output.clone().cpu().numpy(), tight_bbox_regression.long().clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals

            # remove interferences
            # all_keypoints = np.squeeze(all_preds)
            # for i in range(17):
            #     if all_keypoints[i][1] < xmin or all_keypoints[i][1] > xmax or all_keypoints[i][2] < ymin or all_keypoints[i][2] > ymax:
            #         all_keypoints[i][1] = 0
            #         all_keypoints[i][2] = 0
            # all_preds[0] = all_keypoints

            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images_with_bbox(config, input, meta, target, pred*4, output, bboxes_regression,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

def cp_train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, inter_target, inter_target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)
        output, inter_output = outputs

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        # inter_target = inter_target.cuda(non_blocking=True)
        # inter_target_weight = inter_target_weight.cuda(non_blocking=True)

        loss = criterion(output, target, target_weight)

        # loss += criterion(inter_output, inter_target, inter_target_weight)
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def cp_validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, inter_target, inter_target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            output, inter_output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)
                output_flipped = outputs_flipped[0]
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        # name_values, perf_indicator = val_dataset.evaluate(
        #     config, all_preds, output_dir, all_boxes, image_path,
        #     filenames, imgnums
        # )

        model_name = config.MODEL.NAME
        # if isinstance(name_values, list):
        #     for name_value in name_values:
        #         _print_name_value(name_value, model_name)
        # else:
        #     _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            # if isinstance(name_values, list):
            #     for name_value in name_values:
            #         writer.add_scalars(
            #             'valid',
            #             dict(name_value),
            #             global_steps
            #         )
            # else:
            #     writer.add_scalars(
            #         'valid',
            #         dict(name_values),
            #         global_steps
            #     )
            writer_dict['valid_global_steps'] = global_steps + 1
    perf_indicator = acc.avg
    return perf_indicator

def generate_points_from_labels(meta):
    image_list = meta['image']
    kpts = defaultdict(list)
    for image in image_list:
        kpts[image].append()

def rl_train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # inter_scores = meta['interference_maps'].cuda(non_blocking=True)
        # cat_maps = meta['kpt_cat_maps'].cuda(non_blocking=True)
        # with torch.no_grad():
        #     amaps_target, amaps_target_weight = generate_association_map_from_gt_heatmaps(target, inter_target)

        # compute output

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        all_ins_target = all_ins_target.cuda(non_blocking=True)
        all_ins_target_weight = all_ins_target_weight.cuda(non_blocking=True)
        # amaps_target = amaps_target.cuda(non_blocking=True)
        # amaps_target_weight = amaps_target_weight.cuda(non_blocking=True)

        outputs = model(input)
        output, inter_output, amap_output = outputs
        # target heatmap loss
        st_loss = criterion(output, target, target_weight)
        # multi-instances heatmap loss
        mt_loss = criterion(inter_output, all_ins_target, all_ins_target_weight)
        # association loss
        # rel_loss_all = criterion(amap_output, amaps_target, amaps_target_weight)
        # for amap, gt_amap, n in zip(amap_output, meta['association_maps'], meta['num_points']):
        #     gt_map = gt_amap[:n, :n].to(amap.device)
        #     rel_loss = F.mse_loss(amap, gt_map)
        #     rel_loss_all += rel_loss
        # rel_loss_all /= len(amap_output)
        loss = st_loss + mt_loss #+ rel_loss_all


        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)
            print('st_pose loss loss:%.6f' % (st_loss.cpu().detach().numpy()),
                  'mt_pose loss:%.6f' % (mt_loss.cpu().detach().numpy()))
            print('*'*100)
                  # 'relation loss:%.6f' % (rel_loss_all.cpu().detach().numpy()))
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)

def generate_association_map_from_gt_heatmaps(targets, all_targets):
    targets = F.interpolate(targets,(28,28),mode="bilinear", align_corners=False).cpu().numpy()
    all_targets = F.interpolate(all_targets, (28, 28), mode="bilinear", align_corners=False).cpu().numpy()
    b,c,w,h = targets.shape
    heatmaps = (np.max(targets, axis=1) >= 0.5).astype(np.float32)
    heatmaps = heatmaps.reshape((-1, w*h))
    AMaps = heatmaps[...,None] * heatmaps[:,None]
    weights = (np.max(all_targets, axis=1) >= 0.5).astype(np.float32)
    weights = weights.reshape((-1, w*h))
    weights = weights[...,None] * weights[:,None]
    return torch.from_numpy(AMaps), torch.from_numpy(weights)

def rl_validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, use_meta = True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, meta) in enumerate(val_loader):
            # compute output
            # with torch.no_grad():
            #     amaps_target, amaps_target_weight = generate_association_map_from_gt_heatmaps(target, inter_target)
            #     gt_all_kpt_scores = torch.max(target, inter_target)
            #     amaps_target = amaps_target.cuda(non_blocking=True)
            #     gt_all_kpt_scores = gt_all_kpt_scores.cuda(non_blocking=True)
            #     flipped_amaps_target, _ = generate_association_map_from_gt_heatmaps(target.flip(3), inter_target.flip(3))
            #     flipped_gt_all_kpt_scores = gt_all_kpt_scores.flip(3)
            #     flipped_gt_all_kpt_scores = flipped_gt_all_kpt_scores.cuda(non_blocking=True)
            #     flipped_amaps_target = flipped_amaps_target.cuda(non_blocking=True)

            # compute output

            # target = target.cuda(non_blocking=True)
            # target_weight = target_weight.cuda(non_blocking=True)
            # inter_target = inter_target.cuda(non_blocking=True)
            # inter_target_weight = inter_target_weight.cuda(non_blocking=True)
            # amaps_target = amaps_target.cuda(non_blocking=True)
            # amaps_target_weight = amaps_target_weight.cuda(non_blocking=True)
            if use_meta:
                outputs = model(input, meta)
            else:
                outputs = model(input)
            output, inter_output, amap_output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                if use_meta:
                    outputs_flipped = model(input_flipped, meta)
                else:
                    outputs_flipped = model(input_flipped)
                output_flipped = outputs_flipped[0]
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator
def dp_train(config, train_loader, model, criterion, optimizer, epoch,
                output_dir, tb_log_dir, writer_dict, lr_scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    lr = AverageMeter()
    lr.update(lr_scheduler.get_lr()[0])
    end = time.time()
    # for i, (input, target_weight, meta) in enumerate(train_loader):
    for i, (input, dp_body_mask_target, dp_body_part_target, target_sf, target_u, target_v, target_weight, target_uv_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        dp_body_mask_target = dp_body_mask_target.cuda(non_blocking=True)
        dp_body_part_target = dp_body_part_target.cuda(non_blocking=True)
        target_sf = target_sf.cuda(non_blocking=True)
        target_u = target_u.cuda(non_blocking=True)
        target_v = target_v.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        target_uv_weight = target_uv_weight.cuda(non_blocking=True)
        #
        # print("input:",input.size())
        # compute output

        body_mask_preds, body_part_preds, surface_preds, u_preds, v_preds = model(input)
        # compute loss
        if isinstance(body_mask_preds, tuple):
            mask_loss = criterion['person_mask'](body_mask_preds[0], dp_body_mask_target)
            part_loss = criterion['part_mask'](body_part_preds[0], dp_body_part_target)
            surface_loss = criterion['surface'](surface_preds[0], target_sf, target_weight)
            u_loss = criterion['dp_u'](u_preds[0], target_u, target_uv_weight)
            v_loss = criterion['dp_v'](v_preds[0], target_v, target_uv_weight)
            loss = 0.5 * mask_loss + 0.5 * part_loss + 1. * surface_loss + 1. * u_loss + 1. * v_loss
            inter_mask_loss = []
            inter_part_loss = []
            inter_surface_loss = []
            inter_u_loss = []
            inter_v_loss = []
            inter_loss = 0
            for n_out in range(len(body_mask_preds)):
                if n_out == 0:
                    continue
                inter_mask_loss.append(criterion['person_mask'](body_mask_preds[n_out], dp_body_mask_target))
                inter_part_loss.append(criterion['part_mask'](body_part_preds[n_out], dp_body_part_target))
                inter_surface_loss.append(criterion['surface'](surface_preds[n_out], target_sf, target_weight))
                inter_u_loss.append(criterion['dp_u'](u_preds[n_out], target_u, target_uv_weight))
                inter_v_loss.append(criterion['dp_v'](v_preds[n_out], target_v, target_uv_weight))
                inter_loss = inter_loss + 0.5 * inter_mask_loss[-1] + 0.5 * inter_part_loss[-1] + 1. * inter_surface_loss[-1] + 1. * inter_u_loss[-1] + 1. * inter_v_loss[-1]
            loss = loss + 0.1 * inter_loss / n_out
        else:
            mask_loss = criterion['person_mask'](body_mask_preds, dp_body_mask_target)
            part_loss = criterion['part_mask'](body_part_preds, dp_body_part_target)
            surface_loss = criterion['surface'](surface_preds, target_sf, target_weight)
            u_loss = criterion['dp_u'](u_preds, target_u, target_uv_weight)
            v_loss = criterion['dp_v'](v_preds, target_v, target_uv_weight)
            loss = 1.*mask_loss + 1.*part_loss + 1*(1.*surface_loss + 1.*u_loss + 1.*v_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:

            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'LRate {lr.val:.6f}\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f}))'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                lr=lr,
                data_time=data_time, loss=losses)
            print('-' * 50)
            logger.info(msg)
            if isinstance(body_mask_preds, tuple):
                inter_loss_str = ''
                for i_l in range(len(inter_mask_loss)):
                    inter_loss_str += 'inter_dp_body_loss_%d: %.6f '%(i_l+1, inter_mask_loss[i_l]) + \
                                       'inter_dp_part_loss_%d: %.6f '%(i_l+1, inter_part_loss[i_l]) + \
                                       'inter_surface_loss_%d: %.6f '%(i_l+1, inter_surface_loss[i_l]) + \
                                       'inter_dp_u_loss_%d: %.6f '%(i_l+1, inter_u_loss[i_l]) + \
                                       'inter_dp_v_loss_%d: %.6f '%(i_l+1, inter_v_loss[i_l])
                print(inter_loss_str)

            print('dp_body loss:%.6f' % (mask_loss.cpu().detach().numpy()),
                  'dp_part loss:%.6f' % (part_loss.cpu().detach().numpy()),
                  # 'full mask loss:%.6f' % (full_mask_loss.cpu().detach().numpy()))
                  'surface_points loss:%.6f' % (surface_loss.cpu().detach().numpy()),
                  'u_points loss:%.6f' % (u_loss.cpu().detach().numpy()),
                  'v_points loss:%.6f' % (v_loss.cpu().detach().numpy()),)
                  # 'dp_i_points loss:%.6f' % (dp_index_uv_loss.cpu().detach().numpy()),
                  # 'dp_u_points loss:%.6f' % (dp_u_loss.cpu().detach().numpy()),
                  # 'dp_v_points loss:%.6f' % (dp_v_loss.cpu().detach().numpy()),)
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)

            # save_debug_dp_images(config, input, target, output, prefix)



def dp_validate(config, val_loader, val_dataset, model, output_dir,
             tb_log_dir, writer_dict=None):
    part_label = ['', 'Torso', 'Torso', 'Right Hand', 'Left Hand', 'Left Foot', 'Right Foot', 'Upper Leg Right',
                  'Upper Leg Left', 'Upper Leg Right', 'Upper Leg Left'
        , 'Lower Leg Right', 'Lower Leg Left', 'Lower Leg Right', 'Lower Leg Left',
                  'Upper Arm Left', 'Upper Arm Right', 'Upper Arm Left', 'Upper Arm Right', 'Lower Arm Left',
                  'Lower Arm Right', 'Lower Arm Left', 'Lower Arm Right', 'Head',
                  'Head']
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # num_samples = len(val_dataset)
    all_preds = {}

    # all_preds = np.zeros((num_samples, 3),
    #                      dtype=np.float32)
    # all_boxes = np.zeros((num_samples, 6))
    # all_bb = np.empty((0,4),dtype=np.float32)
    # cls_segms = [[] for _ in range(2)]
    all_boxes = []
    all_bodys = []
    all_scores = []
    all_inputs = []
    image_path = []
    filenames = []
    imgnums = []
    # idx = 0
    with torch.no_grad():
        end = time.time()
        # for i, (input, dp_body_mask_target, dp_body_part_target, target_sf, target_u, target_v, target_weight, target_uv_weight, meta) in enumerate(val_loader):
        # for i, (input,  target_sf, target_xy, target_uv, target_tag, meta) in enumerate(val_loader):
        for i, (input, meta) in enumerate(val_loader):
            # compute output

            input = input.cuda(non_blocking=True)

            body_mask_preds, body_part_preds,surface_preds, u_preds, v_preds = model(input)

            # test aug
            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                body_mask_preds_flipped, body_part_preds_flipped, surface_preds_flipped, u_preds_flipped, v_preds_flipped = model(input_flipped)
                surface_preds_flipped = flip_dp_back(surface_preds_flipped.cpu().numpy())
                surface_preds_flipped = torch.from_numpy(surface_preds_flipped.copy()).cuda()

                body_mask_preds_flipped = body_mask_preds_flipped.cpu().numpy()[:, :, :, ::-1]
                body_mask_preds_flipped = torch.from_numpy(body_mask_preds_flipped.copy()).cuda()

                body_mask_preds = (body_mask_preds + body_mask_preds_flipped) * 0.5
                surface_preds = (surface_preds + surface_preds_flipped) * 0.5


            body_mask_preds = body_mask_preds.clone().cpu().numpy()
            # full_mask_preds = np.zeros((body_mask_preds.shape[0], 1, body_mask_preds.shape[2], body_mask_preds.shape[3]))

            surface_preds = surface_preds.clone().cpu().numpy()
            u_preds = u_preds.clone().cpu().numpy()
            v_preds = v_preds.clone().cpu().numpy()
            body_part_preds = body_part_preds.clone().cpu().numpy()
            input = input.clone().cpu().numpy()
            # vis_debug
            # for i_res in range(target_sf.shape[0]):
            #     uv = target_sf[i_res]
            #     dp_img = np.zeros((3,uv.shape[0],uv.shape[1]))
            #     surface_idx = uv
            #     for i in range(1, 25):
            #         loc = np.where(surface_idx == i)
            #         if len(loc[0]) > 0:
            #             print(part_label[i])
            #             dp_img[0, loc[0], loc[1]] += i
            #             dp_img[1, loc[0], loc[1]] += u_preds[i_res][i,loc[0], loc[1]]*255
            #             dp_img[2, loc[0], loc[1]] += v_preds[i_res][i,loc[0], loc[1]]*255
            #             cv2.imshow('dp_part', np.transpose(dp_img, (1, 2, 0)))
            #             cv2.waitKey(0)
            '''
            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                output_flipped = model(input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0

                output = (output + output_flipped) * 0.5
            '''

            # measure accuracy and record loss

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            bbox = meta['bbox'].numpy()
            # bbox[:,2] = bbox[:,0] + bbox[:,2] - 1
            # bbox[:,3] = bbox[:,1] + bbox[:,3] - 1
            all_boxes += list(bbox)
            all_scores += list(score)

            all_body = get_final_dp_preds(
                body_mask_preds, surface_preds, u_preds, v_preds, bbox, body_part_preds, input)


            all_bodys += all_body

            # double check this all_boxes parts

            # all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            # all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            # all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            # all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])
            # idx += num_images

            # if i % config.PRINT_FREQ == 0:
            # if i % 32 == 0:
            #     msg = 'Test: [{0}/{1}]\t' \
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
            #           ''.format(
            #               i, len(val_loader), batch_time=batch_time)
            #     logger.info(msg)
            #
            #     prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
            #     save_debug_segms_images(config, input, target, output, prefix)
        all_scores = np.asarray(all_scores).reshape((-1,1))
        all_boxes = np.asarray(all_boxes).reshape((-1,4))
        all_preds['all_boxes'] = np.hstack([all_boxes, all_scores])
        all_preds['all_bodys'] = all_bodys
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, image_path,
            filenames, imgnums)

        _, full_arch_name = get_model_name(config)
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, full_arch_name)
        else:
            _print_name_value(name_values, full_arch_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            # writer.add_scalar('valid_acc', acc.avg, global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid', dict(name_values), global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator
# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )

def train_aaai(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    # crit = torch.nn.MSELoss().cuda()
    # crite = torch.nn.BCELoss().cuda()

    for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, limbs_target, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs, multi_ouptuts = model(input)

        target = target.cuda(non_blocking=True)  # ([64,17,64,48])
        target_weight = target_weight.cuda(non_blocking=True)
        all_ins_target = all_ins_target.cuda(non_blocking=True)
        all_ins_target_weight = all_ins_target_weight.cuda(non_blocking=True)
        #  only target
        if isinstance(outputs, list):
            target_loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                target_loss += criterion(output, target, target_weight)
        else:
            output = outputs
            target_loss = criterion(output, target, target_weight)

        #  0.5*interference + target
        if isinstance(multi_ouptuts, list):
            multi_loss = criterion(multi_ouptuts[0], all_ins_target, all_ins_target_weight)
            for multi_output in multi_ouptuts[1:]:
                multi_loss += criterion(multi_output, all_ins_target, all_ins_target_weight)
        else:
            multi_output = multi_ouptuts
            multi_loss = criterion(multi_output, all_ins_target, all_ins_target_weight)


        loss = target_loss + multi_loss

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            # print('bbox_loss', bbox_loss)
            print('target_loss', target_loss)
            print('multi_loss', multi_loss)
            # print('relation_loss', relation_loss)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate_aaai(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, limbs_target, meta) in enumerate(val_loader):
            # compute output
            outputs, multi_ouptuts = model(input)


            if isinstance(outputs, list):
                # output = outputs[-1]
                output = outputs[0]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped, multi_ouptuts_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    # output_flipped = outputs_flipped[-1]
                    output_flipped = outputs_flipped[0]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals


            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

def train_aaai_refine(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    # crit = torch.nn.MSELoss().cuda()
    crite = torch.nn.BCELoss().cuda()

    for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, limbs_target, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs, limbs_ouptuts = model(input)

        target = target.cuda(non_blocking=True)  # ([64,17,64,48])
        target_weight = target_weight.cuda(non_blocking=True)
        target_limbs = target_limbs.cuda(non_blocking=True)


        #  only target
        if isinstance(outputs, list):
            target_loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                target_loss += criterion(output, target, target_weight)
        else:
            output = outputs
            target_loss = criterion(output, target, target_weight)

        skelton_loss = 0.01*crite(limbs_ouptuts, target_limbs)

        loss = target_loss + skelton_loss

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)
            print('kpt_loss:', target_loss, ' limbs_loss:', skelton_loss)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate_aaai_refine(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, limbs_target, meta) in enumerate(val_loader):
            # compute output
            outputs, limbs_ouptuts = model(input)

            if isinstance(outputs, list):
                # output = outputs[-1]
                output = outputs[0]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped, limbs_ouptuts_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    # output_flipped = outputs_flipped[-1]
                    output_flipped = outputs_flipped[0]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals

            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

def train_aaai_both(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    # crit = torch.nn.MSELoss().cuda()
    crite = torch.nn.BCELoss().cuda()

    for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, target_limbs, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        multi_outputs, outputs, limbs_ouptuts = model(input)

        target = target.cuda(non_blocking=True)  # ([64,17,64,48])
        target_weight = target_weight.cuda(non_blocking=True)
        all_ins_target = all_ins_target.cuda(non_blocking=True)
        all_ins_target_weight = all_ins_target_weight.cuda(non_blocking=True)
        target_limbs = target_limbs.cuda(non_blocking=True)

        #  only target
        if isinstance(outputs, list):
            target_loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                target_loss += criterion(output, target, target_weight)
        else:
            output = outputs
            target_loss = criterion(output, target, target_weight)

        #  0.5*interference + target
        if isinstance(multi_outputs, list):
            multi_loss = criterion(multi_outputs[0], all_ins_target, all_ins_target_weight)
            for multi_output in multi_outputs[1:]:
                multi_loss += criterion(multi_output, all_ins_target, all_ins_target_weight)
        else:
            multi_output = multi_outputs
            multi_loss = criterion(multi_output, all_ins_target, all_ins_target_weight)

        # skelton_loss = 0.01*crite(limbs_ouptuts, target_limbs)
        if limbs_ouptuts is not None:
            skelton_loss = 0.01*crite(limbs_ouptuts, target_limbs)
        else:
            skelton_loss = 0. * target.mean()

        loss = multi_loss + target_loss + skelton_loss

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)
            print('multi_kpt_loss:', multi_loss, 'kpt_loss:', target_loss, ' limbs_loss:', skelton_loss)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate_aaai_both(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, target_limbs, meta) in enumerate(val_loader):

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            # compute output
            multi_outputs, outputs, limbs_ouptuts = model(input)

            # check prediction quality
            # target_joints = meta['joints']
            # target_joints_vis = meta['joints_vis']
            # gt_heat = target
            # pred_heat = outputs
            # batch_size = target.shape[0]
            # num_joints = target.shape[1]
            # dist_kpts = []
            # num_2 = 0
            # num_5 = 0
            # num_2_to_5 = 0
            # for k in range(batch_size):  # target:([1,17,128,96]), outputs:([1,32,128,96]), for one image
            #     gt_joints = target_joints[k]
            #     gt_joints_vis = target_joints_vis[k]
            #     for i in range(num_joints):
            #         if gt_joints_vis[i][0] == 0:
            #             continue
            #         else:
            #             gt_joint_loc = torch.where(gt_heat[k][i]==gt_heat[k][i].max())
            #             pred_joint_loc = torch.where(pred_heat[k][i] == pred_heat[k][i].max())
            #             dist = ((gt_joint_loc[0]-pred_joint_loc[0])**2 + (gt_joint_loc[1]-pred_joint_loc[1])**2)**(0.5)
            #             dist = dist.detach().cpu().numpy()
            #             if (dist < 5).any():
            #                 pred_heat[k][i][gt_joint_loc] = 5
            #             dist_kpts.append(dist)
            # for i in dist_kpts:
            #     if (i <= 2).any():
            #         num_2 += 1
            #     elif (i >= 5).any():
            #         num_5 += 1
            #     else:
            #         num_2_to_5 += 1
            # percent_2 = num_2 / len(dist_kpts)
            # print('percent of joints less than 2 pixels:', percent_2)
            # percent_2_to_5 = num_2_to_5 / len(dist_kpts)
            # print('percent of joints between 2 to 5 pixels:', percent_2_to_5)
            # percent_5 = num_5 / len(dist_kpts)
            # print('percent of joints large than 5 pixels:', percent_5)

            if isinstance(outputs, list):
                # output = outputs[-1]
                output = outputs[0]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                multi_ouptuts_flipped, outputs_flipped, limbs_ouptuts_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    # output_flipped = outputs_flipped[-1]
                    output_flipped = outputs_flipped[0]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5


            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)



            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals

            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator
def rpg_train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    crite = torch.nn.BCELoss().cuda()

    for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, target_limbs, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        multi_outputs, outputs, limbs_ouptuts = model(input)

        target = target.cuda(non_blocking=True)  # ([64,17,64,48])
        target_weight = target_weight.cuda(non_blocking=True)
        all_ins_target = all_ins_target.cuda(non_blocking=True)
        all_ins_target_weight = all_ins_target_weight.cuda(non_blocking=True)
        target_limbs = target_limbs.cuda(non_blocking=True)

        #  only target
        if isinstance(outputs, list):
            target_loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                target_loss += criterion(output, target, target_weight)
        else:
            output = outputs
            target_loss = criterion(output, target, target_weight)

        #  0.5*interference + target
        if multi_outputs is not None:
            if isinstance(multi_outputs, list):
                multi_loss = criterion(multi_outputs[0], all_ins_target, all_ins_target_weight)
                for multi_output in multi_outputs[1:]:
                    multi_loss += criterion(multi_output, all_ins_target, all_ins_target_weight)
            else:
                multi_output = multi_outputs
                multi_loss = criterion(multi_output, all_ins_target, all_ins_target_weight)
        else:
            multi_loss = 0. * target.mean()
        # print(limbs_ouptuts.size(),target_limbs.size())
        if limbs_ouptuts is not None:
            skelton_loss = 0.01*crite(limbs_ouptuts, target_limbs)
        else:
            skelton_loss = 0. * target.mean()

        loss = multi_loss + target_loss + skelton_loss

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)
            print('multi_kpt_loss:', multi_loss.clone().detach().cpu().numpy(), 'kpt_loss:', target_loss.clone().detach().cpu().numpy(),
                  ' limbs_loss:', skelton_loss.clone().detach().cpu().numpy())

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def rpg_validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    all_dists = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, target_limbs, meta) in enumerate(val_loader):
            # compute output
            multi_outputs, outputs, limbs_ouptuts = model(input)

            if isinstance(outputs, list):
                # output = outputs[-1]
                output = outputs[0]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                multi_ouptuts_flipped, outputs_flipped, limbs_ouptuts_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    # output_flipped = outputs_flipped[-1]
                    output_flipped = outputs_flipped[0]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)
            # gt_preds, gt_maxvals, gt_coords = get_final_preds(
            #     config, target.clone().cpu().numpy(), c, s)
            # dists = erros_analyze(coords, gt_coords, meta['joints_vis'].cpu().numpy()[:, :, 0])
            # dists = np.concatenate([dists, meta['crowd_index'].cpu().numpy()], axis=2)
            # all_dists.append(dists)
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals

            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1
    # all_dists = np.vstack(all_dists)
    # pickle.dump(all_dists, open('w32_rsgnet_crowdpose_erros_statistics.pkl', 'wb'))
    return perf_indicator

def rsgnet_train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    crite = torch.nn.BCELoss().cuda()
    # rel_crite = torch.nn.MSELoss().cuda()
    if config.MODEL.UDP_POSE_ON:
        udp_criterion = UDPLosses(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()

    for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, target_limbs, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        person_target, _ = torch.max(target, dim=1)
        b, h, w = person_target.size()
        person_target = person_target.reshape(b, 1, h * w)
        relation_target = torch.matmul(person_target.permute(0, 2, 1), person_target)
        # relation_target = relation_target.cuda(non_blocking=True)
        multi_outputs, outputs, limbs_ouptuts, relation_scores = model(input, relation_target)

        target = target.cuda(non_blocking=True)  # ([64,17,64,48])
        target_weight = target_weight.cuda(non_blocking=True)
        all_ins_target = all_ins_target.cuda(non_blocking=True)
        all_ins_target_weight = all_ins_target_weight.cuda(non_blocking=True)
        target_limbs = target_limbs.cuda(non_blocking=True)

        if config.MODEL.UDP_POSE_ON:

            udp_target = meta['udp_target'].cuda(non_blocking=True)  # ([64,17,64,48])
            udp_target_weight = meta['udp_target_weight'].cuda(non_blocking=True)
            udp_outputs = outputs[1]
            outputs = outputs[0]

        #  only target
        if isinstance(outputs, list):
            target_loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                target_loss += criterion(output, target, target_weight)
        else:
            output = outputs
            target_loss = criterion(output, target, target_weight)

        #  0.5*interference + target
        if multi_outputs is not None:
            if isinstance(multi_outputs, list):
                multi_loss = criterion(multi_outputs[0], all_ins_target, all_ins_target_weight)
                for multi_output in multi_outputs[1:]:
                    multi_loss += criterion(multi_output, all_ins_target, all_ins_target_weight)
            else:
                # print(multi_output.size(), all_ins_target.size(), all_ins_target_weight.size())
                multi_output = multi_outputs
                multi_loss = criterion(multi_output, all_ins_target, all_ins_target_weight)
        else:
            multi_loss = 0. * target.mean()
        # print(limbs_ouptuts.size(),target_limbs.size())
        if limbs_ouptuts is not None:
            skelton_loss = 0.01*crite(limbs_ouptuts, target_limbs)
        else:
            skelton_loss = 0. * target.mean()
        # relation loss

        # diffs = (relation_target - relation_scores)**2
        relation_loss = 0.001 * torch.mean(relation_scores)

        loss = multi_loss + target_loss + skelton_loss + relation_loss
        if config.MODEL.UDP_POSE_ON:
            loss_udp_hm, loss_udp_os = udp_criterion(udp_outputs, udp_target, udp_target_weight)
            loss = loss + loss_udp_hm + loss_udp_os

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)
            print('multi_kpt_loss:', multi_loss.clone().detach().cpu().numpy(), 'kpt_loss:', target_loss.clone().detach().cpu().numpy(),
                  ' limbs_loss:', skelton_loss.clone().detach().cpu().numpy(),'relation loss:', relation_loss.clone().detach().cpu().numpy())
            if config.MODEL.UDP_POSE_ON:
                print('udp_hm_loss:', loss_udp_hm.clone().detach().cpu().numpy(), 'udp_os_loss:', loss_udp_os.clone().detach().cpu().numpy())

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def rsgnet_validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, target_limbs, meta) in enumerate(val_loader):
            # compute output
            # print('input', input.size())
            multi_outputs, outputs, limbs_ouptuts,_ = model(input)

            # check prediction quality
            # target_joints = meta['joints']
            # target_joints_vis = meta['joints_vis']
            # gt_heat = target
            pred_heat = outputs.detach().cpu()
            # batch_size = target.shape[0]
            # num_joints = target.shape[1]
            # dist_kpts = []
            # num_5 = 0
            # num_10 = 0
            # num_5_to_10 = 0
            # for k in range(batch_size):  # target:([1,17,128,96]), outputs:([1,32,128,96]), for one image
            #     gt_joints = target_joints[k]
            #     gt_joints_vis = target_joints_vis[k]
            #     for i in range(num_joints):
            #         if gt_joints_vis[i][0] == 0:
            #             continue
            #         else:
            #             gt_joint_loc = torch.where(gt_heat[k][i] == gt_heat[k][i].max())
            #             pred_joint_loc = torch.where(pred_heat[k][i] == pred_heat[k][i].max())
            #             dist = ((gt_joint_loc[0] - pred_joint_loc[0]) ** 2 + (gt_joint_loc[1] - pred_joint_loc[1]) ** 2) ** (0.5)
            #             dist = dist.numpy()
            #             if (dist < 10).any() :
            #                 pred_heat[k][i][gt_joint_loc] = 5
            #             dist_kpts.append(dist)
            # for i in dist_kpts:
            #     if (i <= 5).any():
            #         num_5 += 1
            #     elif (i >= 10).any():
            #         num_10 += 1
            #     else:
            #         num_5_to_10 += 1
            # percent_5 = num_5 / len(dist_kpts)
            # print('percent of joints less than 5 pixels:', percent_5)
            # percent_5_to_10 = num_5_to_10 / len(dist_kpts)
            # print('percent of joints between 5 to 10 pixels:', percent_5_to_10)
            # percent_10 = num_10 / len(dist_kpts)
            # print('percent of joints large than 10 pixels:', percent_10)

            outputs = pred_heat.cuda(non_blocking=True)

            if config.MODEL.UDP_POSE_ON:
                udp_outputs = outputs[1]
                outputs = outputs[0]
            if isinstance(outputs, list):
                # output = outputs[-1]
                output = outputs[0]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                multi_ouptuts_flipped, outputs_flipped, limbs_ouptuts_flipped,_ = model(input_flipped)
                if config.MODEL.UDP_POSE_ON:
                    udp_output_flipped = outputs_flipped[1]
                    outputs_flipped = outputs_flipped[0]
                    udp_output_flipped = flip_back_offset(udp_output_flipped.cpu().numpy(),
                                                      val_dataset.flip_pairs)
                    udp_output_flipped = torch.from_numpy(udp_output_flipped.copy()).cuda()
                    udp_outputs = (udp_outputs + udp_output_flipped) * 0.5

                if isinstance(outputs_flipped, list):
                    # output_flipped = outputs_flipped[-1]
                    output_flipped = outputs_flipped[0]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            # dark_preds, dark_maxvals = get_final_preds_with_dark(
            #     config, output.clone().cpu().numpy(), c, s)
            # preds = (dark_preds + preds) * 0.5
            # maxvals = (dark_maxvals + maxvals) * 0.5
            # if config.MODEL.UDP_POSE_ON:
            #     udp_preds, udp_maxvals = get_final_preds_with_offsets(
            #         config, udp_outputs.clone().cpu().numpy(), c, s)
            #     preds = (dark_preds + udp_preds) * 0.5
            #     maxvals = (dark_maxvals + udp_maxvals) * 0.5
            # gt_preds, gt_maxvals, gt_coords = get_final_preds(
            #     config, target.clone().cpu().numpy(), c, s)
            # dists = erros_analyze(coords, gt_coords, meta['joints_vis'].cpu().numpy()[:, :, 0])
            # dists = np.concatenate([dists, meta['crowd_index'].cpu().numpy()], axis=2)
            # all_dists.append(dists)
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals

            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1
    # all_dists = np.vstack(all_dists)
    # pickle.dump(all_dists, open('w32_rsgnet_crowdpose_erros_statistics.pkl', 'wb'))
    return perf_indicator

def trpnet_train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    crite = torch.nn.BCELoss().cuda()
    # rel_crite = torch.nn.MSELoss().cuda()
    if config.MODEL.UDP_POSE_ON:
        udp_criterion = UDPLosses(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()

    for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, target_limbs, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        person_target, _ = torch.max(target, dim=1)
        b, h, w = person_target.size()
        person_target = person_target.reshape(b, 1, h, w)
        person_target = torch.nn.functional.interpolate(person_target, scale_factor=1/2, mode='bilinear',
                                                          align_corners=True)
        person_target = torch.squeeze(person_target)
        b, h, w = person_target.size()
        person_target = person_target.reshape(b, 1, h * w)
        relation_target = torch.matmul(person_target.permute(0, 2, 1), person_target)
        # relation_target = relation_target.cuda(non_blocking=True)

        # multi_outputs, outputs, limbs_ouptuts, relation_scores = model(input, relation_target)
        multi_outputs, outputs, relation_scores = model(input, relation_target)
        multi_outputs, outputs, _ = model(input)
        # multi_outputs, outputs = model(input)

        target = target.cuda(non_blocking=True)  # ([64,17,64,48])
        target_weight = target_weight.cuda(non_blocking=True)
        all_ins_target = all_ins_target.cuda(non_blocking=True)
        all_ins_target_weight = all_ins_target_weight.cuda(non_blocking=True)
        # target_limbs = target_limbs.cuda(non_blocking=True)

        if config.MODEL.UDP_POSE_ON:

            udp_target = meta['udp_target'].cuda(non_blocking=True)  # ([64,17,64,48])
            udp_target_weight = meta['udp_target_weight'].cuda(non_blocking=True)
            udp_outputs = outputs[1]
            outputs = outputs[0]

        #  only target
        if isinstance(outputs, list):
            target_loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                target_loss += criterion(output, target, target_weight)
        else:
            output = outputs
            target_loss = criterion(output, target, target_weight)

        #  0.5*interference + target
        if multi_outputs is not None:
            if isinstance(multi_outputs, list):
                multi_loss = criterion(multi_outputs[0], all_ins_target, all_ins_target_weight)
                for multi_output in multi_outputs[1:]:
                    multi_loss += criterion(multi_output, all_ins_target, all_ins_target_weight)
            else:
                # print(multi_output.size(), all_ins_target.size(), all_ins_target_weight.size())
                multi_output = multi_outputs
                multi_loss = criterion(multi_output, all_ins_target, all_ins_target_weight)
        else:
            multi_loss = 0. * target.mean()
        # print(limbs_ouptuts.size(),target_limbs.size())

        # relation loss
        relation_loss = 0.001 * torch.mean(relation_scores)

        loss = multi_loss + target_loss + relation_loss
        # loss = multi_loss + target_loss

        if config.MODEL.UDP_POSE_ON:
            loss_udp_hm, loss_udp_os = udp_criterion(udp_outputs, udp_target, udp_target_weight)
            loss = loss + loss_udp_hm + loss_udp_os

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)
            print('multi_kpt_loss:', multi_loss.clone().detach().cpu().numpy(), 'kpt_loss:', target_loss.clone().detach().cpu().numpy(),
                        'relation loss:', relation_loss.clone().detach().cpu().numpy())
            # print('multi_kpt_loss:', multi_loss.clone().detach().cpu().numpy(), 'kpt_loss:',
            #       target_loss.clone().detach().cpu().numpy())
            if config.MODEL.UDP_POSE_ON:
                print('udp_hm_loss:', loss_udp_hm.clone().detach().cpu().numpy(), 'udp_os_loss:', loss_udp_os.clone().detach().cpu().numpy())

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def trpnet_validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, all_ins_target, all_ins_target_weight, target_limbs, meta) in enumerate(val_loader):
            # compute output
            # print('input', input.size())

            multi_outputs, outputs, _ = model(input)
            # multi_outputs, outputs = model(input)

            if config.MODEL.UDP_POSE_ON:
                udp_outputs = outputs[1]
                outputs = outputs[0]
            if isinstance(outputs, list):
                # output = outputs[-1]
                output = outputs[0]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                # multi_ouptuts_flipped, outputs_flipped, limbs_ouptuts_flipped,_ = model(input_flipped)
                multi_ouptuts_flipped, outputs_flipped, _ = model(input_flipped)
                # multi_ouptuts_flipped, outputs_flipped= model(input_flipped)
                if config.MODEL.UDP_POSE_ON:
                    udp_output_flipped = outputs_flipped[1]
                    outputs_flipped = outputs_flipped[0]
                    udp_output_flipped = flip_back_offset(udp_output_flipped.cpu().numpy(),
                                                      val_dataset.flip_pairs)
                    udp_output_flipped = torch.from_numpy(udp_output_flipped.copy()).cuda()
                    udp_outputs = (udp_outputs + udp_output_flipped) * 0.5

                if isinstance(outputs_flipped, list):
                    # output_flipped = outputs_flipped[-1]
                    output_flipped = outputs_flipped[0]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            # dark_preds, dark_maxvals = get_final_preds_with_dark(
            #     config, output.clone().cpu().numpy(), c, s)
            # preds = (dark_preds + preds) * 0.5
            # maxvals = (dark_maxvals + maxvals) * 0.5
            # if config.MODEL.UDP_POSE_ON:
            #     udp_preds, udp_maxvals = get_final_preds_with_offsets(
            #         config, udp_outputs.clone().cpu().numpy(), c, s)
            #     preds = (dark_preds + udp_preds) * 0.5
            #     maxvals = (dark_maxvals + udp_maxvals) * 0.5
            # gt_preds, gt_maxvals, gt_coords = get_final_preds(
            #     config, target.clone().cpu().numpy(), c, s)
            # dists = erros_analyze(coords, gt_coords, meta['joints_vis'].cpu().numpy()[:, :, 0])
            # dists = np.concatenate([dists, meta['crowd_index'].cpu().numpy()], axis=2)
            # all_dists.append(dists)
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals

            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1
    # all_dists = np.vstack(all_dists)
    # pickle.dump(all_dists, open('w32_rsgnet_crowdpose_erros_statistics.pkl', 'wb'))
    return perf_indicator

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
