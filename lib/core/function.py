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
        person_target = person_target.reshape(b, 1, h, w)
        person_target = torch.nn.functional.interpolate(person_target, scale_factor=1/2, mode='bilinear',
                                                          align_corners=True)
        person_target = torch.squeeze(person_target)
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
                multi_output = multi_outputs
                multi_loss = criterion(multi_output, all_ins_target, all_ins_target_weight)
        else:
            multi_loss = 0. * target.mean()
        if limbs_ouptuts is not None:
            skelton_loss = 0.01 * crite(limbs_ouptuts, target_limbs)
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
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)
            print('multi_kpt_loss:', multi_loss.clone().detach().cpu().numpy(), 'kpt_loss:',
                  target_loss.clone().detach().cpu().numpy(),
                  ' limbs_loss:', skelton_loss.clone().detach().cpu().numpy(), 'relation loss:',
                  relation_loss.clone().detach().cpu().numpy())
            if config.MODEL.UDP_POSE_ON:
                print('udp_hm_loss:', loss_udp_hm.clone().detach().cpu().numpy(), 'udp_os_loss:',
                      loss_udp_os.clone().detach().cpu().numpy())

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred * 4, output,
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

            multi_outputs, outputs, limbs_ouptuts,_ = model(input)

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

def generate_points_from_labels(meta):
    image_list = meta['image']
    kpts = defaultdict(list)
    for image in image_list:
        kpts[image].append()

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
