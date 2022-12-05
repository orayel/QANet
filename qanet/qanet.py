import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone

from .features_enhance import build_features_enhance
from .features_merging import build_features_merging
from .position_embeding import build_position_embeding
from .answer_branch import build_answer_branch
from .question_branch import build_question_branch
from .question2answer import build_question2answer
from .eprs_detector import build_eprs_detector
from .criterion import build_criterion
from .utils import nested_tensor_from_tensor_list

__all__ = ["QANet"]


def rescoring_mask(scores, masks, threshold):
    masks_pred = (masks > threshold).float()
    return scores * ((masks * masks_pred).sum([1, 2]) / (masks_pred.sum([1, 2]) + 1e-6))


@META_ARCH_REGISTRY.register()
class QANet(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # device
        self.device = torch.device(cfg.MODEL.DEVICE)

        # backbone
        self.backbone = build_backbone(cfg)
        output_shape = self.backbone.output_shape()
        # features enhance module
        self.fem = build_features_enhance(cfg, output_shape)
        # features merging module
        self.fmm = build_features_merging(cfg)
        # position embeding
        self.pe = build_position_embeding(cfg)
        # question branch
        self.qb = build_question_branch(cfg)
        # answer branch
        self.ab = build_answer_branch(cfg)
        # question to answer
        self.q2a = build_question2answer(cfg)

        # error-prone region detector module
        self.epr_dm = build_eprs_detector(cfg)
        # criterion
        self.criterion = build_criterion(cfg, self.epr_dm)

        # data and preprocessing
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)

        # inference
        self.obj_threshold = cfg.MODEL.INFERENCE.OBJ_THRESHOLD
        self.mask_threshold = cfg.MODEL.INFERENCE.MASK_THRESHOLD
        self.max_detections = cfg.MODEL.INFERENCE.MAX_DETECTIONS

    def normalizer(self, image):
        image = (image - self.pixel_mean) / self.pixel_std
        return image

    def preprocess_inputs(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, 32)
        return images

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            gt_classes = targets_per_image.gt_classes
            target["labels"] = gt_classes.to(self.device)
            h, w = targets_per_image.image_size
            if not targets_per_image.has('gt_masks'):
                gt_masks = BitMasks(torch.empty(0, h, w))
            else:
                gt_masks = targets_per_image.gt_masks
                if self.mask_format == "polygon":
                    if len(gt_masks.polygons) == 0:
                        gt_masks = BitMasks(torch.empty(0, h, w))
                    else:
                        gt_masks = BitMasks.from_polygon_masks(
                            gt_masks.polygons, h, w)

            target["masks"] = gt_masks.to(self.device)
            new_targets.append(target)

        return new_targets

    def forward(self, batched_inputs):

        images = self.preprocess_inputs(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        max_shape = images.tensor.shape[2:]

        # forward
        features = self.backbone(images.tensor)
        features = self.fem(features)
        features = self.fmm(features)
        features = self.pe(features)
        # location sensitive features
        lsf = self.qb(features)
        # mask features, edge features, object features, error-prone region features
        mf, ef, of, eprf = self.ab(features)
        output = self.q2a(lsf, mf, ef, of, eprf)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            losses = self.criterion(output, targets, max_shape)
            return losses
        else:
            results = self.inference(
                output, batched_inputs, max_shape, images.image_sizes)
            processed_results = [{"instances": r} for r in results]
            return processed_results

    # TODO
    def forward_test(self, images):
        # for inference, onnx, tensorrt
        # input images: BxCxHxW, fixed, need padding size
        # normalize
        images = (images - self.pixel_mean[None]) / self.pixel_std[None]
        features = self.backbone(images)
        features = self.encoder(features)
        output = self.decoder(features)

        pred_scores = output["pred_logits"].sigmoid()
        pred_masks = output["pred_masks"].sigmoid()
        pred_object = output["pred_scores"].sigmoid()
        pred_scores = torch.sqrt(pred_scores * pred_object)
        pred_masks = F.interpolate(
            pred_masks, scale_factor=4.0, mode="bilinear", align_corners=False)
        return pred_scores, pred_masks

    def inference(self, output, batched_inputs, max_shape, image_sizes):

        results = []
        pred_obj = output["pred_obj"].sigmoid()
        pred_masks = output["pred_masks"].sigmoid()

        for _, (obj_pred_per_image, mask_pred_per_image, batched_input, img_shape) in enumerate(zip(
                pred_obj.flatten(1), pred_masks, batched_inputs, image_sizes)):

            ori_shape = (batched_input["height"], batched_input["width"])
            result = Instances(ori_shape)

            keep = obj_pred_per_image > self.obj_threshold
            scores = obj_pred_per_image[keep]
            mask_pred_per_image = mask_pred_per_image[keep]

            if scores.size(0) == 0:
                result.scores = scores
                result.pred_classes = torch.zeros_like(scores)
                results.append(result)
                continue

            h, w = img_shape
            scores = rescoring_mask(scores, mask_pred_per_image, self.mask_threshold)
            scores *= ((mask_pred_per_image * (mask_pred_per_image > self.mask_threshold)).sum([1, 2])
                       / ((mask_pred_per_image > self.mask_threshold).sum([1, 2]) + 1e-6))

            # upsampling to 1x resolution, and then crop to remove padding region, then resize to origin size
            mask_pred_per_image = F.interpolate(
                mask_pred_per_image[:, None], size=max_shape, mode="bilinear", align_corners=False)[:, :, :h, :w]
            mask_pred_per_image = F.interpolate(
                mask_pred_per_image, size=ori_shape, mode='bilinear', align_corners=False).squeeze(1)

            mask_pred = mask_pred_per_image > self.mask_threshold
            result.pred_masks = mask_pred
            result.scores = scores
            result.pred_classes = torch.zeros_like(scores)
            results.append(result)

        return results
