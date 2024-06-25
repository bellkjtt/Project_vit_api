import torch
from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np

class ViTDetModel:
    def __init__(self, config_file, checkpoint_file, device='cuda:0', iou_threshold=0.5, confidence_threshold=0.3):
        self.device = device
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

        if torch.cuda.is_available():
            print("CUDA is available")
            print(f"Model is on CUDA: {next(self.model.parameters()).is_cuda}")
        else:
            print("CUDA is not available")
            self.device = 'cpu'  # fallback to CPU

    def predict(self, img):
        result = inference_detector(self.model, img)
        return result

    def filter_results(self, result):
        filtered_result = []
        if result.pred_instances.scores is not None:
            mask = result.pred_instances.scores >= self.confidence_threshold
            filtered_result = [
                result.pred_instances.bboxes[mask].cpu().numpy(),
                result.pred_instances.labels[mask].cpu().numpy(),
                result.pred_instances.scores[mask].cpu().numpy()
            ]
        return filtered_result

    def show_result(self, img_path, result, out_file):
        img = mmcv.imread(img_path)
        self.model.show_result(img, result, out_file=out_file, score_thr=self.confidence_threshold)

    def get_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def nms(self, boxes, scores, iou_threshold):
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            ious = np.array([self.get_iou(boxes[i], boxes[j]) for j in order[1:]])
            inds = np.where(ious <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def apply_nms(self, filtered_result):
        boxes, labels, scores = filtered_result
        nms_result = []
        
        for class_id in np.unique(labels):
            mask = labels == class_id
            class_boxes = boxes[mask]
            class_scores = scores[mask]
            
            keep = self.nms(class_boxes, class_scores, self.iou_threshold)
            nms_result.extend([(class_boxes[i], class_id, class_scores[i]) for i in keep])
        
        return nms_result

    def predict_with_nms(self, img):
        result = self.predict(img)
        filtered_result = self.filter_results(result)
        nms_result = self.apply_nms(filtered_result)
        return nms_result