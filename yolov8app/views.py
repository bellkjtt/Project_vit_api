from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import cv2
import os
from .test import ViTDetModel

# ViTDet 모델 설정 및 가중치 파일 경로
BASE_DIR = 'C:/Users/Administer/새 폴더/------'
config_file = os.path.join(BASE_DIR, 'mmdetection/projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py')
checkpoint_file = os.path.join(BASE_DIR, 'mysite/videt.pth')

# ViTDet 모델 초기화 (기본 임계값 설정)
model = ViTDetModel(config_file, checkpoint_file, iou_threshold=0.5, confidence_threshold=0.3)

@csrf_exempt
def predict(request):
    if request.method == 'POST' and 'image' in request.FILES:
        # 요청에서 이미지를 읽기
        image_file = request.FILES['image']
        image = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        # 요청에서 confidence와 iou 값을 읽기
        confidence = float(request.POST.get('confidence', 0.3))
        iou = float(request.POST.get('iou', 0.5))

        # ViTDet 모델의 임계값 업데이트
        model.confidence_threshold = confidence
        model.iou_threshold = iou

        # ViTDet 모델 예측 (NMS 적용)
        result = model.predict_with_nms(image)
        
        # JSON 응답 생성
        detections = []

        for box, class_id, score in result:
            xmin, ymin, xmax, ymax = box
            detections.append({
                'xmin': int(xmin),
                'ymin': int(ymin),
                'xmax': int(xmax),
                'ymax': int(ymax),
                'confidence': float(score),
                'class': int(class_id)
            })
        
        return JsonResponse(detections, safe=False)
    else:
        return JsonResponse({"error": "Invalid request method or missing image"}, status=400)