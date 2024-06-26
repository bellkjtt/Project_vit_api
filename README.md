# vit_api

mm detection을 추가해서 사용해야 한다.

https://github.com/open-mmlab/mmdetection

```cmd
git clone https://github.com/open-mmlab/mmdetection.git


vit.pth 파일도 다운 받아야 합니다.
https://github.com/open-mmlab/mmdetection/tree/main/projects/ViTDet


그런 다음 views.py에 있는 checkpoint와 config 파일의 경로를 바꾸고, python manage.py runserver를 통해 동작시킵니다.
실행 사항은 python.py를 실행시켜 해당하는 ip로 주면 됩니다. 로컬 상황에서는 127.0.0.1:8000으로 줘야 합니다.
