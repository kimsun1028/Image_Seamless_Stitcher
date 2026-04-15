# Image Seamless Stitcher

OpenCV와 NumPy를 사용해 3장의 이미지를 하나의 파노라마처럼 자연스럽게 이어 붙이는 스크립트 프로젝트입니다.

## Overview

이 프로젝트는 아래 순서로 동작합니다.

1. `data/image01.jpg`, `data/image02.jpg`, `data/image03.jpg`를 읽습니다.
2. 가운데 이미지(`image02`)를 기준으로 좌/우 이미지를 호모그래피로 정렬합니다.
3. 겹치는 영역은 feather blending으로 자연스럽게 합칩니다.
4. 중심 이미지는 경계만 부드럽게 섞고 내부 선명도를 최대한 유지합니다.
5. 결과를 `data/stitched_result.jpg`로 저장하고 창으로 미리보기합니다.

## Project Structure

```text
Image_Seamless_Stitcher/
├─ image_seamless_stitcher.py
├─ data/
│  ├─ image01.jpg
│  ├─ image02.jpg
│  ├─ image03.jpg
│  └─ stitched_result.jpg   # 실행 후 생성
└─ screenshot.png           # 선택: 결과 예시 이미지
```

## Requirements

- Python 3.9+
- numpy
- opencv-python

설치:

```bash
pip install numpy opencv-python
```

## How To Run

프로젝트 루트에서 실행:

```bash
python image_seamless_stitcher.py
```

성공 시 콘솔에 저장 경로가 출력되고, OpenCV 창에 stitching 결과가 표시됩니다.

## Input / Output

- Input:
  - `data/image01.jpg` (오른쪽 이미지)
  - `data/image02.jpg` (가운데 기준 이미지)
  - `data/image03.jpg` (왼쪽 이미지)
- Output:
  - `data/stitched_result.jpg`

## Notes

- 현재 스크립트는 상대 경로(`data/...`)를 사용하므로, 반드시 프로젝트 루트에서 실행하세요.
- OpenCV 창이 뜬 상태에서 아무 키를 누르면 종료됩니다.
- 특징점이 충분하지 않으면 `Not enough good matches` 오류가 날 수 있습니다. 이 경우 질감이 더 풍부한 이미지로 테스트해 보세요.
