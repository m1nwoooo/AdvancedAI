# AdvancedAI
Image Deblurring using U-Net Architecture with Skip Connections


# Image Deblurring with Residual U-Net

## 📋 프로젝트 개요

흐릿하게 촬영된 이미지로부터 선명한 이미지를 복원하는 CNN 기반 Deep Learning 프로젝트입니다. ResNet과 U-Net 구조를 결합하여 효과적인 deblurring 성능을 달성했습니다.

네트워크의 구조 및 설계 과정은 Assignment.pdf로 첨부하였습니다.

**Environment**: RTX 3080 GPU

---

## 🎯 프로젝트 목표

- CNN 구조를 바탕으로 Image Deblur 네트워크 설계 및 구현
- GoPro Dataset을 활용한 모델 학습(3 Hours)
- PSNR(Peak Signal-to-Noise Ratio) 지표를 통한 성능 평가
- 딥러닝 모델 개발의 전반적인 과정 경험 (구조 설계, Loss 함수 선택, Dropout 등)

---

## 📈 Results

### Performance
- **PSNR**: 27.84 dB (Validation Set)
- **Dataset**: GoPro Dataset
- **GPU**: RTX 3080

### Qualitative Results

| Input (Blurred) | Output (Deblurred) |
|-----------------|-------------------|
|<img width="512" height="512" alt="01" src="https://github.com/user-attachments/assets/693df66e-8c2c-47dd-9e93-eb984f6fef97" /> | <img width="512" height="512" alt="deblurred_01" src="https://github.com/user-attachments/assets/746cfa2e-f2f9-4c3f-9682-3146dee4b7fb" />) |

효과적인 blur 제거 및 선명한 이미지 복원 확인

---

## 📁 Project Structure
```
deblur-project/
├── train.py              # 모델 학습 코드
├── infer.py              # 추론 코드
├── model.py              # 네트워크 아키텍처
├── dataset.py            # 데이터 로더
├── loss.py               # Loss functions
└── README.md
```

---

## 🚀 Usage

### Training
```bash
python train.py --data_path ./data/GoPro --batch_size 8 --epochs 350
```

### Inference
```bash
python infer.py --input ./input_image.jpg --output ./output_image.jpg --checkpoint ./best_model.pth
```

---

## 🎓 Key Learnings

1. **강의 이론의 실전 적용**: CNN, Residual Connection, Batch Normalization 등
2. **PyTorch 모델 구현 경험**: 전체 파이프라인 구축
3. **하이퍼파라미터 튜닝**: Loss weights, Learning rate scheduling
4. **기본 기법의 효과**: 복잡한 기법 없이도 의미 있는 성능 달성

---

## 🔮 Future Improvements

- **Attention Mechanism**: Self-attention, Channel attention 추가
- **GAN-based Approach**: Adversarial loss로 시각적 품질 향상
- **Transformer Architecture**: Vision Transformer 적용
- **Multi-scale Training**: 다양한 해상도에서 학습

---

## 📝 References

- U-Net: Convolutional Networks for Biomedical Image Segmentation
- Deep Residual Learning for Image Recognition (ResNet)
- Perceptual Losses for Real-Time Style Transfer and Super-Resolution
- GoPro Dataset for Dynamic Scene Deblurring

---

**학번**: 202110410  
**과목**: 인공지능심화
