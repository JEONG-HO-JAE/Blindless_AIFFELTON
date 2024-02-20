![image](https://github.com/sujin7822/README/assets/122075306/8155c73c-d085-4313-9f76-da22229644ef)

# AIFFEL X VISUWORKS
* 모두의연구소 산하 인공지능 교육기관 AIFFEL과 인공지능 기반 안구 솔루션 전문기업 VISUWORKS가 협력하여 진행된 기업 연계 프로젝트로서,
  [Vessel Segmentation On Fundus Image] 주제에 대해 VISUWORKS로 부터 제공 받은 안저 사진 데이터를 기반으로 프로젝트를 진행하였습니다.

<br/>

## 1. 프로젝트 소개

### 1.1 개요
- 주제 : **Vessel Segmentation On Fundus Image**  
- 기간 : 2024.01.02(화)  ~ 2024.02.21(수)  
- 방식 : 팀 프로젝트
- Keyword : Segmentation
- 배경
    * 사람의 시야를 건강하게 유지하는데 도움이 되기를 바라는 마음으로 시작.
    * 안저 사진 진단 시 의사에게 도움이 되는 도구를 만들고자 함.
    * 의사가 안저 사진 판독 시 소요되는 시간 및 정확도 상승 기대.

<br/>

### 1.2 프로젝트 목표
- 다양한 model과 data generator를 개발 후 정량, 정성평가를 통해 성능 좋은 model과 data generator를 찾는다.
- Vessel Segmentation

<br/>

### 1.3 구성원 

|이름|구성|역할|
|-----| -----|---------| 
|정호재|팀장 |Task & Time management,​ Data generator 및 실험 설계​, Segmentation task정성평가 및 정량 평가를 위한 tool 설계| 
|김 산|팀원 |데이터 구성 및 전처리​, Ablation Study | 
|김수진|팀원 |Model 구현​, Ablation Study | 

<br/>

### 1.4 기술 스택
- Tensorflow, Pytorch, OpenCV, PIL, QGIS, Pandas, numpy, Matplotlib, Seaborn 외


<br/>

## 2. 데이터 정의 및 전처리

### 2.1 데이터셋 정의
- 다양한 외부 데이터 셋(15개)을 바탕으로 데이터 수집을 하여 하나의 데이터 셋으로 통합
    - Datasets: AFIO, ARIA, ChaseDB, DR_Hagis, DRIVE, DualModal2019, FIVES,  
      HRF, IOSTAR_STAR, LES-AV, ORVS, RETA, STARE, TREND, UoA_DR
    - RGB3채널의 안저 이미지
    - Label: 혈관 mask에 대한 1채널 grayscale 이미지

<p align="center"><img src="https://github.com/3n952/aiffel-sia/assets/107621083/1d15dab1-ed77-448f-bd75-a02fdf874048"></p> 

- train/val/test 데이터 셋
  각 데이터 셋 마다 healthy / unhealthy 비율이 다르고, 해상도도 모두 상이함.  
  -> 위의 내용을 고려하여 Train / val / Test = 6:2:2 비율로 나누어 데이터 셋을 구성

|Train|Test|
|---|---|
|1380|233|

|Healthy|Unhealthy|Healthy|Unhealthy|
|---|---|---|---|
|636|744|105|208|

<br/>

### 2.2 Pre-processing
- Fundus image의 3채널(RGB)에는 Noise 정보가 많이 표현됨. -> Noise 정보가 가장 표현되지 않는 Green Channel을 추출
- 추출된 Green Channel에 CLAHE(Contrast Limited Adaptive Histogram Equalization​) 기법을 적용

<p align="center"><img src="https://github.com/3n952/aiffel-sia/assets/107621083/e0e899aa-63eb-4d60-ac3e-90fe06d40e21"></p> 
  
- outlier(붉은색을 띄는 Fundus이미지)에 대한 강건한 전처리 기법 추가
  1. 3채널 이미지에서 r 채널 픽셀분포를 확인.​
  2. 170이상의 값을 가지는 픽셀이 전체의 25%이상인지 확인.  
  (픽셀값 100 과 전체 픽셀의 50으로 선정한 기준: red image의 픽셀 히스토그램을 분석했을 때, 200을 넘는 값이 전체의 0.2가 넘는다는 것을 확인.  
   이에 조금 강건하게 적용시키고자 170을 넘는 값이 전체의 .25인 경우에 대해 해당 전처리를 적용.(170 / 25의 기준은 실험적으로 정한 기준))​
  3. 2단계의 과정이 확인된 이미지의 r채널에 대해 CLAHE기법을 적용.​
  4. 3단계의 결과는 혈관이 조금 밝게 보임. 이를 보다 뚜렷하게 하기 위해 원래의 g 채널에 CLAHE를 적용한 이미지의 픽셀 값과 합침.​
  5. 동일한 가중치로 픽셀을 합치지 않고, red channel + clahe에 더 가중치를 주고 이미지를 합성.

<p align="center"><img src="https://github.com/3n952/aiffel-sia/assets/107621083/8c5290bc-5095-48ce-b211-d88c51dbd230"></p> 

<br/>

## 3. Experimentations
> Generator / Model ablation study


### 3.1 Experimentation Process 
<img width="641" src="https://github.com/3n952/aiffel-sia/assets/107621083/2e693739-4c42-4fa5-9238-8631f019775e">  

1. **'Unet - FG/SG/AG'** => Generator Ablation study 
2. **the others** => Model Ablatation study

<br/>

### 3.2 Generator Ablation study
> Baseline model인 Dense Unet을 기반으로 FG/SG/AG의 효과와 성능을 비교,분석


- **Baseline Model**: Dense Unet
  
<p align="center"><img src="https://github.com/3n952/aiffel-sia/assets/107621083/e58ff1d5-9cff-437c-af55-e1c589d0cc06"></p>

<br/>

1. Full sized Generator(FG): 정해진 크기로 resize
<p align="center"><img src="https://github.com/3n952/aiffel-sia/assets/107621083/61479217-fb5f-4666-a7fd-a3e7dc27ab34"></p>
<br/>
3. Slice sized Generator(SG): 전체 이미지를 1024, 1024로 resize 시킨 후 512, 512사이즈로 이미지를 잘라 patches들을 구성
<p align="center"><img src="https://github.com/3n952/aiffel-sia/assets/107621083/f8a24230-25c9-4c6a-af19-a7e8a0c43769"></p>
<br/>
5. Adaptive sized Generator(AG): 각 이미지를 512, 512 사이즈로 이미지를 잘라 patches를 구성한 후, 학습 시에 batch size에 맞게 이미지 patch를 random하게 구성
<p align="center"><img src="https://github.com/3n952/aiffel-sia/assets/107621083/72c82383-949a-4752-ba39-4d46d47c314a"></p>
<br/>

### Quantitative Evaluation
<br/>

- Test Dataset에 대한 정량 평가 결과는 FG와 SG사이에 큰 차이가 없음.​   
- 반면 AG는 전반적으로 좋지 않은 점수를 기록.

<p align="center"><img src="https://github.com/3n952/aiffel-sia/assets/107621083/da546980-9bd0-4dd9-9ad9-a0b389daeae0"width="600" height="400"></p>


### Qualitative Evaluation

<br/>
- FG vs SG

- FG와 SG는 정성적 평가에서도 큰 차이가 없음
- 두 Generator 모두 비교적 굵은 혈관과 미세 혈관을 잘 구현함

<p align="center"><img src="https://github.com/3n952/aiffel-sia/assets/107621083/2c3ce3c2-8292-4ccc-b3bf-0a56acad18cb"width="600" height="400"></p>


<br/>
- FG vs AG

- FG / AG에서는 큰 차이가 존재: AG는 굵은 혈관을 부자연스럽게 구현하고 중간 굵기와 미세 혈관은 잘 구현하지 못함​
- AG가 FG에 비해 Sensitivity가 높지만 Specificity가 낮음
- 혈관을 굵게 표현한 다는 점이 sensitivity를 높이면서 동시에 specificity를 낮추는 요소로 작용했다고 판단
​- IOU 지표 또한 많이 차이나는 것으로 미뤄보아 AG의 전반적인 성능이 많이 떨어짐

<p align="center"><img src="https://github.com/3n952/aiffel-sia/assets/107621083/3f490744-b73d-432d-9189-f637da0cf152"width="600" height="400"></p>



<br/>

### 3.3 Model Ablation study
> 모델의 결과 지표를 통해 효과와 성능을 비교,분석

### 1. Inception + Pyramid Unet
- **Model** : Inception + Pyramid Unet

    - Unet을 기본 구조로 하고 Inception Module과 Pyramid Pooling Module을 통합
    - inception Module : 다양한 kernel size를 사용함으로써 다양한 크기의 혈관의 feature를 반영하는 feature map을 구성

    - Pyramid Pooling Module : 여러 다양한 크기의 피라미드 영역을 생성하고, 각 영역에 대해 풀링 연산을 수행한 후, 이를 합치는 방식으로 작동함으로써 다양한 크기의 혈관 및 구조에 대한 정보를 통합
         
<p align="center">
  <img src="https://github.com/sujin7822/README/assets/122075306/988f48e9-e9d6-4ef5-ac63-17231fc0badd" alt="Image1" width="300" height="200" />
  <img src="https://github.com/sujin7822/README/assets/122075306/0e999b63-553d-443a-9f78-cfbcab754500" alt="Image2" width="300" height="100" />
  <img src="https://github.com/sujin7822/README/assets/122075306/6bf4eea8-4f32-42a3-b6e1-9fa21e1748bc" alt="Image3" width="300" height="200" />
</p>

<br/>

### **_Compare with Unet_**
### Quantitative Evaluation
- 앞선 기대와는 다르게 평가지표에서 Inception + Pyramid Unet의 sensitivity에서 성능이 좋지 않고 ​다른 지표들 또한 높은 폭으로 상승하지 않음

![image](https://github.com/sujin7822/README/assets/122075306/91bb2eb5-8c73-4adc-a1eb-9c6bee7f362b)


### Qualitative Evaluation

- 빨간 색 원을 비교해 보면 Unet이 Inception + Pyramid Unet보다 미세혈관을 더 잘 표현

- 파란색 원을 보면 Unet이 Inception + Pyramid Unet보다 깔끔한 segmentation을 수행

![image](https://github.com/sujin7822/README/assets/122075306/1d11ee01-7384-412d-811b-7467c3e5b169)


<br/>

### **_Wrap up_**
- Unet이 미세혈관 부분을 Inception + Pyramid Unet보다 잘 구현​
- Inception + Pyramid Unet 이 Unet보다 깔끔하고 정확한 Segmentation을 하지 못함
- 그 이유는 Pyramid Pooling 때문이라고 판단
<br/>


### 2. SD Unet
- **Model** : SD Unet
    - Unet을 기본 구조로 하고 structured dropout을 통합
    - 과적합을 방지하여 성능 향상을 기대
       - structured dropout : 전통적인 완전 연결 층의 dropout을 사용하지 않고 structured dropout을 적용하여 정규화를 진행

<p align="center">
  <img src="https://github.com/sujin7822/README/assets/122075306/eae77ca8-4542-42f3-b836-26f69c6ecdc5" alt="Image1" width="300" height="200" />
  <img src="https://github.com/sujin7822/README/assets/122075306/1e1731ca-3aea-461e-bf5b-631c967b589b" alt="Image2" width="200" height="100" />
  <img src="https://github.com/sujin7822/README/assets/122075306/196fdb50-c788-4de3-81a9-ed6a6458bfee" width="300" height="200" />
</p>

### **_Compare with Dense Unet_**
### Quantitative Evaluation

- 정량평가로는 SD Unet이 Unet보다 좋은지 판단하기 어려움이 있음

![image](https://github.com/sujin7822/README/assets/122075306/3b0c694f-5e20-4a76-8e95-1273aa1d6597)


### Qualitative Evaluation

- 빨간색원을 보면 Unet이 SD Unet보다 조금 더 미세혈관을 더 잘 표현
- 하지만 다른 사진들에서 비교했을 때 Unet이 SD Unet보다 미세혈관을 보편적으로 잘 나타낸다고 결론내기 어려움

![image](https://github.com/sujin7822/README/assets/122075306/86d305d4-b72a-4eac-95d2-86768530401d)

- 이 그림에서는 SD Unet이 미세혈관을 더 잘 표현
- SD Unet이 정성적 평가에서 노이즈에 민감하고 굵은 혈관을 깔끔하게 표현하지 못함
- 노색 원들을 보면 SD Unet이 noise을 vessel이라고 오판한 경우가 많다는 것을 알 수 있고  
 파란색 원을 보면 SD Unet이  굵은 혈관을 표현 못함

![image](https://github.com/sujin7822/README/assets/122075306/e92596b0-3007-4159-b9fb-d9ac4a37b554)


- Noise가 많지 않은 사진에서도 파란색 부분을 비교해보면 오른쪽 SD Unet이 굵은 혈관을  
 잘 구현하지 못한다는 것을 확인

![image](https://github.com/sujin7822/README/assets/122075306/9cc7f3ac-f051-4b2f-89fc-b0978306e289)


<br/>

### **_Wrap up_**
- Unet과 SD Unet 모두 정량 정성 평가에서 확연한 차이가 없음
- 하지만 SD Unet에서 두 가지 취약성을 발견
     - 첫째, Noise 에 취약
     - 둘째, 굵은 혈관을 깔끔하게 Segmentation하지 못함
<br/>

### 3.1 FR Unet w DS
- **Model** : FR Unet with Deep Supervsion
   - Unet++가 기존의 Unet을 수정하여 성능 향상을 이루었고  
    FR Unet이 Unet++의 구조를 차용​

    - Feature Aggregation Module : 다양한 kernel size를 사용함으로써  
     다양한 크기의 혈관의 feature를 반영하는 feature map 구성

    - Modified Residual Block : 빠른 over fitting 문제 해소


<p align="center">
  <img src="https://github.com/sujin7822/README/assets/122075306/f6866c30-c45a-4a3d-9bce-cc91972caa6f" alt="Image1" width="300" height="200" />
  <img src="https://github.com/sujin7822/README/assets/122075306/a0e69432-d250-4ed0-9399-667f287be142" alt="Image2" width="600" height="200" />
</p>

### **_Compare with Unet_**
### Quantitative Evaluation
- Test Dataset의 모든 평가지표가 아주 미세하게 상승​
- 논문에서 사용한 test dataset (DRIVE, CHASE_DB1)의 평가지표가 상승
![image](https://github.com/sujin7822/README/assets/122075306/8b211d6b-1fa9-4a24-ae1e-26af3dccdd53)

### Qualitative Evaluation

- 빨간색원을 보면 Unet이 FR Unet보다 미세혈관을 더 잘 표현함
- 파란색원을 비교해 보면 FR Unet이 Unet보다 굵은혈관을 더 잘 표현함

![image](https://github.com/sujin7822/README/assets/122075306/5e937fa7-3289-4ba2-8344-b059f453df40)

<br/>

### **_Wrap up_**
- FR Unet은 Unet보다 미세혈관을 잘 구현하지 못함
     - 프로젝트의 중요 목표(미세혈관 구현)와 상반되는 결과
- FR Unet은 Unet보다 굵은혈관을 잘 구현
- 위 두 특징이 평가지표에서 trade off로 작용하여 FR Unet이 유의미한 성능 차이를 기록하지 못함
<br/>

### 3.5 FR Unet wo DS
- **Model** : FR Unet Without Deep Supervision

![image](https://github.com/sujin7822/README/assets/122075306/8749d447-7dfc-44f7-a169-8573309c7c9f)

   - 다양한 scale로 존재하는 liver나 lung이 ​deep supervision으로  
    더 정확한 segmentation을 가능하게 함
   - 이를 다르게 해석하면 다양한 scale로 존재하지 않는 객체인 경우  
   (논문에서는 cell nuclei 우리 프로젝트에서는 혈관과 대응)
   - 이러한 경우에는 deep supervision 이 성능에 안 좋은 영향을 끼칠 수 있다고 판단
   - 이를 확인하기 위해서 Deep Supervision을 적용하지 않은 FR Unet으로 모델을 학습

<p align="center">
  <img src="https://github.com/sujin7822/README/assets/122075306/8165d63e-54f2-4085-9a9e-006163d736ef" alt="Image1" width="300" height="200" />
  <img src="https://github.com/sujin7822/README/assets/122075306/161e0d0a-5551-45f9-b15f-c35434da9bbf" alt="Image2" width="600" height="200" />
</p>

### **_Compare with Dense Unet_**
### Quantitative Evaluation

- Sensitivity 평가지표 값 대폭 상승​

- 다른 평가지표 값 소폭 하락

![image](https://github.com/sujin7822/README/assets/122075306/27d3ed52-2db4-4717-9123-453af008a0b8)

### Qualitative Evaluation

- 모든 부분에서 FR Unet wo DS가 Unet과 FR Unet w DS에 비해 좋은 성능을 보임

- FR Unet wo DS가 미세혈관을 더욱 잘 잡으면서, 동시에 연속적인 혈관의 특징까지 잘 파악

![image](https://github.com/sujin7822/README/assets/122075306/79ec5449-9b68-42be-b7a5-ad975606784f)

- 빨간색 원을 보면 Deep Supervision을 사용하지 않고 학습된 FR Unet이 Unet처럼 미세혈관을 잘 잡는 다는 것 확인

- 파란색 원을 보면 Deep Supervision을 사용해 학습된 FR Unet이 여전히 굵은 혈관을 잘 잡음

![image](https://github.com/sujin7822/README/assets/122075306/bbbfa918-1e69-43b3-b5a8-b39631f7138d)


<br/>

### **_Wrap up_**
- FR Unet wo DS가 FR Unet w DS에서 미세혈관을 잘 구현하지 못했던 문제를 해결
- FR Unet wo DS는 FR Unet w DS 처럼 굵은혈관을 잘 구현

☛ FR Unet  without Deep Supervision은 우리 프로젝트의 목표인 “혈관 구현력” 을 가장 잘 실현함.
<br/>

## 4. The Final Puzzle Piece we should make 

### **_프로젝트 마지막 종착점_**
> 프로젝트가 가지는 향후 발전 가능성 + 사용성을 제고하는 방법을 고려
- 혈관 관련 질병을 Classification 해주는 모델 개발하여 Grad CAM 적용​

- Segmentation + Classification 모델을 병렬로 구성하여 서비스 배포

![image](https://github.com/sujin7822/README/assets/122075306/07d416a8-b5ab-45bb-a15b-ad7403d99a8a)

<br/>

## 5. 프로젝트 회고

|이름|내용|
|-------|--------| 
|정호재|안녕하세요| 
|김 산|회고 내용을| 
|김수진|적어주세요| 
