## URBAN TECH HAKATON musobaqasiga taqdim etilgan loyiha!
Jamoamiz insonlar vaqrini qadrlagan holda uni tejashlari uchun Innovatsion yechim taklif etdi. Biz savdo do'konlari javonida mavjud narsalarni aniqlash uchun YOLO ning so'nggi versiyasi, ya'ni YOLOv7 ilovasini taqdim etamiz. Ushbu ilovadan oddiygina javondagi buyumlarning rasmlari yordamida inventarizatsiyani kuzatish uchun foydalanish mumkin.

![Result image](https://github.com/shayanalibhatti/Retail-Store-Item-Detection-using-YOLOv5/blob/master/results.png)

## Introduction
Keling, sizga jamoamiz bilan hakaton davamida nimalar qilganimiz haqida batafsil yozaman.
Object Detection (Ob'ektni aniqlash) - bu ob'ektlarni aniqlash, lokalizatsiya qilish va tasniflashni talab qiladigan Computer Vision vazifasi. Ushbu vazifada, avvalo, rasmda biron bir qiziqish ob'ekti mavjudligini aniqlash uchun Machine Learning modelini tuzib oldik. Agar mavjud bo'lsa, rasmda mavjud bo'lgan ob'ekt(lar) atrofida chegara chizig'ini qo'ydik. Natijamiz, model chegara qutisi bilan ifodalangan ob'ektni tasnifladi. Bu vazifa real vaqtda amalga oshirilishi uchun ob'ektni tezkor aniqlashni talab qiladi. Uning asosiy ilovalaridan biri bu o'z-o'zidan boshqariladigan transport vositalarida real vaqt rejimida ob'ektni aniqlashda foydalanishdir.

Iroda Abdurahimova va A'lonur Abdurahimova dastlab real vaqtda obyektni aniqlashni amalga oshiradigan YOLOv1, v2 va v3 modellari ishlab chiqdik. YOLO "Sizning obyektingizga faqat bir marta qaraydi" - bu tasvir va videolarda ob'ektlarni aniqlash, mahalliylashtirish va tasniflash uchun ishlatiladigan real vaqt rejimida chuqur o'rganishning zamonaviy algoritmidir. Ushbu algoritm juda tez, aniq va ob'ektni aniqlashga asoslangan loyihalarda birinchi o'rinda turadi.

YOLO versiyasining har biri avvalgisining aniqlik darajasini yaxshilashda davom etadi. Biz jamoamiz bilan ishlab chiqilgan YOLOv4'ni sinovdan o'tkazdik, bu modelning ishlashini yanada oshirdi va YOLOv5 modeli bilan modelimizni train qilishga qaror qildik. Bu model hajmini sezilarli darajada kamaytiradi (Darknetdagi YOLOv4 244 MB hajmga ega, YOLOv5 eng kichik modeli esa 27 MB). . YOLOv5, shuningdek, Roboflow.ai veb-saytidan olingan quyidagi grafikda ko'rsatilganidek, YOLOv4 ga qaraganda tezroq aniqlik va soniyasiga ko'proq kadrlarni da'vo ko'rsata oldi.
![yolo_vs_detnet](https://github.com/shayanalibhatti/Retail-Store-Item-Detection-using-YOLOv5/blob/master/yolo%20vs%20efficientdet.png)

Fig 1.1: YOLOv5 vs EfficientDetNetni taqqoslash

Biz tovarlarni aniqlash uchun YOLOv5-dan foydalanishga e'tibor qaratdik.
## Objective

YOLOv5-dan SKU110k ma'lumotlar to'plamidan foydalangan holda rasmlardagi chakana mahsulotlar ustidan chegara qutilarini chizish uchun foydalandik.
![Result image](https://github.com/shayanalibhatti/Retail-Store-Item-Detection-using-YOLOv5/blob/master/results.png)

1.2-rasm: Javon tasviri (chapda) va ob'ektlarga cheklov chizmasi chizilgan kerakli natija bilan (o'ngda)

## Ma'lumotlar:

Ushbu vazifani bajarish uchun avval SKU110k tasvir ma'lumotlar to'plamini quyidagi havoladan yuklab oldim:

http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz
SKU110k ma'lumotlar to'plami zich joylashgan muhitda chakana savdo ob'ektlari tasvirlariga asoslangan. U oʻqitish, tekshirish va sinov toʻplami tasvirlarini hamda tegishli .csv fayllarini taqdim etadi, ularda ushbu tasvirlardagi barcha obʼyektlarning chegaralangan joylashuvi uchun maʼlumotlar mavjud. .csv fayllari quyidagi ustunlarda yozilgan obyektni chegaralovchi maydon maʼlumotlariga ega:

image_name,x1,y1,x2,y2,class,image_width,image_height

x1,y1 chegaralovchi qutining yuqori chap koordinatalari va x2,y2 chegaralovchi qutining pastki o'ng koordinatalari, qolgan parametrlar o'z-o'zidan tushunarli. Bir cheklovchi quti uchun train_0.jpg tasviri parametrlariga misol quyida ko'rsatilgan. Har bir tasvir uchun bir nechta chegaralovchi qutilar, har bir ob'ekt uchun bitta quti mavjud.

train_0.jpg, 208, 537, 422, 814, object, 3024, 3024

SKU110k ma'lumotlar to'plamida bizda sinov to'plamida 2940 tasvir, poezdda 8232 tasvir va tasdiqlash to'plamida 587 tasvir mavjud. Har bir rasmda har xil miqdordagi ob'ektlar bo'lishi mumkin, shuning uchun chegaralangan qutilar soni har xil.

## Methodology
Ma'lumotlar to'plamidan men to'plamidan atigi 998 ta rasm oldim va turli formatlarda, shu tarzda YO'llab-quvvatv5 qo'llab-quvvatlovchi formatda onlayn tasvirni uzatish xizmatini taqdim etish Roboflow.ai veb-saytiga bordim. Mashg'ulotlar to'plamidan atigi 998 ta rasmni tuzatishning sababi, Roboflow.a tasvir annotatsiyasi faqat birinchi 1000 ta rasm uchun bepul.

### Preprocessing
Tasvirlarni oldindan qayta ishlash ularning o'lchamlarini 416x416x3 ga o'zgartirishni o'z ichiga oladi. Bu Roboflow platformasida amalga oshiriladi. Izohlangan, o'lchami o'zgartirilgan rasm quyidagi rasmda ko'rsatilgan:

![Annotated image](https://github.com/shayanalibhatti/Retail-Store-Item-Detection-using-YOLOv5/blob/master/roboflow_data_image_annotated.jpg)

Fig 1.3: Roboflow tomonidan izohlangan rasm

### Automatic Annotation
Roboflow.ai veb-saytida chegaralovchi quti izohi .csv fayli va o‘quv to‘plamidagi rasmlar yuklanadi va Roboflow.ai annotatsiya xizmati yuqoridagi rasmda ko‘rsatilganidek .csv fayllarida berilgan izohlar yordamida rasmlarga avtomatik ravishda chegaralovchi qutilarni chizadi.

### Data Generation
Roboflow, shuningdek, foydalanuvchi tomonidan belgilangan bo'linish asosida ma'lumotlar to'plamini yaratish imkoniyatini beradi. Men 70-20-10 ta o'quv-validatsiya-test to'plamidan foydalandim. Roboflow-da ma'lumotlar yaratilgandan so'ng, biz har bir rasm uchun alohida matn faylida barcha izohli ob'ektlar uchun asl tasvirlarni, shuningdek, barcha chegaralangan qutilarni olamiz, bu qulay.
Nihoyat, biz etiketli fayllar bilan yaratilgan ma'lumotlarni yuklab olish uchun havolani olamiz. Bu havolada faqat sizning hisob qaydnomangiz bilan chegaralangan va baham ko‘rilmasligi kerak bo‘lgan kalit mavjud.

### Hardware Used
Model Tesla P100 16 GB grafik kartasi bilan Google Colab Pro noutbukida o'qitildi. Uning narxi $9,99 va bir oylik foydalanish uchun yaxshi. Google Colab noutbukidan ham foydalanish mumkin, u bepul, lekin seans vaqti cheklangan.

## Code
Kod biriktirilgan fayllardagi jupyter daftarida mavjud. Biroq, butun kodni Google Colab daftariga nusxalash tavsiya etiladi.

U dastlab COCO ma'lumotlar to'plami uchun o'qitilgan, lekin men qilgan narsam bo'lgan maxsus vazifalar uchun o'zgartirilishi mumkin. Men YOLOv5 ni klonlash va talablar.txt faylida ko'rsatilgan bog'liqliklarni o'rnatishdan boshladim. Bundan tashqari, model Pytorch uchun yaratilgan, shuning uchun men uni import qildik.

```
!git clone https://github.com/ultralytics/yolov5  # clone repo
!pip install -r yolov5/requirements.txt  # install dependencies
%cd yolov5
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
```

Keyin men Roboflow.ai saytida yaratgan ma'lumotlar to'plamini yuklab olaman. Quyidagi kod trening, test va tekshirish to'plami va izohlarni ham yuklab oladi. Shuningdek, u .yaml faylini yaratadi, unda oʻqitish va tekshirish toʻplami hamda maʼlumotlarimizda qanday sinflar mavjudligi mavjud. Agar siz ma'lumotlar uchun Roboflow dan foydalansangiz, kalitni kodga kiritishni unutmang, chunki u har bir foydalanuvchi uchun noyobdir.

```
# Export code snippet and paste here
%cd /content
!curl -L "ADD THE KEY OBTAINED FROM ROBOFLOW" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

Ushbu fayl modelga o'qitish va tekshirish to'plami tasvirlarining joylashuv yo'lini, shuningdek sinflar soni va sinflar nomlarini aytadi. Ushbu vazifa uchun sinflar soni "1" va sinf nomi "ob'ekt" dir, chunki biz faqat chegaralovchi qutilarni taxmin qilmoqchimiz. data.yaml faylini quyida ko'rish mumkin:
![yaml](https://github.com/shayanalibhatti/Retail-Store-Item-Detection-using-YOLOv5/blob/master/data_yaml.jpg)

### Network Architecture
Keyin YOLOv5 uchun tarmoq arxitekturasini aniqlaymiz. Bu muallif Glenn Jocher tomonidan COCO ma'lumotlar to'plami bo'yicha trening uchun foydalanilgan bir xil arxitektura. Men tarmoqda hech narsani o'zgartirmadim. Biroq, cheklovchi quti o'lchamini, rangini o'zgartirish va teglarni olib tashlash uchun bir nechta sozlash kerak edi, aks holda teglar juda ko'p qutilar tufayli tasvirni aralashtirib yuboradi. Ushbu sozlashlar detect.py va utils.py fayllarida qilingan. Tarmoq custom_yolov5.yaml fayli sifatida saqlanadi.

```
%cd /content/
##write custom model .yaml
#you can configure this based on other YOLOv5 models in the models directory
with open('yolov5/models/custom_yolov5s.yaml', 'w') as f:
  # parameters
  f.write('nc: ' + num_classes + '\n')
  #f.write('nc: ' + str(len(class_labels)) + '\n')
  f.write('depth_multiple: 0.33'  + '\n') # model depth multiple
  f.write('width_multiple: 0.50'  + '\n')  # layer channel multiple
  f.write('\n')
  f.write('anchors:' + '\n')
  f.write('  - [10,13, 16,30, 33,23] ' + '\n')
  f.write('  - [30,61, 62,45, 59,119]' + '\n')
  f.write('  - [116,90, 156,198, 373,326] ' + '\n')
  f.write('\n')

  f.write('backbone:' + '\n')
  f.write('  [[-1, 1, Focus, [64, 3]],' + '\n')
  f.write('   [-1, 1, Conv, [128, 3, 2]],' + '\n')
  f.write('   [-1, 3, Bottleneck, [128]],' + '\n')
  f.write('   [-1, 1, Conv, [256, 3, 2]],' + '\n')
  f.write('   [-1, 9, BottleneckCSP, [256]],' + '\n')
  f.write('   [-1, 1, Conv, [512, 3, 2]], ' + '\n')
  f.write('   [-1, 9, BottleneckCSP, [512]],' + '\n')
  f.write('   [-1, 1, Conv, [1024, 3, 2]],' + '\n')
  f.write('   [-1, 1, SPP, [1024, [5, 9, 13]]],' + '\n')
  f.write('   [-1, 6, BottleneckCSP, [1024]],' + '\n')
  f.write('  ]' + '\n')
  f.write('\n')

  f.write('head:'  + '\n')
  f.write('  [[-1, 3, BottleneckCSP, [1024, False]],'  + '\n')
  f.write('   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1, 0]],' + '\n')
  f.write('   [-2, 1, nn.Upsample, [None, 2, "nearest"]],' + '\n')
  
  f.write('   [[-1, 6], 1, Concat, [1]],' + '\n')
  f.write('   [-1, 1, Conv, [512, 1, 1]],' + '\n')
  f.write('   [-1, 3, BottleneckCSP, [512, False]],' + '\n')
  f.write('   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1, 0]],' + '\n')
  
  f.write('   [-2, 1, nn.Upsample, [None, 2, "nearest"]],' + '\n')
  f.write('   [[-1, 4], 1, Concat, [1]],' + '\n')
  f.write('   [-1, 1, Conv, [256, 1, 1]],' + '\n')
  f.write('   [-1, 3, BottleneckCSP, [256, False]],' + '\n')
  f.write('   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1, 0]],' + '\n')
  f.write('\n' )
  f.write('   [[], 1, Detect, [nc, anchors]],' + '\n')
  f.write('  ]' + '\n')

print('custom model config written!')
```

## Training
Endi men mashg'ulot jarayonini boshlayman. Men tasvir o'lchamini (img) 416x416, partiya hajmi 32 deb belgiladim va model 300 davr uchun ishlaydi. Agar biz og'irliklarni aniqlamasak, ular tasodifiy ravishda ishga tushiriladi.

```
# 300 davr mobaynida (epoch) maxsus ma'lumotlarga yolov5 ni sinadik.
%cd /content/yolov5/
!python train.py --img 416 --batch 32 --epochs 300 --data '../data.yaml' --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results --nosave --cache

```

Google Colab Pro tomonidan taqdim etilgan Tesla P100 16 GB grafik protsessorida mashq bajarish uchun 4 soat 37 daqiqa vaqt ketdi. Trening tugagandan so'ng, modelning vazni Google diskida last_yolov5_results.pt sifatida saqlanadi.

```
from google.colab import drive
drive.mount('/content/gdrive',force_remount=True)
%cp /content/yolov5/weights/last_yolov5s_results.pt /content/gdrive/My\ Drive
```

## Observations
Model quyidagi kod yordamida o'qitilgandan so'ng muhim baholash ko'rsatkichlarini tasavvur qilishimiz mumkin:

```
# Agar tensor taxtasi biron sababga ko'ra ishlamasa, biz eski maktab grafiklarini ham chiqarishimiz mumkin...
from utils.utils import plot_results; plot_results()  # plot results.txt as results.png
Image(filename='./results.png', width=1000)  # view results.png
```

Ob'ektni aniqlash vazifalari uchun odatda quyidagi 3 parametr qo'llaniladi:
· GIoU - Ittifoqning umumiy kesishmasi bo'lib, u bizning chegaraviy qutimizning yerdagi haqiqatga qanchalik yaqin ekanligini ko'rsatadi.
· Ob'ektivlik ob'ektning tasvirda mavjud bo'lish ehtimolini ko'rsatadi. Bu erda u yo'qotish funktsiyasi sifatida ishlatiladi.
· mAP - o'rtacha o'rtacha aniqlik bo'lib, bizning chegaralangan qutidagi bashoratlarimiz qanchalik to'g'ri ekanligini ko'rsatadi. Bu aniqlik egri chizig'i ostidagi maydon.
Ko'rinib turibdiki, umumiy kesishuv (GIoU) ​​yo'qolishi va ob'ektivlik yo'qolishi ham o'qitish, ham tasdiqlash uchun kamayadi. Biroq, o'rtacha o'rtacha aniqlik (mAP) 0,5 bo'lgan IoU chegarasi chegarasi uchun 0,7 da. Quyida ko'rsatilgandek eslab qolish 0,8 da turadi:

![Observations](https://github.com/shayanalibhatti/Retail-Store-Item-Detection-using-YOLOv5/blob/master/observations.png)

1.4-rasm: Model tayyorlashning muhim parametrlarini kuzatish

Endi quyidagi kod yordamida modelimiz sinov to'plami tasvirlarida qanday ishlashini tekshiradigan qism keladi:

```
# biz buni ishga tushirganimizda, biz .007 soniya vaqtini ko'rdik. That is 140 FPS on a TESLA P100!
%cd /content/yolov5/
!python detect.py --weights weights/last_yolov5s_results.pt --img 416 --conf 0.4 --source ../test/images
```

## Results
Quyidagi rasmlarda ob'ektlarga chegaralovchi qutilarni chizishga o'rgatilgan YOLOv5 algoritmimiz natijasi ko'rsatilgan. Natijalar juda yaxshi.

![results](https://github.com/shayanalibhatti/Retail-Store-Item-Detection-using-YOLOv5/blob/master/result1.jpg)

Fig 1.5: Asl sinov toʻplami tasviri (chapda) va YOLOv5 tomonidan chizilgan cheklovchi qutilar (oʻngda)

ESDA OLING: Men biriktirgan model faqat 998 ta rasmga o'rgatilgan. Optimal natijalarga erishish uchun SKU maʼlumotlar toʻplamidagi barcha tasvirlarni oʻrgating.

## Conclusion
Qarama-qarshiliklardan tashqari, YOLOv5 yaxshi ishlaydi va bizning ehtiyojlarimizga mos ravishda sozlanishi mumkin. Biroq, modelni o'rgatish sezilarli GPU quvvati va vaqtini talab qilishi mumkin. Katta ma'lumotlar to'plamini o'qitish jarayonini tezlashtirish uchun kamida 16 GB GPU yoki afzalroq TPU bilan Google Colab-dan foydalanish tavsiya etiladi.

Ushbu chakana ob'ekt detektori ilovasidan do'kon javonlari inventarini kuzatish yoki odamlar narsalarni tanlab olish va buning uchun avtomatik ravishda to'lov olish uchun aqlli do'kon kontseptsiyasi uchun foydalanish mumkin. YOLOv5 ning kichik vaznli oʻlchami va yaxshi kadrlar tezligi oʻrnatilgan tizimga asoslangan real vaqtda obyektni aniqlash vazifalari uchun birinchi tanlov boʻlishga yoʻl ochadi.

Bundan tashqari, mening ishimni eslatib o'ting va agar siz tadqiqotda yoki biron bir maqolada foydalansangiz, kredit bering. Bu ishlab chiquvchilarni o'z ishlarini dunyo bilan baham ko'rishga undaydi.
