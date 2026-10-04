# Drone Target Detection

Gerçek zamanlı dron hedef tespiti için Python tabanlı bir proje.

Özellikler:
- OpenCV tabanlı renk tespiti
- Dairesel hedef tespiti
- İsteğe bağlı YOLO modeli ile nesne tespiti
- Hedef merkez koordinat hesaplama
- Dron için yönlendirme çıktısı

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# veya .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Çalıştırma

### 1) Renk tabanlı hedef tespiti

```bash
python main.py --source 0 --mode color --color red
```

### 2) Dairesel hedef tespiti

```bash
python main.py --source 0 --mode circle
```

### 3) YOLO modeli ile tespit

```bash
python main.py --source 0 --mode yolo --model yolov8n.pt
```

### 4) Video dosyası ile çalıştırma

```bash
python main.py --source footage.mp4 --mode color --color red
```

### 5) Kayıt alarak çalıştırma

```bash
python main.py --source 0 --mode yolo --model yolov8n.pt --output drone_detection_output.mp4
```

## Argümanlar

- `--source`: Kamera numarası veya video dosyası yolu
- `--mode`: `color`, `circle`, `yolo`
- `--color`: `red`, `green`, `blue`, `yellow`, `orange`, `purple`
- `--model`: YOLO modeli yolu (`yolov8n.pt` gibi)
- `--output`: Çıktı video dosyası yolu
- `--conf`: YOLO güven seviyesi (varsayılan: 0.5)
- `--show-debug`: Hata/ayrıntılı içeriği göster

## Dron için çıktı formatı

Program hedef merkezini hesaplar ve aşağıdaki bilgileri ekrana yazdırır:

- Hedef bulundu / bulunamadı
- Hedef merkez noktasını (`cx`, `cy`)
- Görüntü merkezine göre hata değeri (`dx`, `dy`)
- Dron yönlendirme durumu (`centered`, `move left`, `move right`, `move up`, `move down`)

## Proje yapısı

```text
.
├── main.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── detector.py
│   └── utils.py
└── models/
    └── .gitkeep
```

## Örnek sonuç

- Hedef kırmızı renkteyse tespit edilir
- Daire şeklinde hedef bulunursa merkez noktası işaretlenir
- YOLO ile farklı nesneler de tespit edilebilir

## Notlar

- YOLO kullanmak için `ultralytics` kurulmalıdır.
- Gerçek dron uygulamalarında bu çıktıları kontrol algoritması ile birleştirmeniz önerilir.
- Görüntü işleme için iyi ışık koşulları çok önemlidir.

## Lisans

MIT
