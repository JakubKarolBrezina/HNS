# YOLO – Detekcia objektov

Tento projekt demonštruje detekciu objektov pomocou algoritmu
**YOLO (You Only Look Once)** s využitím predtrénovaných modelov poskytovaných
knižnicou **Ultralytics**.  
YOLO je systém na detekciu objektov v reálnom čase, ktorý dokáže rozpoznať
viaceré objekty v obrázkoch, videách alebo v živom video streame z kamery
bez potreby trénovania vlastného modelu.

---

## Použité technológie

Projekt je implementovaný v jazyku Python a využíva nasledujúce knižnice:

- Python **3.9 alebo novší**
- ultralytics
- opencv-python
- numpy

Správca balíkov: **pip**

---

## Nastavenie projektu (Setup)

### Klonovanie repozitára

Najskôr je potrebné naklonovať repozitár a prejsť do jeho adresára:

```bash
git clone https://github.com/your-username/yolo-object-detection.git
cd yolo-object-detection
Vytvorenie virtuálneho prostredia (odporúčané)
Na predchádzanie konfliktom medzi knižnicami sa odporúča vytvoriť
virtuálne prostredie:

python -m venv venv
Aktivácia virtuálneho prostredia:

Windows:

venv\Scripts\activate
macOS / Linux:

source venv/bin/activate
Inštalácia závislostí
Po aktivácii virtuálneho prostredia nainštaluj potrebné knižnice:

pip install --upgrade pip
pip install ultralytics opencv-python numpy
Spustenie detekcie objektov
Po úspešnej inštalácii je projekt pripravený na spustenie.
Detekciu objektov je možné vykonať nad obrázkom, videom alebo v reálnom čase
pomocou webkamery.

Detekcia objektov na obrázku
python detect.py --source data/image.jpg
Detekcia objektov vo videu
python detect.py --source data/video.mp4
Detekcia objektov v reálnom čase (webkamera)
python detect.py --source 0
Pri prvom spustení sa automaticky stiahne predtrénovaný YOLO model,
čo môže trvať niekoľko sekúnd v závislosti od rýchlosti internetového pripojenia.

Očakávaný výstup
Počas behu programu sú detegované objekty zobrazované pomocou:

ohraničujúcich boxov (bounding boxes)

názvov tried objektov (napr. osoba, auto, zviera)

hodnôt pravdepodobnosti (confidence score)

Pri použití webkamery sa výsledky zobrazujú v reálnom čase v samostatnom okne.
Program je možné ukončiť stlačením klávesy Q.

Uloženie výsledkov
Výsledky detekcie sa automaticky ukladajú do adresára:

runs/detect/
Každé spustenie vytvorí nový podpriečinok (napr. exp, exp2, …),
ktorý obsahuje spracované obrázky alebo videá s vyznačenými objektmi.

Podporované formáty
Obrázky: JPG, PNG

Videá: MP4, AVI

Projekt využíva predtrénované YOLO modely, ako napríklad yolov8n.pt
alebo yolov8s.pt, a nevyžaduje žiadne dodatočné trénovanie.

