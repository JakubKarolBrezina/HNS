Detekčné algoritmy počítačového videnia

Tento repozitár obsahuje implementácie moderných algoritmov počítačového videnia so zameraním na:

detekciu objektov a osôb

detekciu ľudí

odhad pózy človeka (pose estimation)

Projekt je určený primárne na vzdelávacie, experimentálne a školské účely. Jednotlivé moduly je však možné ďalej rozširovať a adaptovať pre praktické použitie.

Použité technológie

YOLO (Ultralytics)

Google MediaPipe

HRNet

OpenPose

Python 3.9+

OpenCV

NumPy

Štruktúra repozitára
.
├── .idea/                     # Konfiguračné súbory vývojového prostredia
├── DetAlgo/                   # Dokumentácia a projektové materiály
├── HRNet/                     # Modul HRNet – odhad pózy človeka
│   └── HRNet.py
├── mediapipe/                 # Modul MediaPipe – odhad pózy
│   └── mediapipe_pose.py
├── models/                    # Predtrénované modely
│   └── pose_landmarker_full.task
├── openpose/                  # Modul OpenPose – detekcia kostry tela
│   └── openpose.py
├── test/                      # Testovacie a experimentálne súbory
├── yolo/                      # Modul YOLO – detekcia objektov
│   └── README.md
└── README.md                  # Hlavná dokumentácia

Systémové požiadavky

Python 3.9 alebo novší

pip (správca balíkov pre Python)

Podporované operačné systémy:

Windows

Linux

macOS

Nastavenie prostredia
1. Overenie inštalácie Pythonu
python --version


Ak Python nie je nainštalovaný, je možné ho stiahnuť z:
https://www.python.org/downloads/

2. Vytvorenie virtuálneho prostredia
python -m venv venv


Aktivácia virtuálneho prostredia:

Windows:

venv\Scripts\activate


Linux / macOS:

source venv/bin/activate

3. Inštalácia závislostí
pip install ultralytics mediapipe opencv-python numpy requests


V závislosti od konkrétneho modulu môžu byť potrebné ďalšie knižnice. Podrobnosti sú uvedené v README jednotlivých modulov.

Prehľad modulov
YOLO – Detekcia objektov a osôb

Umiestnenie: /yolo

Funkcionalita:

detekcia objektov a osôb

spracovanie obrázkov

spracovanie videí

real-time detekcia pomocou webkamery

vykreslenie ohraničujúcich boxov

zobrazenie názvu triedy a pravdepodobnosti

Podrobný návod na nastavenie a spustenie sa nachádza v súbore yolo/README.md.

MediaPipe – Odhad pózy človeka

Umiestnenie: /mediapipe

Funkcionalita:

detekcia jednej alebo viacerých osôb

sledovanie 33 bodov ľudského tela

vizualizácia kostry tela

detekcia zdvihnutej ľavej a pravej ruky

orientačný odhad vzdialenosti od kamery

orientačný odhad výšky osoby

zobrazenie FPS v reálnom čase

HRNet – Presný odhad pózy človeka

Umiestnenie: /HRNet

Funkcionalita:

presná detekcia kĺbov tela

vysoká stabilita výstupu

vhodné pre analytické a výskumné účely

použiteľné pre statické aj dynamické scény

OpenPose – Viac-osobová detekcia pózy

Umiestnenie: /openpose

Funkcionalita:

detekcia viacerých osôb naraz

sledovanie kĺbov a končatín

vizualizácia kostry tela

Modely

Priečinok /models obsahuje predtrénované modely používané jednotlivými modulmi.

Modely:

sa načítavajú automaticky,

nevyžadujú manuálnu konfiguráciu.

Testovanie

Priečinok /test obsahuje:

testovacie skripty,

experimentálne súbory,

pomocné testy použité počas vývoja.

Poznámky

Jednotlivé moduly je možné spúšťať samostatne.

Modely sa načítavajú alebo sťahujú automaticky.

Presnosť detekcie závisí od hardvéru, kvality kamery a svetelných podmienok.

Odhady vzdialenosti a výšky osoby sú orientačné.

