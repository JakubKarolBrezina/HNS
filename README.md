# Detekčné algoritmy počítačového videnia

Tento repozitár obsahuje viacero modulov z oblasti počítačového videnia, zameraných najmä na:

- detekciu objektov a osôb  
- detekciu ľudí  
- odhad pózy človeka (pose estimation)

Implementácie využívajú moderné knižnice a frameworky ako **YOLO (Ultralytics)**, **Google MediaPipe**, **HRNet** a **OpenPose**.

Repozitár je určený predovšetkým na vzdelávacie, experimentálne a školské účely. Jednotlivé moduly je možné ďalej rozširovať aj pre praktické použitie.

---

## Prehľad repozitára

Repozitár je rozdelený na samostatné moduly, pričom každý modul rieši konkrétny prístup alebo algoritmus z oblasti počítačového videnia:

- **YOLO** – detekcia objektov a osôb  
- **MediaPipe** – odhad pózy človeka v reálnom čase  
- **HRNet** – presný odhad pózy pomocou hlbokých neurónových sietí  
- **OpenPose** – viac-osobová detekcia pózy a kostry tela  
- **Models** – predtrénované modely používané jednotlivými modulmi  
- **Test** – testovacie a experimentálne súbory  
- **DetAlgo** – dokumentácia, poznámky a projektové materiály  

Každý modul je možné nastaviť a spustiť nezávisle.

---

## Štruktúra repozitára

.
├── .idea/ # Konfiguračné súbory vývojového prostredia (IDE)
├── DetAlgo/ # Dokumentácia a projektové materiály
├── HRNet/ # Modul HRNet – odhad pózy človeka
│ └── HRNet.py
├── mediapipe/ # Modul MediaPipe – odhad pózy človeka
│ └── mediapipe_pose.py
├── models/ # Predtrénované modely
│ └── pose_landmarker_full.task
├── openpose/ # Modul OpenPose – detekcia kostry tela
│ └── openpose.py
├── test/ # Testovacie a experimentálne súbory
├── yolo/ # Modul YOLO – detekcia objektov
│ └── README.md
└── README.md # Hlavný README súbor repozitára


---

## Systémové požiadavky

- Python 3.9 alebo novší  
- pip (správca balíkov pre Python)

### Podporované operačné systémy

- Windows  
- Linux  
- macOS  

---

## Nastavenie prostredia (Setup)

### 1. Overenie inštalácie Pythonu
python --version
Ak Python nie je nainštalovaný, je možné ho stiahnuť z:
https://www.python.org/downloads/

2. Vytvorenie virtuálneho prostredia
python -m venv venv
Aktivácia virtuálneho prostredia
Windows

venv\Scripts\activate
Linux / macOS

source venv/bin/activate
3. Inštalácia potrebných knižníc
pip install ultralytics mediapipe opencv-python numpy requests
V závislosti od konkrétneho modulu môžu byť potrebné aj ďalšie knižnice (pozri README v príslušnom module).

Moduly projektu
YOLO – Detekcia objektov a osôb
Modul YOLO slúži na detekciu objektov a osôb v:

obrázkoch

videách

reálnom čase pomocou webkamery

Umiestnenie modulu:

/yolo
Funkcionalita
detekcia objektov a osôb

spracovanie obrázkov, videí a live streamu

vykreslenie ohraničujúcich boxov

zobrazenie názvu triedy a pravdepodobnosti

Podrobný návod na nastavenie a spustenie sa nachádza v súbore:

yolo/README.md
MediaPipe – Odhad pózy človeka
Modul MediaPipe slúži na detekciu a sledovanie pózy človeka v reálnom čase.

Umiestnenie modulu:

/mediapipe
Funkcionalita
detekcia jednej alebo viacerých osôb

sledovanie 33 bodov ľudského tela

vykreslenie kostry tela

detekcia zdvihnutej ľavej a pravej ruky

odhad vzdialenosti od kamery

odhad výšky osoby

zobrazenie FPS v reálnom čase

HRNet – Odhad pózy človeka
HRNet je pokročilý model pre presný odhad pózy človeka, vhodný najmä na analytické a výskumné účely.

Umiestnenie modulu:

/HRNet
Funkcionalita
presná detekcia kĺbov tela

vysoká stabilita výstupu

vhodné pre statické aj dynamické scény

OpenPose – Detekcia kostry tela
OpenPose umožňuje detekciu pózy viacerých osôb naraz a detailné sledovanie kostry tela.

Umiestnenie modulu:

/openpose
Funkcionalita
viac-osobová detekcia pózy

detekcia kĺbov a končatín

vizualizácia kostry tela

Modely
Priečinok obsahujúci predtrénované modely:

/models
Modely:

sa načítavajú automaticky

nevyžadujú manuálnu konfiguráciu

Testovanie
Priečinok obsahujúci testovacie a experimentálne súbory:

/test
Obsahuje:

testovacie skripty

experimentálne súbory

pomocné testy použité počas vývoja

Poznámky
Jednotlivé moduly je možné spúšťať samostatne.

Modely sa načítavajú alebo sťahujú automaticky.

Presnosť detekcie závisí od:

hardvéru

kvality kamery

svetelných podmienok

Odhady vzdialenosti a výšky osoby sú orientačné.
