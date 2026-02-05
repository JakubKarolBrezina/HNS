# Detekčné algoritmy počítačového videnia

Tento repozitár obsahuje viacero projektov z oblasti **počítačového videnia**,
zameraných na **detekciu objektov, detekciu ľudí a odhad pózy človeka**.
Implementácie využívajú moderné knižnice **YOLO (Ultralytics)** a **Google MediaPipe**.

Repozitár je určený na **vzdelávacie, experimentálne a školské účely**.

---

## Prehľad repozitára

Repozitár je rozdelený na niekoľko samostatných modulov, pričom každý rieši
konkrétnu úlohu z oblasti počítačového videnia:

- **YOLO – detekcia objektov a osôb**
- **MediaPipe – odhad pózy človeka**
- **Models – predtrénované modely**
- **Test – testovacie a experimentálne súbory**
- **Dokumentácia – podporné materiály k projektu**

Každý modul je možné nastaviť a spustiť samostatne.

---

## Štruktúra repozitára

```text
.
├── DetAlgo/                   # Dokumentácia a projektové materiály
├── mediapipe/                 # Modul MediaPipe – odhad pózy človeka
│   └── README.md              # Detailný návod pre MediaPipe
├── yolo/                      # Modul YOLO – detekcia objektov
│   └── README.md              # Detailný návod pre YOLO
├── models/                    # Predtrénované modely
├── test/                      # Testovacie a experimentálne súbory
├── pose_landmarker_full.task  # Model MediaPipe Pose Landmarker
└── README.md                  # Hlavný README súbor repozitára
Systémové požiadavky
Softvér
Python 3.9 alebo novší

pip (správca balíkov pre Python)

Podporované platformy
Windows

Linux

macOS

Globálny postup nastavenia (Setup)
1. Overenie inštalácie Pythonu
python --version
Ak Python nie je nainštalovaný, stiahni ho z:
https://www.python.org/downloads/

2. Vytvorenie virtuálneho prostredia (odporúčané)
python -m venv venv
Aktivácia virtuálneho prostredia:

Windows

venv\Scripts\activate
Linux / macOS

source venv/bin/activate
3. Inštalácia potrebných knižníc
Na inštaláciu všetkých knižníc používaných v projekte:

pip install ultralytics mediapipe opencv-python numpy requests
Moduly projektu
1. YOLO – Detekcia objektov
Modul YOLO slúži na detekciu objektov a ľudí v obrazoch, videách
alebo v reálnom čase pomocou webkamery.

Umiestnenie:

/yolo
Funkcionalita:

detekcia objektov a osôb

spracovanie obrázkov, videí a webkamery

ohraničujúce boxy s názvom triedy a pravdepodobnosťou

Podrobný návod na nastavenie a spustenie sa nachádza v súbore:

yolo/README.md
2. MediaPipe – Odhad pózy človeka
Modul MediaPipe slúži na detekciu a sledovanie pózy človeka v reálnom čase.

Umiestnenie:

/mediapipe
Funkcionalita:

detekcia viacerých osôb

sledovanie 33 bodov tela

vykreslenie kostry tela

detekcia zdvihnutej ľavej a pravej ruky

odhad vzdialenosti od kamery

odhad výšky osoby

zobrazenie FPS v reálnom čase

Podrobný návod sa nachádza v súbore:

mediapipe/README.md
Modely
Priečinok models obsahuje predtrénované modely, ktoré sú využívané
jednotlivými modulmi.

/models
modely sa načítavajú automaticky

nie je potrebná manuálna konfigurácia

Testovanie
Priečinok test obsahuje testovacie a experimentálne súbory použité
počas vývoja projektu.

/test
- 📚 **akademickejšiu verziu (bakalárka / semestrálka)**  
- 🧹 prečistiť názvy priečinkov a README konzistenciu  
- 🧾 doplniť **ciele projektu alebo zadanie**

stačí povedať – toto máš už fakt veľmi dobre spravené 💪
