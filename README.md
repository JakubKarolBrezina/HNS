# Detekčné algoritmy počítačového videnia

Tento repozitár obsahuje viacero projektov z oblasti počítačového videnia,
zameraných na detekciu objektov, detekciu ľudí a odhad pózy človeka.
Implementácie využívajú knižnice YOLO (Ultralytics) a Google MediaPipe.

Repozitár je určený na vzdelávacie, experimentálne a školské účely.

---

## Prehľad repozitára

Repozitár je rozdelený na samostatné moduly, pričom každý modul rieši konkrétnu
úlohu z oblasti počítačového videnia:

- YOLO – detekcia objektov a osôb
- MediaPipe – odhad pózy človeka
- Models – predtrénované modely
- Test – testovacie a experimentálne súbory
- Dokumentácia – podporné materiály k projektu

Každý modul je možné nastaviť a spustiť nezávisle.

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
Python 3.9 alebo novší

pip (správca balíkov pre Python)

Podporované operačné systémy:

Windows

Linux

macOS

Nastavenie prostredia (Setup)
Overenie inštalácie Pythonu
python --version
V prípade, že Python nie je nainštalovaný, je možné ho stiahnuť z:
https://www.python.org/downloads/

Vytvorenie virtuálneho prostredia
python -m venv venv
Aktivácia virtuálneho prostredia:

Windows:

venv\Scripts\activate
Linux / macOS:

source venv/bin/activate
Inštalácia potrebných knižníc
pip install ultralytics mediapipe opencv-python numpy requests
Moduly projektu
YOLO – Detekcia objektov
Modul YOLO slúži na detekciu objektov a osôb v obrazoch, videách alebo
v reálnom čase pomocou webkamery.

Umiestnenie modulu:

/yolo
Funkcionalita:

detekcia objektov a osôb

spracovanie obrázkov, videí a webkamery

ohraničujúce boxy s názvom triedy a pravdepodobnosťou

Podrobný návod na nastavenie a spustenie sa nachádza v súbore:

yolo/README.md
MediaPipe – Odhad pózy človeka
Modul MediaPipe slúži na detekciu a sledovanie pózy človeka v reálnom čase.

Umiestnenie modulu:

/mediapipe
Funkcionalita:

detekcia viacerých osôb

sledovanie 33 bodov ľudského tela

vykreslenie kostry tela

detekcia zdvihnutej ľavej a pravej ruky

odhad vzdialenosti od kamery

odhad výšky osoby

zobrazenie FPS v reálnom čase

Podrobný návod sa nachádza v súbore:

mediapipe/README.md
Modely
Priečinok models obsahuje predtrénované modely používané jednotlivými modulmi.

/models
Modely sa načítavajú automaticky a nevyžadujú manuálnu konfiguráciu.

Testovanie
Priečinok test obsahuje testovacie a experimentálne súbory použité počas vývoja.

/test
Poznámky
Jednotlivé moduly je možné spúšťať samostatne

Modely sa načítavajú alebo sťahujú automaticky

Presnosť detekcie závisí od hardvéru, kamery a svetelných podmienok

Odhady vzdialenosti a výšky sú orientačné



