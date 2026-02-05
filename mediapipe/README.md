# MediaPipe – Odhad pózy človeka

Tento projekt realizuje odhad pózy človeka v reálnom čase pomocou
**Google MediaPipe Pose Landmarker**.
Aplikácia deteguje a sleduje body ľudského tela zo vstupu z webkamery
alebo z video súboru.

---

## Popis projektu

Aplikácia v reálnom čase analyzuje obrazový vstup a vykonáva detekciu
pózy človeka. Na základe detegovaných bodov tela vizualizuje
kostru celého tela a poskytuje základné informácie o polohe osoby,
ako je približná vzdialenosť od kamery, odhad výšky osoby
a detekcia zdvihnutých rúk.

---

## Funkcionalita

Aplikácia umožňuje:
- detekciu jednej alebo viacerých osôb v obraze
- sledovanie 33 bodov ľudského tela
- vizualizáciu celej kostry tela v reálnom čase
- detekciu zdvihnutej ľavej a pravej ruky
- približný odhad vzdialenosti osoby od kamery
- približný odhad výšky osoby
- zobrazenie FPS v reálnom čase a priemerného FPS
- fungovanie aj pri zhoršených svetelných podmienkach

---

## Systémové požiadavky

### Softvér
- Python **3.9 alebo novší**
- pip (správca balíkov pre Python)

### Použité Python knižnice
- mediapipe
- opencv-python
- numpy
- requests

---

## Nastavenie projektu (Setup)

### Overenie inštalácie Pythonu

```bash
python --version
V prípade, že Python nie je nainštalovaný, je možné ho stiahnuť z:
https://www.python.org/downloads/

Vytvorenie virtuálneho prostredia (odporúčané)
python -m venv venv
Aktivácia virtuálneho prostredia:

Windows:

venv\Scripts\activate
macOS / Linux:

source venv/bin/activate
Inštalácia závislostí
Po aktivácii virtuálneho prostredia nainštaluj potrebné knižnice:

pip install --upgrade pip
pip install mediapipe opencv-python numpy requests
Štruktúra projektu
.
├── pose.py            # Hlavný skript aplikácie
├── video.mp4          # Voliteľný vstupný video súbor
└── README.md
Spustenie aplikácie
Spustenie pomocou webkamery
python pose.py
použije sa predvolená kamera (ID 0)

spustí sa odhad pózy v reálnom čase

Spustenie nad video súborom
python pose.py --source video.mp4
video je spracované po jednotlivých snímkach

výsledky detekcie sú zobrazované priamo vo videu

Očakávaný výstup
Po spustení aplikácie sa otvorí okno zobrazujúce spracovaný video vstup.

Vizuálny výstup:
zobrazené body ľudského tela (landmarky)

spojnice medzi bodmi tvoriace kostru tela

zvýraznenie zdvihnutej ľavej a pravej ruky

textové informácie na obrazovke:

odhad vzdialenosti od kamery

odhad výšky osoby

aktuálne FPS

priemerné FPS

Správanie aplikácie:
podpora detekcie viacerých osôb naraz

beh v reálnom čase v závislosti od výkonu hardvéru

ukončenie aplikácie zatvorením okna alebo stlačením klávesy Q

Obmedzenia
odhad vzdialenosti a výšky je orientačný

neprebieha kalibrácia kamery

presnosť závisí od uhla kamery a viditeľnosti celého tela

výkon závisí od hardvéru a svetelných podmienok

Poznámky
model MediaPipe Pose sa načítava automaticky

nie je potrebná manuálna konfigurácia modelu

najlepšie výsledky sa dosahujú pri viditeľnosti celého tela



