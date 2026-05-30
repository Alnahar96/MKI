# Execution Checklist

Schrittweise Abarbeitung vor Abgabe.

## 1. Umgebung vorbereiten
- [ ] Virtuelle Umgebung anlegen: `python -m venv .venv`
- [ ] Aktivieren: `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
- [ ] Pakete installieren: `pip install -r requirements.txt`

## 2. Notebooks ausführen
- [ ] `notebooks/01_data_mining.ipynb`
- [ ] `notebooks/02_machine_learning.ipynb`
- [ ] `notebooks/03_deep_learning.ipynb`
- [ ] `notebooks/04_ki_agents.ipynb`
- [ ] `notebooks/05_xai.ipynb`

## 3. Belege sichern
- [ ] Screenshots in `screenshots/<thema>/` ablegen
- [ ] Ergebnis-Artefakte in `results/{figures,metrics,models}/` ablegen
- [ ] Reflexionen in `reflection/0X_<thema>_reflection.md` ausfüllen

## 4. LaTeX kompilieren
- [ ] `cd docs`
- [ ] `pdflatex main.tex`
- [ ] `biber main`
- [ ] `pdflatex main.tex`
- [ ] `pdflatex main.tex`
- [ ] PDF auf vollständige Querverweise / Literatur prüfen

## 5. Qualität & Integrität
- [ ] `ai_usage_log.md` vollständig gepflegt
- [ ] Keine erfundenen Ergebnisse, keine fremden Inhalte ohne Quelle
- [ ] Alle `[TODO]`-Marker abgearbeitet
- [ ] Repository pushen / ZIP erzeugen
