# MKI ePortfolio – Bahaa Alnahar

ePortfolio zum Modul **Methoden der Künstlichen Intelligenz** (MKI) an der Hochschule Augsburg, Studiengang *International Information Systems*.

- **Autor:** Bahaa Alnahar
- **Matrikel-Nr.:** 2117485
- **Modul:** Methoden der Künstlichen Intelligenz

## Themenschwerpunkte

1. Data Mining
2. Machine Learning
3. Deep Learning
4. KI-Agents
5. Explainable AI (XAI)

## Struktur

```
docs/         LaTeX-Dokumentation (main.tex, Kapitel, Literatur)
code/         Python-Quellcode je Themenbereich
notebooks/    Jupyter-Notebooks je Themenbereich
data/         Roh- und aufbereitete Daten
screenshots/  Belegende Konsolen- und Plot-Screenshots
results/      Ergebnis-Artefakte (Plots, Metriken, Modelle)
reflection/   Reflexions-Notizen je Themenbereich
```

## Schnellstart

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

LaTeX-Kompilierung siehe `execution_checklist.md`.

## Akademische Integrität

Alle Experimente werden eigenständig ausgeführt; alle KI-Werkzeug-Nutzungen sind in `ai_usage_log.md` transparent dokumentiert.
