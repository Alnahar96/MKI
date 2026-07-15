# KI-Nutzungs-Log

Transparente Dokumentation der im Rahmen des ePortfolios eingesetzten KI-Werkzeuge.

Die KI wurde als Lernhilfe, zur technischen Unterstützung, zur Fehleranalyse und
zur Qualitätskontrolle eingesetzt. Alle Programme, Workflows und Experimente wurden
von mir lokal ausgeführt, überprüft, angepasst und fachlich eingeordnet.

---

## 1. Data-Mining-Experiment mit dem Iris-Datensatz

**Tool:** ChatGPT  
**Datum:** 30.05.2026  

**Prompt:**  
Erstelle eine verständliche Grundstruktur für ein Python-Experiment mit dem
Iris-Datensatz. Erkläre die einzelnen Verarbeitungsschritte, die verwendeten
Bibliotheken und mögliche Ursachen, falls die Visualisierung nicht korrekt
gespeichert wird.

**Verwendung:**  
ChatGPT wurde zur Strukturierung des Python-Skripts, zur Erklärung der darin
enthaltenen Verarbeitungsschritte und zur Untersuchung des Problems beim
Speichern der Visualisierung verwendet.

**Eigene Leistung:**  
Ich habe das Skript in Visual Studio Code angelegt und lokal ausgeführt. Dabei
habe ich die Dateipfade an meine Projektstruktur angepasst, die benötigten
Bibliotheken überprüft und die Ausgaben kontrolliert. Die erzeugte Grafik habe ich
gespeichert, visuell geprüft und anschließend in mein ePortfolio eingebunden.

**Reflexion:**  
Die Unterstützung erleichterte mir das Verständnis des Skriptaufbaus. Durch die
lokale Ausführung wurde deutlich, dass Code und Dateipfade an die eigene
Entwicklungsumgebung angepasst und die erzeugten Ergebnisse kontrolliert werden
müssen.

---

## 2. Aufgabe 3: LaTeX-Aufbereitung der Beispieldaten

**Tool:** ChatGPT  
**Datum:** 03.06.2026  

**Prompt:**  
Überführe die in Aufgabe 3 vorgegebene Tabelle mit den Merkmalen Feuchte, Säure,
Temperatur und Wachstum in eine übersichtliche LaTeX-Tabelle. Übernimm die
vorhandenen Werte unverändert und verwende eine für ein wissenschaftliches
ePortfolio geeignete Beschriftung.

**Verwendung:**  
ChatGPT wurde zur Übertragung der vorgegebenen Tabelle in LaTeX und zur
Formulierung einer passenden Tabellenbeschriftung eingesetzt.

**Eigene Leistung:**  
Ich habe die übertragenen Werte mit der ursprünglichen Aufgabenstellung
verglichen, die Tabelle in mein LaTeX-Dokument eingefügt und das Ergebnis nach
der Kompilierung überprüft. Die Position der Tabelle sowie ihre Einordnung in den
zugehörigen Abschnitt habe ich selbst festgelegt.

**Reflexion:**  
Die LaTeX-Formatierung konnte dadurch schneller umgesetzt werden. Trotzdem war
die Kontrolle der Werte notwendig, da Übertragungsfehler die weiteren
Berechnungen beeinflussen könnten.

---

## 3. Aufgabe 3: Erklärung der euklidischen Distanz

**Tool:** ChatGPT  
**Datum:** 03.06.2026  

**Prompt:**  
Erkläre die Berechnung der euklidischen Distanz zwischen zwei Datenpunkten mit
mehreren Merkmalen. Zeige die Formel schrittweise und erläutere, wie sie zur
Zuordnung eines Datenpunktes zum nächstgelegenen Clusterzentrum verwendet wird.

**Verwendung:**  
ChatGPT wurde zur Erklärung der Formel, ihrer einzelnen Rechenschritte und ihrer
Anwendung bei der Zuordnung zu einem Clusterzentrum verwendet.

**Eigene Leistung:**  
Ich habe die erklärte Formel auf die Daten aus Aufgabe 3 übertragen und die
Distanzberechnung nachvollzogen. Danach habe ich die mathematische Darstellung in
LaTeX in mein ePortfolio aufgenommen und mit dem k-Means-Verfahren verknüpft.

**Reflexion:**  
Die schrittweise Erklärung half mir zu verstehen, warum ein Datenpunkt dem
Clusterzentrum mit der kleinsten Distanz zugeordnet wird. Gleichzeitig wurde
deutlich, dass unterschiedlich skalierte Merkmale die Distanz beeinflussen
können.

---

## 4. Aufgabe 3: Reproduzierbare Berechnung von k-Means und k-Nächste-Nachbarn

**Tool:** ChatGPT  
**Datum:** 07.06.2026  

**Prompt:**  
Verbessere den Python-Code und berücksichtige die Datenvorbereitung, die
numerische Kodierung, das k-Means-Clustering und die Zuordnung der zusätzlichen
Datenpunkte 11 und 16 mit dem k-Nächste-Nachbarn-Verfahren. Gib auch die
relevanten Zwischenergebnisse aus.

**Verwendung:**  
ChatGPT wurde zur Verbesserung des vorhandenen Python-Codes und zur Ergänzung der
im Prompt genannten Verarbeitungsschritte verwendet.

**Eigene Leistung:**  
Ich habe den überarbeiteten Code lokal ausgeführt und die ausgegebenen
Zwischenergebnisse kontrolliert. Dabei habe ich die numerische Kodierung, die
Clusterzuordnungen und die Ergebnisse für die Datenpunkte 11 und 16 mit der
Aufgabenstellung abgeglichen. Die Ergebnisse habe ich anschließend in meinem
ePortfolio dargestellt und interpretiert.

**Reflexion:**  
Bei der Kontrolle wurde deutlich, dass die numerische Kodierung kategorialer
Werte eine methodische Annahme darstellt. Dadurch können Abstände entstehen, die
nicht zwingend der ursprünglichen Bedeutung der Kategorien entsprechen.

---

## 5. Machine-Learning-Workflow mit KNIME

**Tool:** ChatGPT  
**Zeitraum:** Juni 2026  

**Prompt:**  
Ordne die benötigten Nodes für Datenimport, Partitionierung, Training,
Vorhersage und Bewertung in der richtigen Reihenfolge ein.

**Verwendung:**  
ChatGPT wurde verwendet, um die für den Workflow benötigten KNIME-Nodes den
einzelnen Verarbeitungsschritten zuzuordnen und in eine sinnvolle Reihenfolge zu
bringen.

**Eigene Leistung:**  
Ich habe den Workflow selbst in KNIME aufgebaut, die Nodes eingefügt und
miteinander verbunden. Anschließend habe ich die CSV-Datei eingelesen, die Daten
in Trainings- und Testdaten aufgeteilt, den Decision Tree trainiert, Vorhersagen
erzeugt und die Ergebnisse mit dem Scorer ausgewertet. Die Einstellungen der
Nodes und die erzeugten Ausgaben habe ich selbst kontrolliert und dokumentiert.

**Reflexion:**  
Die Zuordnung der Nodes half mir, den Ablauf eines vollständigen
Machine-Learning-Prozesses in KNIME zu verstehen. Die korrekte Konfiguration und
Ausführung des Workflows musste jedoch von mir selbst vorgenommen und überprüft
werden.

---

## 6. Vergleich von KNIME und einem neuronalen Modell mit dem Iris-Datensatz

**Tool:** ChatGPT  
**Datum:** 30.06.2026  

**Prompt:**  
Zeige eine kompakte Python-Umsetzung eines Multi-Layer-Perceptrons für den
Iris-Datensatz, das mit dem zuvor in KNIME trainierten Decision Tree verglichen
werden kann.

**Verwendung:**  
ChatGPT wurde zur Bereitstellung und Verbesserung einer kompakten Code-Struktur
für das Multi-Layer-Perceptron verwendet.

**Eigene Leistung:**  
Ich habe den Python-Code lokal eingerichtet und ausgeführt. Die Datenaufteilung,
das Training und die ausgegebenen Ergebnisse habe ich kontrolliert. Anschließend
habe ich die Resultate des neuronalen Modells mit dem zuvor in KNIME erstellten
Decision Tree verglichen und die Unterschiede in meinem ePortfolio beschrieben.

**Reflexion:**  
Der Vergleich zeigte mir, dass die Modelle nicht nur anhand ihrer Genauigkeit,
sondern auch hinsichtlich ihrer Struktur und Nachvollziehbarkeit betrachtet
werden sollten.

---

## 7. Planung und Strukturierung der IMDB-Sentimentanalyse

**Tool:** ChatGPT  
**Datum:** 06.07.2026  

**Prompt:**  
Strukturiere ein Deep-Learning-Projekt zur Sentimentanalyse mit dem
IMDB-Datensatz in mehrere klar getrennte Python-Skripte. Vorgesehen sind
Datenanalyse, Konfiguration, Sequenzvorbereitung, TF-IDF-Baseline,
Embedding-Modell, Auswertung, Modellvergleich und Experimente. Achte darauf, dass
alle Skripte dieselben Konfigurationswerte und Datenaufteilungen verwenden.

**Verwendung:**  
ChatGPT wurde zur Aufteilung des Deep-Learning-Projekts in getrennte Skripte und
zur Abstimmung ihrer gemeinsamen Konfiguration verwendet.

**Eigene Leistung:**  
Ich habe die Ordner und Dateien in meinem lokalen Repository angelegt, die
vorgeschlagene Struktur an mein Projekt angepasst und jedes Skript einzeln
ausgeführt. Die erzeugten Dateien in den Ausgabeordnern habe ich kontrolliert und
die für das ePortfolio relevanten Ergebnisse und Screenshots ausgewählt.

**Reflexion:**  
Die Trennung der Aufgaben in mehrere Skripte machte das Projekt übersichtlicher.
Für vergleichbare Ergebnisse musste ich jedoch selbst darauf achten, dass alle
Skripte dieselben Einstellungen und Datenaufteilungen verwendeten.

---

## 8. Erklärung der Deep-Learning-Konfiguration

**Tool:** ChatGPT  
**Zeitraum:** Juli 2026  

**Prompt:**  
Erkläre die Parameter `RANDOM_SEED`, `VOCAB_SIZE`, `MAX_SEQUENCE_LENGTH`,
`EMBEDDING_DIM`, `BATCH_SIZE`, `EPOCHS` und `DROPOUT_RATE` im Kontext einer
IMDB-Sentimentanalyse.

**Verwendung:**  
ChatGPT wurde zur Erklärung der Bedeutung der im Prompt genannten
Konfigurationsparameter verwendet.

**Eigene Leistung:**  
Ich habe die Parameter in meiner Konfigurationsdatei verwendet und überprüft, an
welchen Stellen sie in den einzelnen Skripten eingesetzt werden. Die Erklärungen
habe ich auf mein eigenes Modell und meine Experimente bezogen und anschließend
in meinem Deep-Learning-Abschnitt dargestellt.

**Reflexion:**  
Dadurch wurde deutlich, dass die Parameter verschiedene Bereiche des Projekts
beeinflussen, beispielsweise die Reproduzierbarkeit, die Eingabedaten, das
Training und die Modellkomplexität.

---

## 9. Datenvorbereitung und Behebung des Importfehlers

**Tool:** ChatGPT  
**Datum:** 06.07.2026  

**Prompt:**  
Überprüfe den Importfehler in `prepare_sequences.py`, der bei
`tensorflow.keras.preprocessing.sequence` beziehungsweise `pad_sequences`
auftritt. Erkläre eine versionskompatible Importmöglichkeit sowie die Funktion
von Padding und Truncation bei unterschiedlich langen IMDB-Rezensionen.

**Verwendung:**  
ChatGPT wurde zur Analyse des konkreten Importfehlers und zur Erklärung einer
passenden Importmöglichkeit sowie von Padding und Truncation verwendet.

**Eigene Leistung:**  
Ich habe die TensorFlow-Version in meiner virtuellen Umgebung geprüft, den Import
im Skript angepasst und das Programm erneut ausgeführt. Danach habe ich
kontrolliert, ob die Sequenzen erfolgreich verarbeitet und die vorbereiteten
Daten korrekt gespeichert wurden.

**Reflexion:**  
Die Fehlerbehebung zeigte, dass ein Import von der installierten
Bibliotheksversion abhängen kann. Die vorgeschlagene Änderung musste deshalb in
meiner lokalen Umgebung getestet werden.

---

## 10. Vergleich von TF-IDF-Baseline und Embedding-Modell

**Tool:** ChatGPT  
**Datum:** 08.07.2026  

**Prompt:**  
Strukturiere einen fairen Vergleich zwischen einer TF-IDF-Darstellung mit
logistischer Regression und einem neuronalen Modell mit trainierbarer
Embedding-Schicht.

**Verwendung:**  
ChatGPT wurde zur Strukturierung des Vergleichs zwischen den beiden
Modellansätzen verwendet.

**Eigene Leistung:**  
Ich habe beide Modelle lokal trainiert und ihre Ergebnisse auf derselben
Aufgabenstellung ausgewertet. Die erzeugten Metriken, Konfusionsmatrizen und
Trainingsausgaben habe ich kontrolliert und miteinander verglichen. Die
Ergebnisdarstellung und die Interpretation im ePortfolio habe ich auf Grundlage
meiner tatsächlich erzeugten Resultate vorgenommen.

**Reflexion:**  
Der Vergleich zeigte, dass die klassische TF-IDF-Baseline in meinem Experiment
besser abschnitt als das einfache Embedding-Modell. Dadurch wurde deutlich, dass
ein neuronales Modell nicht automatisch die bessere Lösung darstellt.

---

## 11. Experiment zum Einfluss der Embedding-Dimension

**Tool:** ChatGPT  
**Datum:** 08.07.2026  

**Prompt:**  
Plane ein kontrolliertes Experiment mit den Embedding-Dimensionen 8, 16, 32 und
64. Verändere nur die Embedding-Dimension und halte alle anderen
Modelleinstellungen konstant.

**Verwendung:**  
ChatGPT wurde zur Planung des kontrollierten Vergleichs der vier
Embedding-Dimensionen verwendet.

**Eigene Leistung:**  
Ich habe die vier Trainingsläufe lokal ausgeführt und darauf geachtet, dass nur
die Embedding-Dimension verändert wurde. Anschließend habe ich die gespeicherten
Ergebnisse und Diagramme kontrolliert und Accuracy, F1-Score, Parameterzahl und
Trainingsdauer miteinander verglichen.

**Reflexion:**  
Das Experiment zeigte, dass eine größere Embedding-Dimension nicht automatisch
zu einer besseren Modellleistung führte. In meinem Versuch lieferte bereits die
Dimension 8 das beste Ergebnis.

---

## 12. Experiment zum Einfluss der Trainingsdatengröße

**Tool:** ChatGPT  
**Datum:** 08.07.2026  

**Prompt:**  
Plane ein reproduzierbares Experiment, bei dem dasselbe Embedding-Modell mit 20,
40, 60, 80 und 100 Prozent eines festen Trainingspools trainiert wird.

**Verwendung:**  
ChatGPT wurde zur Planung der fünf Trainingsläufe mit unterschiedlichen Anteilen
des festen Trainingspools verwendet.

**Eigene Leistung:**  
Ich habe alle fünf Läufe lokal ausgeführt und die erzeugten Konsolen- und
Dateiausgaben kontrolliert. Anschließend habe ich die
Validierungsgenauigkeit, Testgenauigkeit, beste Epoche und Trainingsdauer
verglichen und die Ergebnisse in Tabellen und Diagrammen dargestellt.

**Reflexion:**  
Die Ergebnisse zeigten, dass eine größere Trainingsmenge die Modellleistung
verbessern kann, während der zusätzliche Nutzen bei höheren Datenanteilen
geringer wurde und die Trainingsdauer weiter anstieg.

---

## 13. Explainable-AI-Experiment mit einem Decision Tree

**Tool:** ChatGPT  
**Zeitraum:** Juli 2026  

**Prompt:**  
Strukturiere ein reproduzierbares XAI-Experiment mit dem Iris-Datensatz und einem
Decision Tree.

**Verwendung:**  
ChatGPT wurde zur Strukturierung des Python-Experiments mit einem
interpretierbaren Decision Tree verwendet.

**Eigene Leistung:**  
Ich habe das Skript lokal ausgeführt und die verwendeten Einstellungen
kontrolliert. Danach habe ich Accuracy, Klassifikationsbericht,
Konfusionsmatrix, Baumtiefe, Feature Importances und die Baumvisualisierung
geprüft. Die Entscheidungsregeln und die Bedeutung der wichtigsten Merkmale habe
ich anhand der erzeugten Ergebnisse selbst beschrieben.

**Reflexion:**  
Das Experiment zeigte, dass die Entscheidungen eines begrenzten Decision Trees
direkt anhand seiner Struktur nachvollzogen werden können. Gleichzeitig sind
Feature Importances modellbezogene Werte und keine kausalen Erklärungen.

---

## 14. Überarbeitung des XAI-Abschnitts

**Tool:** ChatGPT  
**Datum:** Juli 2026  

**Prompt:**  
Prüfe den vorhandenen XAI-Abschnitt auf Wiederholungen und formuliere notwendige
Ersatzabschnitte in wissenschaftlichem Deutsch.

**Verwendung:**  
ChatGPT wurde zur Erkennung von Wiederholungen und zur sprachlichen Überarbeitung
der davon betroffenen Textstellen verwendet.

**Eigene Leistung:**  
Ich habe die vorgeschlagenen Änderungen mit meinem vorhandenen XAI-Abschnitt und
den tatsächlichen Versuchsergebnissen verglichen. Danach habe ich selbst
entschieden, welche Stellen ersetzt werden, die Änderungen in LaTeX eingefügt und
die neue PDF kontrolliert.

**Reflexion:**  
Die Überarbeitung half dabei, Wiederholungen zu reduzieren und die Darstellung
klarer zu strukturieren. Die fachliche Übereinstimmung mit meinen Ergebnissen
musste dennoch von mir überprüft werden.

---

## 15. Kontrolle der Abbildungen und der LaTeX-Struktur

**Tool:** ChatGPT  
**Zeitraum:** Juli 2026  

**Prompt:**  
Prüfe die LaTeX-Datei. Nenne bei einem Fehler die konkrete Stelle.

**Verwendung:**  
ChatGPT wurde zur Prüfung der von mir bereitgestellten LaTeX-Datei und zur
Lokalisierung erkennbarer Fehlerstellen verwendet.

**Eigene Leistung:**  
Ich habe die genannten Stellen selbst in der LaTeX-Datei gesucht und überprüft.
Erforderliche Änderungen an Abbildungen, Dateinamen, Beschriftungen oder
Referenzen habe ich eigenständig vorgenommen. Anschließend habe ich das Dokument
erneut kompiliert und das Ergebnis in der PDF kontrolliert.

**Reflexion:**  
Die Angabe konkreter Fehlerstellen erleichterte die Korrektur. Ob die Änderung
korrekt war, konnte jedoch erst durch die erneute Kompilierung und visuelle
Prüfung festgestellt werden.

---

## 16. Überprüfung und Verbesserung der PDF-Datei

**Tool:** ChatGPT  
**Zeitraum:** Juli 2026  

**Prompt:**  
Prüfe die kompilierte PDF des ePortfolios auf Vollständigkeit, fachliche
Nachvollziehbarkeit, Lesbarkeit, Seitenumbrüche, Bildunterschriften,
Ergebnisdarstellung, Reflexionen und formale Unstimmigkeiten. Formuliere für
notwendige Änderungen konkrete Ersatztexte und gib an, an welcher Stelle sie
eingefügt werden sollen.

**Verwendung:**  
ChatGPT wurde zur Prüfung der kompilierten PDF und zur Formulierung konkreter
Korrekturvorschläge für erkannte inhaltliche oder formale Probleme verwendet.

**Eigene Leistung:**  
Ich habe die PDF selbst aus meiner LaTeX-Datei erzeugt und bereitgestellt. Die
Hinweise habe ich mit dem Quelltext und meinen Originalergebnissen verglichen.
Anschließend habe ich die ausgewählten Änderungen selbst in LaTeX umgesetzt, das
Dokument neu kompiliert und die überarbeitete Version kontrolliert.

**Reflexion:**  
Die Prüfung der kompilierten PDF war hilfreich, da einige Probleme erst in der
fertigen Darstellung sichtbar wurden. Die Hinweise mussten jedoch mit dem
Quelltext und den tatsächlichen Ergebnissen abgeglichen werden.

---

## 17. Unterstützung bei der Programmierung

**Tool:** ChatGPT  
**Zeitraum:** Mai bis Juli 2026  

**Prompt:**  
Unterstütze mich bei der Programmierung und bei der Korrektur von Fehlern in den
Python-Skripten meines ePortfolios.

**Verwendung:**  
ChatGPT wurde verwendet, um einzelne Codeabschnitte zu erklären, Fehlermeldungen
einzuordnen und mögliche Korrekturen für die betroffenen Python-Skripte
vorzuschlagen.

**Eigene Leistung:**  
Ich habe die Skripte selbst in Visual Studio Code angelegt, ausgeführt und
getestet. Fehlermeldungen und Konsolenausgaben habe ich bereitgestellt, die
vorgeschlagenen Änderungen an den betroffenen Stellen umgesetzt und die Skripte
danach erneut ausgeführt. Die erzeugten Ergebnisse und Dateien habe ich
kontrolliert, bevor ich sie für das ePortfolio verwendet habe.

**Reflexion:**  
Die Unterstützung war besonders bei der Suche nach Ursachen für Fehler hilfreich.
Die tatsächliche Funktion der Korrekturen konnte jedoch nur durch meine lokale
Ausführung und die Kontrolle der erzeugten Ergebnisse bestätigt werden.
