# Micro-Electrode Recordings – STN-DBS Surgery

## 🧠 Progetto di Neuroingegneria – A.A. 2024/2025  
**Gruppo 14:** Angela Debellis, Irene Lionetti, Rossella Petruzzella, Manuela Vitulano  
Politecnico di Torino

---

## 🎯 Obiettivo del progetto

Il progetto nasce all'interno del corso di **Neuroingegneria** e ha come finalità l’ottimizzazione del **targeting della Subthalamic Nucleus (STN)** durante l’intervento di **Deep Brain Stimulation (DBS)** su pazienti affetti da **Malattia di Parkinson (PD)**.  
L’identificazione accurata della STN è fondamentale per il successo dell’intervento e per la corretta impostazione dei parametri di stimolazione.  
A tale scopo sono state utilizzate registrazioni **micro-elettrodiche (MER)** acquisite intraoperatoriamente, e sono stati implementati due modelli di classificazione per distinguere tra segnali **interni** e **esterni** alla STN:

- **k-Nearest Neighbors (kNN)**
- **Classificatore Bayesiano**

---

## 🧪 Metodologia

- Le MER sono state ottenute da **3 pazienti** sottoposti a **DBS bilaterale**.
- I segnali sono stati filtrati (band-pass 200–6000 Hz) e suddivisi in **epoche da 1 secondo** con **overlap del 50%**.
- Da ciascuna epoca sono state estratte **11 caratteristiche** divise in:
  - **Spike-dependent** (ad esempio: spike rate, ampiezza, area)
  - **Spike-independent** (potenza di banda, varianza, entropia, skewness, kurtosis, ecc.)
- I modelli sono stati validati con **Leave-One-Patient-Out Cross Validation (LOPO-CV)**, per simulare un’applicazione clinica realistica.

---

## 📊 Risultati

- Il **kNN** ha raggiunto prestazioni superiori in termini di:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
- Il **classificatore Bayesiano**, sebbene con accuratezza inferiore, ha mostrato maggiore **stabilità** e **capacità di generalizzazione** tra pazienti, suggerendo un potenziale uso in contesti con poca disponibilità di dati di training.

---

## 📁 Struttura della cartella

Neuroengineering/
├── codici/ # Script MATLAB per preprocessing, estrazione feature e classificazione
├── NEURO_2024.pdf # Relazione tecnica finale del progetto
└── README.md # Descrizione del progetto 

---

## ⚙️ Tecnologie utilizzate

- **Linguaggio:** MATLAB R2022a+
- **Toolbox:** Signal Processing Toolbox
- **Modelli:** kNN, Naive Bayes
- **Validazione:** Leave-One-Patient-Out
