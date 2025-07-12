# Analisi del Ritardo e della Fatica Muscolare tramite Stimolazione Elettrica e Simulazione della Conduction Velocity (CV)

🎓 Progetto realizzato per il corso **Ingegneria del Sistema Neuromuscolare** – Politecnico di Torino

## 🎯 Obiettivo

Il progetto si articola in due principali attività, finalizzate allo studio del comportamento neuromuscolare in risposta alla stimolazione elettrica e alla modellazione della fatica muscolare:

1. **Valutazione del ritardo muscolare (LAB1)** attraverso l'analisi dei segnali EMG multicanale derivanti da stimolazione del muscolo **tibiale anteriore**, al variare dell’intensità dello stimolo.

2. **Simulazione dell’insorgenza della fatica muscolare (LAB2)** tramite analisi della variazione della **Conduction Velocity (CV)**. Sono stati implementati diversi algoritmi di stima, anche in presenza di rumore, e ne sono state confrontate l’efficacia e l’accuratezza.

---

## 🧪 Attività principali

### 🔬 LAB1 – Ritardo EMG da stimolazione

- Rimozione dell'artefatto da stimolo
- Calcolo soglia EMG e ritardo per ciascun canale
- Visualizzazione e media statistica
- Script principali: `protocollo_1_finale.m`, `MyPlot.m`, `delay.m`

### ⚙️ LAB2 – Simulazione CV e fatica

- Generazione di EMG simulato con CV decrescente
- Aggiunta di rumore realistico (simulazioni noisy)
- Implementazione metodi:
  - **beamforming**
  - **MLE (Maximum Likelihood Estimation)**
  - **stima da ritardo tra canali**
- Analisi delle metriche di errore e visualizzazione

---

## 📁 Struttura della cartella

- `GRP02.pdf`: relazione scientifica finale
- `GRP02_LAB1/`: script MATLAB per studio sperimentale del ritardo
- `GRP02_LAB2/`: simulazioni MATLAB per la fatica muscolare
- `CV_multich/`: funzioni di stima CV avanzate (beam_CV_est, mle3, sig_dis, ecc.)

---

## 💻 Tecnologie

- MATLAB R2023b
- Signal processing EMG multicanale
- Analisi statistica e visualizzazione

---

## 👩‍🔬 Autori

- **Giulia Ciribilli**  
- **Angela Debellis**  
- **Irene Lionetti**  
- **Valerio Tarditi**  
- **Luana Turchiarulo**  
- **Manuela Vitulano**  

📍 *Politecnico di Torino – A.A. 2024/2025*
