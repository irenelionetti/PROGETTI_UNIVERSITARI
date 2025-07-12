# 🧠 Codici MATLAB – Progetto di Neuroingegneria

Questa cartella contiene tutti gli script MATLAB sviluppati per l'analisi e classificazione di segnali MER simulati, nell’ambito di un progetto didattico di **Ingegneria del Sistema Neuromuscolare**.

## 🔹 Script principale

- **`NEURO_DEF.m`**  
  È lo **script principale** del progetto. Gestisce l’intero workflow:
  - Caricamento dati
  - Normalizzazione
  - Suddivisione in training/validation
  - Addestramento dei classificatori
  - Valutazione delle performance finali

## 🔹 Funzioni di supporto

Tutti gli altri file `.m` sono **moduli funzionali** richiamati da `NEURO_DEF.m`:

| File                        | Descrizione                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `evaluate_classifier.m`     | Calcola accuratezza, precisione, recall e F1-score su validation set        |
| `normalize_data.m`          | Applica la normalizzazione Z-score al dataset                               |
| `split_by_class.m`          | Divide i dati per classe (es. STN vs non-STN)                               |
| `split_training_validation.m` | Suddivide il dataset in training e validation (stratificato)              |
| `train_knn.m`               | Allena un classificatore k-Nearest Neighbors                                |
| `train_bayes.m`             | Allena un classificatore Bayesiano                                          |

## 💡 Modalità d'uso

Aprire lo script `NEURO_DEF.m` ed eseguirlo:  
```matlab
>> run('NEURO_DEF.m')

