# Riconoscimento di Gesti Culturali Italiani con Unity e Random Forest

🎓 **Progetto sviluppato per il corso "Soluzioni di Grafica 3D in Applicazioni Biometriche" – Politecnico di Torino**

## 🎯 Obiettivo

Realizzazione di un sistema interattivo che riconosce gesti iconici della cultura italiana utilizzando:
- una pipeline RGB-D con landmark MediaPipe,
- classificatore Random Forest con classe di rifiuto,
- e una scena Unity ambientata in un aeroporto virtuale.

L’obiettivo finale è **insegnare i gesti ai turisti in modo immersivo e divertente**.

---

## 🛠️ Tecnologie usate

- Python 3.9.13 (scikit-learn, MediaPipe, matplotlib, SHAP)
- Unity 2022.3.6f1 (C# scripting)
- Intel RealSense (RGB-D acquisition)
- Visual Studio Code
- Google Colab

---

## 📁 Contenuto della cartella

- `Colab_Notebooks/`: script per preparazione dataset, normalizzazione e allenamento modelli
- `dataset/`: landmark, distanze 3D e CSV delle classi
- `Unity_Project/`: progetto Unity + codice Python per comunicazione client-server
- `Group14_Paper.pdf`: articolo scientifico del progetto (ENG)
- `README.md`: questa descrizione

---

## 🧪 Classificatori testati

- ✅ **Random Forest** (migliore → F1 ≈ 85%)
- SVM (meno preciso)
- LightGBM
- Outlier detection: Isolation Forest, Local Outlier Factor
- SHAP analysis delle feature usate

---

## 🧠 Gesti riconosciuti

- 🤌 Mano a borsa  
- 🤘 Corna  
- 🤞 Dita incrociate  
- 👌 Segno di perfezione  
- ❌ Gesto non riconosciuto (classe di rifiuto)

---

## 📸 Modalità di utilizzo

1. Unity cattura un frame RGB-D
2. Lo invia al server Python via TCP
3. Il modello elabora e restituisce la classe del gesto
4. Unity visualizza il risultato nel pannello

---

## 👩‍💻 Autori principali

- **Irene Lionetti**  
- Giorgio Di Pisa, Andrea Mamoli, Manuela Vitulano  
Politecnico di Torino – 2025

---

## 📎 Riferimenti

> *"Promoting Cultural Engagement: A Random Forest Approach to Teaching Italian Hand Gestures to Tourists"*  
→ [PDF Paper allegato](./Group14_Paper.pdf)
