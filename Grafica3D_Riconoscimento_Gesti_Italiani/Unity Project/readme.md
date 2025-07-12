# 📦 ISTRUZIONI PER UTILIZZARE IL PROGETTO UNITY

🔗 Il progetto completo in Unity, comprensivo di scena, script C#, documentazione e codice Python integrato, è disponibile su Google Drive:

👉 [Scarica progetto Unity – Google Drive](https://drive.google.com/drive/folders/1YrH2XRiFtrSFZRWKAAQ7J0mEkteQHdLh?usp=sharing)

---

## 🧪 Verifica iniziale
- Assicurarsi che il **server Python** sia attivo eseguendo `capture_server.py`, necessario per ricevere i comandi di cattura da Unity.

---

## 🎮 1) Scena principale
- La scena principale si chiama `SampleScene`
- Aprirla da Unity per iniziare la simulazione

---

## 🎥 2) Controlli di visualizzazione
- Premere **C** per alternare la visuale tra **Main Camera** e **Mobile Camera**

---

## 🕹️ 3) Movimento del personaggio
- Utilizzare i tasti **WASD** per muovere il Player all'interno della scena

---

## 📡 4) Attivazione del Totem
- Avvicinarsi al Totem fino a entrare nel **trigger**:
  - Il movimento del Player viene **bloccato**
  - Si attiva la **Totem Camera**
  - Inizia la presentazione interattiva

---

## 📽️ 5) Presentazione dei gesti
- Seguire le **istruzioni a schermo** per avviare la sequenza dei video
- Usare i pulsanti **Avanti** e **Indietro** per navigare tra i gesti
- Al termine, cliccare per proseguire alla fase successiva

---

## 🤖 6) Acquisizione con RealSense (Webcam Integration Scene)
- Se è disponibile la camera **Intel RealSense**:
  - Premere il pulsante **TAKE PICTURE** per acquisire un frame RGB e Depth
  - I file verranno salvati in una cartella definita nello script `RealSenseCapture.cs`
  - Lo script invia un comando al modulo Python (`spyware_rf.py`) per il riconoscimento del gesto
  - ⚠️ Verificare che il **path definito in `spyware_rf.py`** coincida con quello di salvataggio

---

Per qualsiasi dettaglio aggiuntivo, consulta anche il file `Unity How To.docx` incluso nella cartella Drive.
