calibrated_rf_model.pkl
    Modello Random Forest allenato per la classificazione dei gesti. Contiene i pesi e i parametri appresi durante il training.

capture_server.py
    Script Python che avvia un server TCP per ricevere comandi da Unity. Quando riceve il messaggio "CAPTURE", esegue la classificazione chiamando spyware_rf.main().

hand_landmarker.task
    File di modello MediaPipe per il rilevamento dei landmark della mano. Usato dallo script spyware_rf.py per individuare i punti chiave delle mani.

RestituisciClasse.cs
    Script C# che gestisce il ricevimento dei risultati di classificazione dal sistema Python e restituisce la classe predetta all’applicazione Unity o ad altri componenti.

scaler_rf.pkl
File pickle con lo scaler (normalizzatore) utilizzato per normalizzare i dati prima della classificazione Random Forest.
   
spyware_rf.py
    Script Python che carica immagini RGB e depth, esegue il rilevamento dei landmark delle mani con MediaPipe, calcola le distanze 3D tra i punti, normalizza le feature e classifica il   gesto usando Random Forest. I risultati vengono inviati a Unity tramite socket TCP.

