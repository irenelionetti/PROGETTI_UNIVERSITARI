ISTRUZIONI PER UTILIZZARE IL PROGETTO UNITY


- Verificare che il server Python sia in esecuzione per ricevere i comandi di cattura runnando il codice capture_server.py

1) SCENA PRINCIPALE
La scena principale si chiama: SampleScene
Aprire questa scena per avviare il progetto.

2) CONTROLLI DI VISUALIZZAZIONE
- Premere il tasto C per passare dalla Main Camera alla Mobile Camera.

3) MOVIMENTO DELL'OGGETTO
- Utilizzare i tasti WASD per muovere l'oggetto Player nell'ambiente.

 4) ATTIVAZIONE DEL TOTEM
- Avvicinare il Player al Totem fino a entrare nel trigger.
- Quando il trigger viene attivato:
  • Il movimento del Player viene bloccato.
  • La Totem Camera diventa attiva per mostrare la presentazione.

5)  PRESENTAZIONE DEI GESTI
- Seguire le istruzioni a schermo per far partire la sequenza dei video dei gesti.
- Utilizzare i pulsanti Avanti e Indietro per navigare tra i gesti.
- Alla fine della presentazione, premere il pulsante per avviare l'eventuale fase successiva.

6)  REALSENSE E CATTURA (PASSAGGIO ALLA WEBCAM INTEGRATION SCENE)
- Se è presente la camera RealSense:
  • Premere il pulsante TAKE PICTURE per catturare un frame RGB e Depth.
  • I file vengono salvati in una cartella con path da definire nello script RealSenseCapture.cs
  • Dopo il salvataggio, un comando viene inviato allo script Python per l'elaborazione (accertarsi che nel codice spyware_rf.py sia definito lo stesso path dello script precedente).
