import socket
import threading
import time
import os
import sys


import spyware_rf  

# Configurazione IP e porta
HOST = "127.0.0.1"
PORT = 9999

def handle_client(conn, addr):
    print(f" Connessione da {addr}")
    try:
        data = conn.recv(1024).decode("utf-8").strip()
        print(f" Messaggio ricevuto: {data}")

        if data == "CAPTURE":
            print(" Avvio classificazione...")
            spyware_rf.main()
        else:
            print(" Messaggio sconosciuto.")
    except Exception as e:
        print(" Errore durante la gestione del client:", e)
    finally:
        conn.close()

def start_server():
    print(f"🟢 Server TCP in ascolto su {HOST}:{PORT}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()

        while True:
            conn, addr = s.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, addr))
            client_thread.start()

if __name__ == "__main__":
    start_server()
