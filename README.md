# Generatore GPT-Dante UI

## Overview
Repo contenente l'interfaccia grafica per la generazione di testo con il modello di GePeTto fine-tunato su testi in terza rima.

## Requisiti
Python 3.5.2+

## Utilizzo

Per lanciare la UI prima installare i requirements:

```
pip install -r requirements.txt
```
Una volta installati, modificare la riga 68 dello script **generator_script.py** specificando il percorso al modello fine-tunato di GePpeTto (i modelli sono disponibili qui: https://drive.google.com/drive/folders/1wDutaN9dm6CbJ8V2PM77UvCrcH8JkQU-) e lanciare il server da terminale:

```
flask run
```

