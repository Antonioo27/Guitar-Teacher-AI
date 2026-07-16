# 🎸 Guitar Tutor AI - Script di Valutazione Modelli

Questo set di script modulari e generici permette di valutare facilmente l'accuratezza dei modelli di trascrizione (TabCNN e CRNN) sui dataset di test, calcolando le metriche di MIR_EVAL e salvando i risultati aggregati in file CSV e JSON.

## 📂 Struttura

La cartella `scripts` contiene:
- `eval_utils.py`: Funzioni generiche per la scoperta automatica dei file di test nel dataset, calcolo delle metriche e salvataggio dei risultati. Queste funzioni sono state scritte per essere facilmente importabili e riutilizzabili da altri script (ad es. da Jupyter Notebooks).
- `evaluate.py`: La Command Line Interface (CLI) principale. Prende in input i parametri e lancia il loop di valutazione, stampando un recap a terminale.
- `plot_results.py`: Script che prende in input i file JSON generati dalla valutazione e genera istogrammi per il confronto diretto dei modelli.
- `out/`: La cartella dove verranno salvati tutti i risultati generati in JSON, CSV e i grafici PNG.

## 🚀 Utilizzo (Bash)

Gli script vanno eseguiti dal terminale avendo attivato l'ambiente virtuale del progetto. 

La CLI `evaluate.py` accetta diversi argomenti:
- `--models`: Scegli tra `TabCNN`, `CRNN`, oppure `both` (default).
- `--dataset`: Scegli tra `synthetic`, `real`, oppure `both` (default).
- `--out-name`: (Opzionale) Prefisso per il nome del file generato. (es. se inserisci `valutazione`, creerà `valutazione_results.json` e `valutazione_results.csv`).

### 1️⃣ Test completi (Tutti i modelli, Tutti i dataset)
Questo è il comando base. Valuterà sia TabCNN che CRNN su tutti i file del dataset (Sintetici e Reali).

```bash
python scripts/evaluate.py
```
*(Puoi anche specificare i parametri esplicitamente)*:
```bash
python scripts/evaluate.py --models both --dataset both
```

### 2️⃣ Testare SOLO i file sintetici
Valuta entrambi i modelli solo ed esclusivamente sui file `.wav` generati sinteticamente (situati nella root di `SetTestHomeMade`).
```bash
python scripts/evaluate.py --dataset synthetic --out-name risultati_sintetici
```

### 3️⃣ Testare SOLO i file TrueRecording
Valuta entrambi i modelli solo sui file `.wav` reali registrati con strumenti (situati in `SetTestHomeMade/TrueRecordings`).
```bash
python scripts/evaluate.py --dataset real --out-name risultati_reali
```

### 4️⃣ Testare SOLO TabCNN (Su tutti i dataset)
Vuoi controllare le metriche solo per TabCNN, saltando CRNN.
```bash
python scripts/evaluate.py --models TabCNN --out-name risultati_tabcnn
```

### 5️⃣ Testare SOLO CRNN sui file TrueRecording
Puoi combinare tutti gli argomenti assieme per creare scenari specifici.
```bash
python scripts/evaluate.py --models CRNN --dataset real --out-name crnn_reali
```

## 📊 I Risultati

Per ogni valutazione, i risultati vengono salvati contemporaneamente in due formati per la massima comodità:
1. **JSON**: Perfetto se devi caricare i dati via Python per manipolarli con script avanzati o web-app.
2. **CSV**: Comodo se vuoi aprire i risultati con Excel o Google Sheets per fare grafici rapidi.

Li troverai generati automaticamente nella cartella `scripts/out/`.

## 📈 Generazione dei Grafici

Dopo aver generato i risultati eseguendo `evaluate.py`, puoi visualizzarli in un istogramma affiancato per un confronto diretto.
Utilizza lo script `plot_results.py`:

```bash
python scripts/plot_results.py --input scripts/out/eval_results.json --output scripts/out/comparison_plot.png
```

Se avevi specificato un `--out-name` personalizzato (es: `risultati_reali`), ricordati di passare lo stesso nome:
```bash
python scripts/plot_results.py --input scripts/out/risultati_reali_results.json --output scripts/out/confronto_reali.png
```

Lo script genererà un'immagine in alta risoluzione pronta per essere utilizzata in presentazioni o report.
