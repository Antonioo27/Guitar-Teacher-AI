# Pipeline offline — script CLI

Questi script servono a **valutare la trascrizione di TabCNN** su una traccia
e confrontarla con uno spartito di riferimento, fuori dal web app. Sono pensati
per la sperimentazione e la generazione di metriche/immagini.

Lo script principale è **`predictAndCompare.py`**, che incatena prediction +
confronto in un solo comando.

---

## Prerequisiti

1. **Ambiente Python** — usare il virtualenv del progetto (NON il `python3` di
   sistema, che non ha le dipendenze):

   ```bash
   source .venv/bin/activate          # poi si usa "python"
   # oppure, senza attivare:
   .venv/bin/python -m guitar_tutor_pipeline.src.<script> ...
   ```

2. **Pesi del modello** — `GuitarSet.pt` deve essere presente nel percorso
   indicato da `config.WEIGHTS_PATH`.

3. Tutti i comandi vanno lanciati dalla **root della repo**
   (`Guitar-Teacher-AI/`), perché gli script sono moduli del package.

---

## Flusso

```
.npy (fornito nella repo) ──(predictAndCompare)──▶ note + immagini + metriche
                                   │
                                   └── usa lo spartito .mid come riferimento
```

Le matrici CQT (`.npy`) sono **già incluse nella repo** (in `output_cqt/`),
quindi `predictAndCompare` si lancia direttamente:

```bash
python -m guitar_tutor_pipeline.src.predictAndCompare \
  output_cqt/cmaj/00_CMaj_120_cqt.npy \
  SetTestHomeMade/00_CMaj_120.mid \
  --output_dir output_notes/cmaj
```

---

## Esempi completi (copia-incolla)

Eseguire dalla **root della repo**. Il primo blocco attiva il virtualenv una
volta sola per la sessione.

```bash
source .venv/bin/activate
```

### Esempio A — coppia `00_CMaj_120`

```bash
python -m guitar_tutor_pipeline.src.predictAndCompare \
  output_cqt/cmaj/00_CMaj_120_cqt.npy \
  SetTestHomeMade/00_CMaj_120.mid \
  --output_dir output_notes/cmaj
```

### Esempio B — coppia `00_CMaj_120_Velocity127`

```bash
python -m guitar_tutor_pipeline.src.predictAndCompare \
  output_cqt/velocity127/00_CMaj_120_Velocity127_cqt.npy \
  SetTestHomeMade/00_CMaj_120_Velocity127.mid \
  --output_dir output_notes/velocity127
```

> Verifica i percorsi dei `.npy` disponibili con `find output_cqt -name "*.npy"`.

Al termine, in `--output_dir` trovi note (`.json`/`.csv`), le due immagini
(`_vs_midi.png`, `_overlay.png`) e le metriche (`_metrics.json`).

---

## Cosa fa `predictAndCompare.py` (stato attuale)

A partire da `.npy` + `.mid`:

1. carica la matrice CQT e costruisce il tensore di input;
2. carica TabCNN ed esegue l'inferenza (forward + softmax → tasto per corda);
3. decodifica i frame in note (con isteresi + filtro durata + gap fill);
4. **ghost-fix onset-aware**: fonde i frammenti dello stesso pitch (es. lo
   stesso suono attribuito a corde diverse), usando gli onset rilevati dalla
   CQT per non fondere note realmente ribattute;
5. salva le note in `.json` e `.csv`;
6. allinea predizione e spartito con **DTW basato sul pitch**;
7. salva due immagini e un file di metriche.

### Output prodotti in `--output_dir`

| File | Contenuto |
|------|-----------|
| `<stem>_notes.json` / `.csv` | note predette (time, durata, pitch, corda, tasto, confidenza) |
| `<stem>_vs_midi.png` | piano roll a due corsie (suonato vs spartito) con i collegamenti DTW |
| `<stem>_overlay.png` | predizione e spartito sovrapposti sullo stesso asse → sfasamento temporale |
| `<stem>_metrics.json` | metriche di accuratezza (vedi sotto) |

(`<stem>` = nome del file `.npy`.)

### Metriche calcolate

- **Note-level** (mir_eval): Precision/Recall/F1 su *onset* e su *onset+offset*,
  average overlap ratio, conteggi TP/FP/FN. Una nota è corretta se ha lo stesso
  pitch e l'onset entro la tolleranza (default 50 ms).
- **Frame-level pitch** (stile TabCNN): P/R/F1 + accuracy frame per frame.
- **Tempo/durata**: sfasamento di onset (medio/mediano/std/max), % note entro
  tolleranza, errore medio di durata.

---

## Limiti attuali (da sapere)

- **Solo `.mid`** come riferimento. `parse_jams` esiste in `dataset.py` ma NON è
  ancora collegato a questi script: i file `.jams` (GuitarSet) non sono
  supportati da `predictAndCompare`/`compareWithMidi` per ora.
- **Niente metriche di tablatura** (corda+tasto) né **TDR**: il MIDI non contiene
  l'informazione di corda, quindi si valuta solo a livello di pitch. Servirebbe
  un ground truth `.jams` con le corde.
- Il DTW allinea **solo sul pitch**: robusto a scale temporali diverse, ma non
  usa il tempo nel matching.

---

## Parametri di `predictAndCompare.py`

| Parametro | Default | A cosa serve |
|-----------|---------|--------------|
| `npy_path` (posizionale) | — | matrice CQT `.npy` (output di `convertWawToCQT`) |
| `midi_path` (posizionale) | — | spartito `.mid`/`.midi` di riferimento |
| `--output_dir` | `output_notes` | cartella di output (note, immagini, metriche) |
| `--weights` | `config.WEIGHTS_PATH` | percorso ai pesi del modello |
| `--device` | `cpu` | `cpu` o `cuda` |
| `--sr` | `22050` | **sample rate con cui è stato generato il `.npy`** (vedi nota) |
| `--hop_length` | `512` | hop length con cui è stato generato il `.npy` |
| `--no-ghost-fix` | off | disattiva la risoluzione delle ghost note |
| `--ghost-gap` | `0.12` | gap di sicurezza (s) per fondere frammenti dello stesso pitch |
| `--no-onsets` | off | non usa gli onset della CQT nella ghost-fix (solo gap) |
| `--onset-delta` | `0.07` | soglia di prominenza per il rilevamento onset (più alto = meno onset) |

> ⚠️ **Coerenza del sample rate.** I `.npy` generati da `convertWawToCQT` usano
> `SAMPLE_RATE = 22050, HOP_LENGTH = 512` (diversi dai 44100 di `config.py`).
> `predictAndCompare` usa quei valori di default importandoli da
> `convertWawToCQT`, così i tempi delle note predette combaciano con il MIDI. Se
> generi i `.npy` con parametri diversi, passa `--sr`/`--hop_length` coerenti,
> altrimenti i tempi risulteranno scalati e l'allineamento sfasato.

---

## Script disponibili (riepilogo)

| Script | Input → Output |
|--------|----------------|
| `convertWawToCQT.py` | `.wav` → `.npy` + spettrogramma `.png` (già eseguito: i `.npy` sono nella repo) |
| `runTabCNN.py` | `.npy` → note predette (`.json`/`.csv`) |
| `inspectMidi.py` | `.mid` → ispezione + `.csv` |
| `compareWithMidi.py` | note `.json` + `.mid` → confronto DTW + immagini + metriche |
| `predictAndCompare.py` | `.npy` + `.mid` → **tutto il precedente in un comando** |
