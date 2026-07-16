# Guitar-Teacher-AI
Project of Artificial Intelligence, computer science Unviersity of Bologna

## Panoramica

Pipeline ibrida (Deep Learning + DTW + LLM) per la valutazione dell'esecuzione chitarristica, con frontend Vue 3 e backend FastAPI.

## Struttura del Progetto

```
Guitar-Teacher-AI/
├── requirements.txt                     # Dipendenze Python (pipeline + API)
├── frontend/                            # Frontend Vue 3 + Vite
│   ├── src/
│   │   ├── App.vue                      # App principale
│   │   ├── style.css                    # Design system (dark theme)
│   │   ├── main.js                      # Entry point Vue
│   │   └── components/
│   │       ├── FileUpload.vue           # Drag & drop upload
│   │       ├── AnalysisResults.vue      # Gauge + tabella errori
│   │       ├── FeedbackPanel.vue        # Feedback LLM
│   │       └── StatusBar.vue            # Status server
│   └── ...
├── guitar_tutor_pipeline/
│   ├── __init__.py
│   ├── api.py                           # FastAPI REST API (model selection)
│   └── src/
│       ├── app/                         # Moduli condivisi della pipeline
│       │   ├── config.py                # Configurazione centralizzata
│       │   ├── alignment.py             # Modulo 3: DTW
│       │   ├── feedback.py              # Modulo 4: LLM
│       │   ├── dataset.py               # Parsing MIDI/JAMS
│       │   ├── pipeline.py              # Orchestrazione (model-agnostic)
│       │   └── model_registry.py        # Caricamento dinamico modelli
│       ├── TabCNN_Architecture/         # Architettura TabCNN
│       │   ├── model.py                 # Definizione modello
│       │   ├── inference.py             # Predizione note
│       │   ├── audio_processing.py      # Modulo 1: CQT
│       │   └── ...                      # Script standalone (runTabCNN, ecc.)
│       └── CRNN_Model/                  # Architettura CRNN
│           ├── model.py                 # Definizione modello (High-Resolution)
│           ├── inference.py             # Predizione note
│           └── ...                      # Script standalone
```

## Come avviare

### 1. Backend (FastAPI)

```bash
source venv/bin/activate
uvicorn guitar_tutor_pipeline.api:app --reload --port 8000
```

API disponibile su `http://localhost:8000` con docs Swagger su `/docs`.

### 2. Frontend (Vue 3)

```bash
cd frontend
npm run dev
```

Frontend disponibile su `http://localhost:5173`.

### Endpoints API

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| GET | `/api/health` | Status server e modello attivo |
| GET | `/api/config` | Configurazione pipeline |
| GET | `/api/models` | Lista dei modelli disponibili e quello in uso |
| GET | `/api/model` | Restituisce il modello attualmente selezionato |
| POST | `/api/model` | Cambia il modello attivo (es: `{"model": "CRNN"}`) |
| POST | `/api/transcribe` | Solo trascrizione audio → note (usa il modello attivo) |
| POST | `/api/analyze` | Pipeline completa: audio + spartito → errori + feedback (usa il modello attivo) |

### Pipeline da Terminale
```bash
# 1. Attiva l'ambiente virtuale (se non lo è già)
source venv/bin/activate

# 2. Guarda quali modelli sono disponibili
python -m guitar_tutor_pipeline.src.app.pipeline --list-models

# 3. Lancia un'analisi completa con TabCNN (modello di default)
python -m guitar_tutor_pipeline.src.app.pipeline \
    --audio data/tuo_file_audio.wav \
    --reference data/tuo_spartito.mid \
    --context "Scala di Do Maggiore"

# 4. Lancia la stessa analisi ma forzando l'uso del modello CRNN
python -m guitar_tutor_pipeline.src.app.pipeline \
    --audio data/tuo_file_audio.wav \
    --reference data/tuo_spartito.mid \
    --model CRNN \
    --context "Scala di Do Maggiore"

```

### Script standalone
```bash
source venv/bin/activate

# Step A: Converti un audio in matrice CQT
python -m guitar_tutor_pipeline.src.TabCNN_Architecture.convertWawToCQT path/audio.wav --output_dir output_test/

# Step B: Lancia il modello TabCNN sulla matrice appena creata
python -m guitar_tutor_pipeline.src.TabCNN_Architecture.runTabCNN output_test/audio_cqt.npy --output_dir output_test/

# Step C: Confronta visivamente (genera i grafici) con il MIDI
python -m guitar_tutor_pipeline.src.TabCNN_Architecture.compareWithMidi output_test/audio_notes.json path/spartito.mid --output_dir output_test/

# Oppure tutto insieme in un colpo solo (CQT -> Predizione -> Grafici)
python -m guitar_tutor_pipeline.src.TabCNN_Architecture.predictAndCompare output_test/audio_cqt.npy path/spartito.mid --output_dir output_test/

```

### Test Risultati Finali
Per testare entrambi i modelli:
```bash
python -m guitar_tutor_pipeline.evaluate_models
```

## Note

> [!IMPORTANT]
> Per usare la pipeline TabCNN, il file dei pesi deve essere presente in `guitar_tutor_pipeline/weights/`.

> [!NOTE]
> Per il feedback LLM, crea un file `.env` nella root con `OPENAI_API_KEY=sk-...`
