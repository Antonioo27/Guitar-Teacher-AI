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
│   ├── api.py                           # FastAPI REST API
│   └── src/
│       ├── config.py                    # Costanti centralizzate
│       ├── audio_processing.py          # Modulo 1: CQT
│       ├── dataset.py                   # Modulo 1: JAMS/MIDI
│       ├── model.py                     # Modulo 2: TabCNN
│       ├── inference.py                 # Modulo 2: Inferenza
│       ├── alignment.py                 # Modulo 3: DTW
│       ├── feedback.py                  # Modulo 4: LLM
│       └── pipeline.py                  # Orchestrazione + CLI
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
| GET | `/api/health` | Status server e modello |
| GET | `/api/config` | Configurazione pipeline |
| POST | `/api/transcribe` | Solo trascrizione audio → note |
| POST | `/api/analyze` | Pipeline completa: audio + spartito → errori + feedback |

## Verifiche

- ✅ Tutti i moduli Python importabili
- ✅ Forward pass TabCNN: `(1,1,192,9)` → `6 × (1,21)`
- ✅ FastAPI: 4 route registrate correttamente
- ✅ Frontend Vue: build di produzione OK (15.9 KB CSS + 75.6 KB JS)

## Note

> [!IMPORTANT]
> Per usare la pipeline, il file `tabcnn_best.pt` deve essere in `data/synthtab_weights/`.

> [!NOTE]
> Per il feedback LLM, crea un file `.env` nella root con `OPENAI_API_KEY=sk-...`
