---

# **AI Guitar Tutor \- Trascrizione Neurale e Allineamento Sequenziale per la Valutazione Esecutiva**

## **Obiettivo del Progetto**

Sviluppare una pipeline ibrida (Deep Learning \+ Algoritmica \+ LLM) in grado di ascoltare l'esecuzione di uno studente di chitarra, trascriverne il contenuto in formato simbolico, allinearlo allo spartito ideale e generare un feedback didattico naturale.

## **Architettura e Pipeline Tecnica**

La pipeline si divide in quattro moduli sequenziali. Per ciascun modulo sono indicate le librerie Python standard de facto per l'industria e la ricerca.

### **Modulo 1: Data Ingestion & Preprocessing (L'Udito)**

In questa fase si trasformano i dati grezzi (audio e spartiti) in tensori matematici pronti per la rete neurale.

* **Dataset Scelto:** *GuitarSet* (Audio ad alta qualità \+ annotazioni JAMS ad altissima precisione).  
* **Task Operativi:**  
  * Estrazione delle feature audio: conversione dei file .wav in spettrogrammi. È consigliato usare la **CQT (Constant-Q Transform)** invece del classico Mel-Spectrogram, in quanto la CQT allinea geometricamente le frequenze alle note musicali.  
  * Parsing delle annotazioni: estrazione dei millisecondi esatti di *onset* (attacco) e *pitch* (frequenza) per creare i tensori di Ground Truth (le etichette Y).  
* **Librerie Python:**  
  * librosa (nello specifico librosa.cqt e moduli di display).  
  * jams o pretty\_midi (per leggere le annotazioni del dataset).  
  * numpy e pandas (per la manipolazione dei dati).

### **Modulo 2: Acoustic Model (Il Cervello Percettivo \- Core AI 1\)**

Questo è il cuore dell'esame di Machine Learning. Addestramento di una Rete Neurale per l'Automatic Music Transcription (AMT).

* **Architettura:** Rete Neurale Convoluzionale (CNN) o Convoluzionale-Ricorrente (CRNN). I layer convoluzionali estrarranno le feature timbriche dallo spettrogramma CQT, mentre (se usati) i layer ricorrenti come le LSTM modelleranno la sequenza temporale per individuare gli *onset*.  
* **Fase di Training:** Addestramento su Google Colab. Calcolo della Loss function (es. *Binary Cross-Entropy* per la presenza/assenza di note in determinati frame temporali).  
* **Metriche di Valutazione:** Implementazione di *Precision*, *Recall* e *F1-Score* calcolate con finestre di tolleranza temporale (es. 50 ms). Obbligatorio eseguire uno studio di *ablation* (es. testare l'impatto di diverse dimensioni dei kernel della CNN o della presenza/assenza del layer ricorrente).  
* **Librerie Python:**  
  * torch (PyTorch) o tensorflow/keras.  
  * scikit-learn (per lo split del dataset e alcune metriche metriche).  
  * mir\_eval (una libreria fondamentale e standardizzata per calcolare l'F1-score nell'audio musicale).

### **Modulo 3: Sequence Alignment (La Logica di Controllo)**

Fase di *Inference*. Il modello neurale ha prodotto un array di note previste. Ora bisogna confrontarle in modo elastico con lo spartito teorico dell'esercizio.

* **Algoritmo:** **Dynamic Time Warping (DTW)**.  
* **Task Operativi:**  
  * Allineare la sequenza predetta dall'IA (spesso imperfetta e con note spurie) alla sequenza dello spartito (letta da file MIDI o MusicXML).  
  * Calcolare il delta t per ogni accoppiamento per classificare l'errore: *Nota Corretta a tempo*, *Nota Corretta ma fuori tempo*, *Nota Sbagliata*, *Nota Mancante*.  
* **Librerie Python:**  
  * fastdtw o librosa.sequence.dtw (per l'allineamento computazionalmente efficiente).  
  * scipy.spatial.distance (per le metriche di distanza tra array).

### **Modulo 4: Generazione del Feedback (Il Tutor \- Core AI 2\)**

Trasformazione del log errori algoritmico in un feedback testuale per lo studente.

* **Task Operativi:** Prompt Engineering (o fine-tuning leggero) su un Large Language Model.  
* **Input al LLM:** Il contesto (es. "Esercizio: Scala di Do Maggiore a 120 BPM") e il JSON degli errori prodotto dal DTW (es. \[{"tempo": 2.5, "atteso": "DO", "suonato": "RE", "status": "wrong\_pitch"}\]).  
* **Output del LLM:** Un testo empatico e pedagogico.  
* **Librerie Python:**  
  * transformers (Hugging Face) per usare modelli in locale.  
  * Librerie API (es. openai o mistralai) se decidete di usare modelli cloud-based per snellire l'esecuzione.

