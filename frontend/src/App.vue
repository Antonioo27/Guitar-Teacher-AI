<script setup>
import { ref, computed } from 'vue'
import FileUpload from './components/FileUpload.vue'
import AnalysisResults from './components/AnalysisResults.vue'
import FeedbackPanel from './components/FeedbackPanel.vue'
import StatusBar from './components/StatusBar.vue'

const API_BASE = 'http://localhost:8000'

// State
const audioFile = ref(null)
const referenceFile = ref(null)
const exerciseContext = ref('')
const generateFeedback = ref(true)

const isLoading = ref(false)
const loadingMessage = ref('')
const results = ref(null)
const error = ref(null)
const serverStatus = ref(null)

// Computed
const canAnalyze = computed(() => {
  return audioFile.value && referenceFile.value && !isLoading.value
})

// Check server health on mount
async function checkServer() {
  try {
    const res = await fetch(`${API_BASE}/api/health`)
    serverStatus.value = await res.json()
  } catch {
    serverStatus.value = { status: 'offline' }
  }
}
checkServer()

// Run analysis
async function runAnalysis() {
  if (!canAnalyze.value) return

  isLoading.value = true
  error.value = null
  results.value = null
  loadingMessage.value = 'Caricamento file audio...'

  const formData = new FormData()
  formData.append('audio', audioFile.value)
  formData.append('reference', referenceFile.value)
  formData.append('context', exerciseContext.value)
  formData.append('generate_feedback', generateFeedback.value)

  try {
    loadingMessage.value = 'Trascrizione audio con TabCNN...'

    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      body: formData,
    })

    if (!res.ok) {
      const errData = await res.json()
      throw new Error(errData.detail || `Errore ${res.status}`)
    }

    loadingMessage.value = 'Elaborazione risultati...'
    results.value = await res.json()
  } catch (e) {
    error.value = e.message
  } finally {
    isLoading.value = false
    loadingMessage.value = ''
  }
}

function resetAll() {
  audioFile.value = null
  referenceFile.value = null
  exerciseContext.value = ''
  results.value = null
  error.value = null
}
</script>

<template>
  <div class="app-container">
    <!-- Header -->
    <header class="app-header fade-in">
      <div class="header-content">
        <div class="logo">
          <span class="logo-icon">🎸</span>
          <div>
            <h1>AI Guitar Tutor</h1>
            <p class="subtitle">Trascrizione Neurale & Valutazione Esecutiva</p>
          </div>
        </div>
        <StatusBar :status="serverStatus" @refresh="checkServer" />
      </div>
    </header>

    <main class="app-main">
      <!-- Upload Section -->
      <section v-if="!results" class="upload-section fade-in">
        <div class="upload-grid">
          <!-- Audio File -->
          <FileUpload
            id="audio-upload"
            label="Registrazione Studente"
            description="Carica il file audio dell'esecuzione"
            accept=".wav,.mp3,.flac"
            icon="🎵"
            :file="audioFile"
            @update:file="audioFile = $event"
          />

          <!-- Reference File -->
          <FileUpload
            id="reference-upload"
            label="Spartito di Riferimento"
            description="Carica lo spartito MIDI o le annotazioni JAMS"
            accept=".mid,.midi,.jams"
            icon="📄"
            :file="referenceFile"
            @update:file="referenceFile = $event"
          />
        </div>

        <!-- Exercise Context -->
        <div class="context-section card">
          <label for="exercise-context" class="context-label">
            <span class="context-icon">📝</span>
            Contesto dell'esercizio
            <span class="optional-tag">opzionale</span>
          </label>
          <input
            id="exercise-context"
            v-model="exerciseContext"
            type="text"
            class="context-input"
            placeholder='Es: "Scala di Do Maggiore a 120 BPM", "Arpeggio di Sol Minore"'
          />
          <div class="context-options">
            <label class="checkbox-label" for="feedback-toggle">
              <input
                id="feedback-toggle"
                v-model="generateFeedback"
                type="checkbox"
              />
              <span class="checkbox-custom"></span>
              Genera feedback con AI (OpenAI)
            </label>
          </div>
        </div>

        <!-- Action Button -->
        <div class="action-row">
          <button
            id="analyze-btn"
            class="btn btn-primary btn-analyze"
            :disabled="!canAnalyze"
            @click="runAnalysis"
          >
            <span class="btn-icon">⚡</span>
            Analizza Esecuzione
          </button>
        </div>

        <!-- Error -->
        <div v-if="error" class="error-banner fade-in">
          <span class="error-icon">⚠️</span>
          <div>
            <strong>Errore</strong>
            <p>{{ error }}</p>
          </div>
          <button class="error-dismiss" @click="error = null">✕</button>
        </div>
      </section>

      <!-- Loading -->
      <section v-if="isLoading" class="loading-section fade-in">
        <div class="loading-card card">
          <div class="spinner"></div>
          <h3>Analisi in corso...</h3>
          <p class="loading-message">{{ loadingMessage }}</p>
          <div class="loading-steps">
            <div class="step" :class="{ active: loadingMessage.includes('audio') }">
              <span class="step-num">1</span> Preprocessing CQT
            </div>
            <div class="step" :class="{ active: loadingMessage.includes('TabCNN') }">
              <span class="step-num">2</span> Inferenza TabCNN
            </div>
            <div class="step" :class="{ active: loadingMessage.includes('risultati') }">
              <span class="step-num">3</span> DTW + Feedback
            </div>
          </div>
        </div>
      </section>

      <!-- Results -->
      <section v-if="results && !isLoading" class="results-section fade-in-stagger">
        <AnalysisResults :results="results" />
        <FeedbackPanel
          v-if="results.feedback"
          :feedback="results.feedback"
        />
        <div class="action-row">
          <button class="btn btn-secondary" @click="resetAll">
            ← Nuova Analisi
          </button>
        </div>
      </section>
    </main>

    <!-- Footer -->
    <footer class="app-footer">
      <p>AI Guitar Tutor — Progetto Intelligenza Artificiale, UniBo</p>
    </footer>
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

/* Header */
.app-header {
  padding: 1.5rem 2rem;
  border-bottom: 1px solid var(--border-color);
  background: var(--bg-glass);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  position: sticky;
  top: 0;
  z-index: 100;
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.logo {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.logo-icon {
  font-size: 2.5rem;
  filter: drop-shadow(0 0 10px rgba(245, 158, 11, 0.4));
}

.logo h1 {
  font-size: 1.5rem;
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.subtitle {
  font-size: 0.8rem;
  color: var(--text-muted);
  font-weight: 400;
}

/* Main */
.app-main {
  flex: 1;
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  width: 100%;
}

/* Upload Grid */
.upload-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  margin-bottom: 1.5rem;
}

@media (max-width: 768px) {
  .upload-grid {
    grid-template-columns: 1fr;
  }
}

/* Context */
.context-section {
  margin-bottom: 1.5rem;
}

.context-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
  margin-bottom: 0.75rem;
  font-size: 0.95rem;
}

.context-icon {
  font-size: 1.2rem;
}

.optional-tag {
  font-size: 0.7rem;
  font-weight: 500;
  color: var(--text-muted);
  background: var(--bg-secondary);
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
}

.context-input {
  width: 100%;
  padding: 0.75rem 1rem;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  color: var(--text-primary);
  font-family: var(--font-family);
  font-size: 0.9rem;
  outline: none;
  transition: border-color var(--transition-fast);
}

.context-input:focus {
  border-color: var(--accent-primary);
}

.context-input::placeholder {
  color: var(--text-muted);
}

.context-options {
  margin-top: 0.75rem;
}

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.checkbox-label input[type="checkbox"] {
  display: none;
}

.checkbox-custom {
  width: 18px;
  height: 18px;
  border: 2px solid var(--text-muted);
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all var(--transition-fast);
  flex-shrink: 0;
}

.checkbox-label input:checked + .checkbox-custom {
  background: var(--accent-primary);
  border-color: var(--accent-primary);
}

.checkbox-label input:checked + .checkbox-custom::after {
  content: '✓';
  font-size: 12px;
  color: #0a0e1a;
  font-weight: 700;
}

/* Action */
.action-row {
  display: flex;
  justify-content: center;
  padding: 1rem 0;
}

.btn-analyze {
  padding: 1rem 2.5rem;
  font-size: 1.1rem;
  border-radius: var(--radius-lg);
}

.btn-icon {
  font-size: 1.2rem;
}

/* Error */
.error-banner {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  padding: 1rem 1.25rem;
  background: var(--danger-bg);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: var(--radius-md);
  margin-top: 1rem;
}

.error-icon { font-size: 1.25rem; }

.error-banner strong {
  color: var(--danger);
  display: block;
  margin-bottom: 0.25rem;
}

.error-banner p {
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.error-dismiss {
  margin-left: auto;
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 1rem;
  padding: 0.25rem;
}

/* Loading */
.loading-section {
  display: flex;
  justify-content: center;
  padding: 3rem 0;
}

.loading-card {
  text-align: center;
  padding: 3rem;
  max-width: 400px;
  width: 100%;
  animation: pulse-glow 2s ease infinite;
}

.spinner {
  width: 48px;
  height: 48px;
  border: 3px solid var(--border-color);
  border-top-color: var(--accent-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 0 auto 1.5rem;
}

.loading-message {
  color: var(--text-secondary);
  font-size: 0.9rem;
  margin-top: 0.5rem;
}

.loading-steps {
  display: flex;
  gap: 1rem;
  margin-top: 1.5rem;
  justify-content: center;
}

.step {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.75rem;
  color: var(--text-muted);
  transition: color var(--transition-normal);
}

.step.active {
  color: var(--accent-primary);
}

.step-num {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--bg-secondary);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.65rem;
  font-weight: 700;
}

.step.active .step-num {
  background: var(--accent-primary);
  color: #0a0e1a;
}

/* Footer */
.app-footer {
  text-align: center;
  padding: 1.5rem;
  border-top: 1px solid var(--border-color);
  color: var(--text-muted);
  font-size: 0.8rem;
}
</style>
