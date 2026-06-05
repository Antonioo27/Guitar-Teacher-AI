<script setup>
import { computed, ref } from 'vue'
import PianoRollChart from './PianoRollChart.vue'

const props = defineProps({
  results: { type: Object, required: true },
})

const showChart = ref(false)

const summary = computed(() => props.results?.error_log?.summary || {})
const errors = computed(() => props.results?.error_log?.errors || [])
const predictedCount = computed(() => props.results?.predicted_notes?.length || 0)
const referenceCount = computed(() => props.results?.reference_notes?.length || 0)

const accuracy = computed(() => summary.value.accuracy_percent || 0)
const accuracyColor = computed(() => {
  if (accuracy.value >= 80) return 'var(--success)'
  if (accuracy.value >= 50) return 'var(--warning)'
  return 'var(--danger)'
})

function statusLabel(status) {
  const map = {
    correct: 'Corretta',
    wrong_timing: 'Fuori Tempo',
    wrong_pitch: 'Pitch Sbagliato',
    missing: 'Mancante',
    extra: 'Extra',
  }
  return map[status] || status
}

function statusBadgeClass(status) {
  const map = {
    correct: 'badge-success',
    wrong_timing: 'badge-warning',
    wrong_pitch: 'badge-danger',
    missing: 'badge-danger',
    extra: 'badge-info',
  }
  return map[status] || 'badge-info'
}
</script>

<template>
  <div class="results-container">
    <!-- Summary Cards -->
    <div class="summary-grid">
      <!-- Accuracy Gauge -->
      <div class="card summary-card accuracy-card">
        <div class="gauge">
          <svg viewBox="0 0 120 120" class="gauge-svg">
            <circle cx="60" cy="60" r="50" class="gauge-bg" />
            <circle
              cx="60" cy="60" r="50"
              class="gauge-fill"
              :style="{
                strokeDasharray: `${accuracy * 3.14} 314`,
                stroke: accuracyColor
              }"
            />
          </svg>
          <div class="gauge-value">
            <span class="gauge-number" :style="{ color: accuracyColor }">
              {{ accuracy }}
            </span>
            <span class="gauge-unit">%</span>
          </div>
        </div>
        <p class="summary-label">Accuratezza</p>
      </div>

      <!-- Stats -->
      <div class="card summary-card">
        <div class="stat-icon">🎵</div>
        <div class="stat-value">{{ predictedCount }}</div>
        <div class="stat-label">Note Trascritte</div>
      </div>

      <div class="card summary-card">
        <div class="stat-icon">📄</div>
        <div class="stat-value">{{ referenceCount }}</div>
        <div class="stat-label">Note Spartito</div>
      </div>

      <div class="card summary-card">
        <div class="stat-icon">❌</div>
        <div class="stat-value">{{ errors.length }}</div>
        <div class="stat-label">Errori Trovati</div>
      </div>
    </div>

    <!-- Piano Roll Visualization Toggle -->
    <div class="chart-toggle-row">
      <button class="btn-toggle-chart" @click="showChart = !showChart">
        {{ showChart ? '⬆ Nascondi Grafico Piano Roll' : '📊 Visualizza Grafico Piano Roll' }}
      </button>
    </div>

    <!-- Piano Roll Component -->
    <PianoRollChart 
      v-if="showChart"
      :predicted-notes="results?.predicted_notes || []"
      :reference-notes="results?.reference_notes || []"
      :global-offset="summary.estimated_global_offset_sec || 0"
    />

    <!-- Error Breakdown -->
    <div class="card breakdown-card">
      <h3 class="section-title">📊 Distribuzione Errori</h3>
      <div class="breakdown-bars">
        <div class="bar-row" v-if="summary.correct">
          <span class="bar-label badge badge-success">✓ Corrette</span>
          <div class="bar-track">
            <div
              class="bar-fill bar-success"
              :style="{ width: (summary.correct / summary.total_notes_evaluated * 100) + '%' }"
            ></div>
          </div>
          <span class="bar-value">{{ summary.correct }}</span>
        </div>
        <div class="bar-row" v-if="summary.wrong_timing">
          <span class="bar-label badge badge-warning">⏱ Fuori Tempo</span>
          <div class="bar-track">
            <div
              class="bar-fill bar-warning"
              :style="{ width: (summary.wrong_timing / summary.total_notes_evaluated * 100) + '%' }"
            ></div>
          </div>
          <span class="bar-value">{{ summary.wrong_timing }}</span>
        </div>
        <div class="bar-row" v-if="summary.wrong_pitch">
          <span class="bar-label badge badge-danger">🎹 Pitch Sbagliato</span>
          <div class="bar-track">
            <div
              class="bar-fill bar-danger"
              :style="{ width: (summary.wrong_pitch / summary.total_notes_evaluated * 100) + '%' }"
            ></div>
          </div>
          <span class="bar-value">{{ summary.wrong_pitch }}</span>
        </div>
        <div class="bar-row" v-if="summary.missing">
          <span class="bar-label badge badge-danger">⬜ Mancanti</span>
          <div class="bar-track">
            <div
              class="bar-fill bar-danger"
              :style="{ width: (summary.missing / summary.total_notes_evaluated * 100) + '%' }"
            ></div>
          </div>
          <span class="bar-value">{{ summary.missing }}</span>
        </div>
        <div class="bar-row" v-if="summary.extra">
          <span class="bar-label badge badge-info">➕ Extra</span>
          <div class="bar-track">
            <div
              class="bar-fill bar-info"
              :style="{ width: (summary.extra / summary.total_notes_evaluated * 100) + '%' }"
            ></div>
          </div>
          <span class="bar-value">{{ summary.extra }}</span>
        </div>
      </div>
    </div>

    <!-- Error Details Table -->
    <div v-if="errors.length" class="card table-card">
      <h3 class="section-title">📋 Dettaglio Errori</h3>
      <div class="table-wrapper">
        <table class="error-table">
          <thead>
            <tr>
              <th>Tempo (s)</th>
              <th>Atteso</th>
              <th>Suonato</th>
              <th>Stato</th>
              <th>Delta t</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(err, idx) in errors" :key="idx">
              <td class="mono">{{ err.time?.toFixed(2) }}</td>
              <td>
                <span v-if="err.expected" class="note-expected">{{ err.expected }}</span>
                <span v-else class="note-empty">—</span>
              </td>
              <td>
                <span v-if="err.played" class="note-played">{{ err.played }}</span>
                <span v-else class="note-empty">—</span>
              </td>
              <td>
                <span class="badge" :class="statusBadgeClass(err.status)">
                  {{ statusLabel(err.status) }}
                </span>
              </td>
              <td class="mono">
                {{ err.delta_t != null ? (err.delta_t > 0 ? '+' : '') + err.delta_t.toFixed(3) : '—' }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<style scoped>
.results-container {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

/* Summary Grid */
.summary-grid {
  display: grid;
  grid-template-columns: 1.3fr 1fr 1fr 1fr;
  gap: 1rem;
}

@media (max-width: 768px) {
  .summary-grid {
    grid-template-columns: 1fr 1fr;
  }
}

.summary-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 1.5rem 1rem;
}

.summary-label {
  font-size: 0.85rem;
  color: var(--text-secondary);
  margin-top: 0.5rem;
  font-weight: 500;
}

/* Gauge */
.gauge {
  position: relative;
  width: 100px;
  height: 100px;
}

.gauge-svg {
  transform: rotate(-90deg);
  width: 100%;
  height: 100%;
}

.gauge-bg {
  fill: none;
  stroke: var(--bg-secondary);
  stroke-width: 8;
}

.gauge-fill {
  fill: none;
  stroke-width: 8;
  stroke-linecap: round;
  transition: stroke-dasharray 1s ease;
}

.gauge-value {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

.gauge-number {
  font-size: 1.75rem;
  font-weight: 800;
}

.gauge-unit {
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-top: 0.25rem;
}

/* Stats */
.stat-icon {
  font-size: 1.75rem;
  margin-bottom: 0.5rem;
}

.stat-value {
  font-size: 2rem;
  font-weight: 800;
  color: var(--text-primary);
  line-height: 1;
}

.stat-label {
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-top: 0.35rem;
  font-weight: 500;
}

/* Breakdown */
.section-title {
  margin-bottom: 1.25rem;
  font-size: 1.1rem;
}

.breakdown-bars {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.bar-row {
  display: grid;
  grid-template-columns: 160px 1fr 40px;
  align-items: center;
  gap: 0.75rem;
}

.bar-label {
  font-size: 0.75rem;
  justify-self: start;
}

.bar-track {
  height: 8px;
  background: var(--bg-secondary);
  border-radius: 4px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.8s ease;
  min-width: 4px;
}

.bar-success { background: var(--success); }
.bar-warning { background: var(--warning); }
.bar-danger  { background: var(--danger); }
.bar-info    { background: var(--info); }

.bar-value {
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--text-secondary);
  text-align: right;
}

/* Table */
.table-wrapper {
  overflow-x: auto;
  max-height: 400px;
  overflow-y: auto;
}

.error-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

.error-table th {
  text-align: left;
  padding: 0.75rem;
  border-bottom: 2px solid var(--border-color);
  color: var(--text-muted);
  font-weight: 600;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  position: sticky;
  top: 0;
  background: var(--bg-card);
  z-index: 1;
}

.error-table td {
  padding: 0.6rem 0.75rem;
  border-bottom: 1px solid var(--border-color);
}

.error-table tbody tr:hover {
  background: rgba(245, 158, 11, 0.03);
}

.mono {
  font-family: 'SF Mono', 'Menlo', monospace;
  font-size: 0.8rem;
}

.note-expected {
  color: var(--text-primary);
  font-weight: 600;
}

.note-played {
  color: var(--accent-primary);
  font-weight: 600;
}

.note-empty {
  color: var(--text-muted);
}

/* Chart Toggle */
.chart-toggle-row {
  display: flex;
  justify-content: center;
  margin: 0.5rem 0;
}

.btn-toggle-chart {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: 0.75rem 1.5rem;
  border-radius: var(--radius-md);
  font-weight: 600;
  cursor: pointer;
  transition: all var(--transition-fast);
}

.btn-toggle-chart:hover {
  background: var(--bg-glass);
  border-color: var(--accent-primary);
  color: var(--accent-primary);
}
</style>
