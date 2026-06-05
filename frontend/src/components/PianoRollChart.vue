<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  predictedNotes: { type: Array, required: true },
  referenceNotes: { type: Array, required: true },
  globalOffset: { type: Number, default: 0 }
})

const containerRef = ref(null)

const minPitch = computed(() => {
  const allPitches = [...props.predictedNotes, ...props.referenceNotes].map(n => n.pitch || 40)
  return Math.min(...allPitches) - 2
})

const maxPitch = computed(() => {
  const allPitches = [...props.predictedNotes, ...props.referenceNotes].map(n => n.pitch || 40)
  return Math.max(...allPitches) + 2
})

const maxTime = computed(() => {
  let pMax = 0
  if (props.predictedNotes.length) {
    pMax = Math.max(...props.predictedNotes.map(n => n.time + (n.duration || 0.2)))
  }
  let rMax = 0
  if (props.referenceNotes.length) {
    rMax = Math.max(...props.referenceNotes.map(n => n.time + (n.duration || 0.2)))
  }
  return Math.max(pMax, rMax) + 1.0
})

const pitchRange = computed(() => maxPitch.value - minPitch.value)

// Scala e zoom
const zoomX = ref(150) // pixels per second
const zoomY = ref(20)  // pixels per semitone

// Offset Toggle
const applyOffset = ref(true)

// Posizionamento CSS
const getNoteStyle = (note, isPredicted) => {
  // Se applyOffset è true, trasliamo la predetta dell'offset calcolato. Altrimenti usiamo il tempo grezzo della rete.
  const time = (isPredicted && applyOffset.value) ? Math.max(0, note.time - props.globalOffset) : note.time
  const duration = note.duration || 0.2
  const pitch = note.pitch

  const left = time * zoomX.value
  const width = Math.max(duration * zoomX.value, 4) // almeno 4px di larghezza
  const bottom = (pitch - minPitch.value) * zoomY.value
  const height = zoomY.value * 0.8 // margine per far respirare le note

  return {
    left: `${left}px`,
    bottom: `${bottom}px`,
    width: `${width}px`,
    height: `${height}px`,
  }
}

// Griglia
const pitchLines = computed(() => {
  const lines = []
  for (let p = minPitch.value; p <= maxPitch.value; p++) {
    lines.push({ pitch: p, bottom: (p - minPitch.value) * zoomY.value })
  }
  return lines
})

const timeLines = computed(() => {
  const lines = []
  const step = 0.5 // Ogni mezzo secondo
  for (let t = 0; t <= maxTime.value; t += step) {
    lines.push({ time: t, left: t * zoomX.value })
  }
  return lines
})
</script>

<template>
  <div class="piano-roll-container">
    <div class="controls">
      <label>Zoom X: <input type="range" min="50" max="400" v-model="zoomX" /></label>
      <label>Zoom Y: <input type="range" min="10" max="40" v-model="zoomY" /></label>
      <label class="toggle-offset">
        <input type="checkbox" v-model="applyOffset" /> Applica Offset (Simula DTW)
      </label>
      <div class="legend">
        <span class="legend-item ref">Spartito / Ground Truth (.jams)</span>
        <span class="legend-item pred">Audio Esecuzione (.wav)</span>
      </div>
    </div>
    
    <div class="scroll-wrapper" ref="containerRef">
      <div 
        class="piano-roll-canvas" 
        :style="{ width: `${maxTime * zoomX}px`, height: `${pitchRange * zoomY}px` }"
      >
        <!-- Griglia orizzontale (pitch) -->
        <div 
          v-for="line in pitchLines" 
          :key="'p'+line.pitch" 
          class="grid-line-h"
          :style="{ bottom: `${line.bottom}px`, height: `${zoomY}px` }"
        >
          <span class="pitch-label">{{ line.pitch }}</span>
        </div>

        <!-- Griglia verticale (tempo) -->
        <div 
          v-for="line in timeLines" 
          :key="'t'+line.time" 
          class="grid-line-v"
          :style="{ left: `${line.left}px` }"
        >
          <span class="time-label">{{ line.time }}s</span>
        </div>

        <!-- Note Ground Truth -->
        <div 
          v-for="(note, idx) in referenceNotes" 
          :key="'ref'+idx"
          class="note ref-note"
          :style="getNoteStyle(note, false)"
          :title="`Spartito: ${note.note_name} (${note.pitch}) @ ${note.time.toFixed(2)}s`"
        ></div>

        <!-- Note Predette -->
        <div 
          v-for="(note, idx) in predictedNotes" 
          :key="'pred'+idx"
          class="note pred-note"
          :style="getNoteStyle(note, true)"
          :title="`Esecuzione: ${note.note_name} (${note.pitch}) @ ${note.time.toFixed(2)}s`"
        ></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.piano-roll-container {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 1.5rem;
  margin-top: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.controls {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border-color);
  flex-wrap: wrap;
}

.controls label {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.9rem;
  color: var(--text-secondary);
}

.controls input[type="range"] {
  accent-color: var(--accent-primary);
}

.legend {
  margin-left: auto;
  display: flex;
  gap: 1.5rem;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-primary);
}

.legend-item::before {
  content: '';
  display: block;
  width: 14px;
  height: 14px;
  border-radius: 3px;
}

.legend-item.ref::before {
  background: rgba(16, 185, 129, 0.4);
  border: 1px solid rgba(16, 185, 129, 0.8);
}

.legend-item.pred::before {
  background: rgba(59, 130, 246, 0.6); /* blu acceso */
  border: 1px solid rgba(59, 130, 246, 1);
}

.scroll-wrapper {
  overflow: auto;
  max-height: 450px;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  background: #0f172a; /* Sfondo scuro navy */
  position: relative;
  box-shadow: inset 0 2px 10px rgba(0,0,0,0.5);
}

.scroll-wrapper::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}
.scroll-wrapper::-webkit-scrollbar-track {
  background: #1e293b;
}
.scroll-wrapper::-webkit-scrollbar-thumb {
  background: #475569;
  border-radius: 5px;
}

.piano-roll-canvas {
  position: relative;
  min-width: 100%;
}

.grid-line-h {
  position: absolute;
  left: 0;
  right: 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.04);
  pointer-events: none;
}

.pitch-label {
  position: absolute;
  left: 6px;
  bottom: 2px;
  font-size: 0.65rem;
  color: rgba(255, 255, 255, 0.3);
  font-family: monospace;
}

.grid-line-v {
  position: absolute;
  top: 0;
  bottom: 0;
  border-left: 1px solid rgba(255, 255, 255, 0.04);
  pointer-events: none;
}

.time-label {
  position: absolute;
  top: 6px;
  left: 6px;
  font-size: 0.65rem;
  color: rgba(255, 255, 255, 0.3);
  font-family: monospace;
}

.note {
  position: absolute;
  border-radius: 3px;
  transition: filter 0.1s;
  cursor: pointer;
}

.note:hover {
  filter: brightness(1.4) drop-shadow(0 0 4px currentColor);
  z-index: 10 !important;
}

.ref-note {
  background: rgba(16, 185, 129, 0.35); /* Verde */
  border: 1px solid rgba(16, 185, 129, 0.9);
  z-index: 1;
}

.pred-note {
  background: rgba(59, 130, 246, 0.65); /* Blu */
  border: 1px solid rgba(59, 130, 246, 1);
  z-index: 2;
}
</style>
