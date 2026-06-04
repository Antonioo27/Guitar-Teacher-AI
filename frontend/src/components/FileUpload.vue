<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  id: { type: String, required: true },
  label: { type: String, required: true },
  description: { type: String, default: '' },
  accept: { type: String, default: '' },
  icon: { type: String, default: '📁' },
  file: { type: [File, null], default: null },
})

const emit = defineEmits(['update:file'])

const isDragging = ref(false)
const fileInput = ref(null)

const fileName = computed(() => props.file?.name || null)
const fileSize = computed(() => {
  if (!props.file) return null
  const bytes = props.file.size
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
})

function onDrop(e) {
  isDragging.value = false
  const file = e.dataTransfer?.files[0]
  if (file) emit('update:file', file)
}

function onFileSelect(e) {
  const file = e.target.files[0]
  if (file) emit('update:file', file)
}

function openFileDialog() {
  fileInput.value?.click()
}

function removeFile() {
  emit('update:file', null)
  if (fileInput.value) fileInput.value.value = ''
}
</script>

<template>
  <div
    :id="id"
    class="upload-card card"
    :class="{ dragging: isDragging, 'has-file': file }"
    @dragenter.prevent="isDragging = true"
    @dragover.prevent="isDragging = true"
    @dragleave.prevent="isDragging = false"
    @drop.prevent="onDrop"
    @click="!file && openFileDialog()"
  >
    <input
      ref="fileInput"
      type="file"
      :accept="accept"
      hidden
      @change="onFileSelect"
    />

    <!-- Empty state -->
    <div v-if="!file" class="upload-empty">
      <span class="upload-icon">{{ icon }}</span>
      <h3 class="upload-label">{{ label }}</h3>
      <p class="upload-desc">{{ description }}</p>
      <p class="upload-hint">
        Trascina qui o <span class="upload-link">sfoglia</span>
      </p>
      <p class="upload-formats">{{ accept }}</p>
    </div>

    <!-- File loaded -->
    <div v-else class="upload-loaded">
      <div class="file-info">
        <span class="file-icon">{{ icon }}</span>
        <div class="file-details">
          <span class="file-name">{{ fileName }}</span>
          <span class="file-size">{{ fileSize }}</span>
        </div>
      </div>
      <button class="remove-btn" @click.stop="removeFile" title="Rimuovi file">
        ✕
      </button>
    </div>
  </div>
</template>

<style scoped>
.upload-card {
  cursor: pointer;
  text-align: center;
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
  min-height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.upload-card::before {
  content: '';
  position: absolute;
  inset: 8px;
  border: 2px dashed var(--text-muted);
  border-radius: var(--radius-md);
  opacity: 0.3;
  transition: all var(--transition-normal);
  pointer-events: none;
}

.upload-card:hover::before,
.upload-card.dragging::before {
  border-color: var(--accent-primary);
  opacity: 0.6;
}

.upload-card.dragging {
  background: rgba(245, 158, 11, 0.05);
  border-color: var(--accent-primary);
}

.upload-card.has-file {
  cursor: default;
  border-color: var(--success);
  background: rgba(16, 185, 129, 0.05);
}

.upload-card.has-file::before {
  border-color: var(--success);
  opacity: 0.2;
}

/* Empty state */
.upload-empty {
  padding: 1rem;
}

.upload-icon {
  font-size: 2.5rem;
  display: block;
  margin-bottom: 0.75rem;
  filter: grayscale(0.3);
  transition: filter var(--transition-normal);
}

.upload-card:hover .upload-icon {
  filter: grayscale(0);
}

.upload-label {
  font-size: 1.1rem;
  margin-bottom: 0.35rem;
}

.upload-desc {
  color: var(--text-secondary);
  font-size: 0.85rem;
  margin-bottom: 0.75rem;
}

.upload-hint {
  font-size: 0.8rem;
  color: var(--text-muted);
}

.upload-link {
  color: var(--accent-primary);
  text-decoration: underline;
  cursor: pointer;
}

.upload-formats {
  margin-top: 0.5rem;
  font-size: 0.7rem;
  color: var(--text-muted);
  font-family: monospace;
}

/* File loaded */
.upload-loaded {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  padding: 0 0.5rem;
}

.file-info {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.file-icon {
  font-size: 2rem;
}

.file-details {
  text-align: left;
}

.file-name {
  display: block;
  font-weight: 600;
  font-size: 0.95rem;
  color: var(--text-primary);
  word-break: break-all;
}

.file-size {
  font-size: 0.8rem;
  color: var(--text-muted);
}

.remove-btn {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  border: 1px solid var(--border-color);
  background: var(--bg-secondary);
  color: var(--text-muted);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8rem;
  transition: all var(--transition-fast);
  flex-shrink: 0;
}

.remove-btn:hover {
  background: var(--danger-bg);
  border-color: var(--danger);
  color: var(--danger);
}
</style>
