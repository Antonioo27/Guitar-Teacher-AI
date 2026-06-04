<script setup>
defineProps({
  status: { type: [Object, null], default: null },
})

defineEmits(['refresh'])
</script>

<template>
  <div class="status-bar" v-if="status">
    <div class="status-indicator" :class="status.status === 'ok' ? 'online' : 'offline'">
      <span class="status-dot"></span>
      <span class="status-text">
        {{ status.status === 'ok' ? 'Server Online' : 'Server Offline' }}
      </span>
    </div>
    <div v-if="status.status === 'ok'" class="status-details">
      <span v-if="status.weights_available" class="badge badge-success">Modello ✓</span>
      <span v-else class="badge badge-warning">Pesi mancanti</span>
    </div>
    <button class="refresh-btn" @click="$emit('refresh')" title="Ricontrolla">
      ↻
    </button>
  </div>
</template>

<style scoped>
.status-bar {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.online .status-dot {
  background: var(--success);
  box-shadow: 0 0 8px var(--success);
  animation: pulse-dot 2s ease infinite;
}

.offline .status-dot {
  background: var(--danger);
}

@keyframes pulse-dot {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.status-text {
  font-size: 0.75rem;
  color: var(--text-muted);
  font-weight: 500;
}

.status-details {
  display: flex;
  gap: 0.5rem;
}

.refresh-btn {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  border: 1px solid var(--border-color);
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all var(--transition-fast);
}

.refresh-btn:hover {
  border-color: var(--accent-primary);
  color: var(--accent-primary);
  transform: rotate(180deg);
}
</style>
