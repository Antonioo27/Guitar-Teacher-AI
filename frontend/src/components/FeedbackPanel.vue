<script setup>
defineProps({
  feedback: { type: String, required: true },
})
</script>

<template>
  <div class="card feedback-card">
    <div class="feedback-header">
      <span class="feedback-icon">🎓</span>
      <h3>Feedback del Tutor AI</h3>
    </div>
    <div class="feedback-body">
      <p v-for="(paragraph, i) in feedback.split('\n').filter(p => p.trim())" :key="i" class="feedback-paragraph">
        {{ paragraph }}
      </p>
    </div>
  </div>
</template>

<style scoped>
.feedback-card {
  border-color: rgba(245, 158, 11, 0.2);
  background: linear-gradient(135deg, rgba(26, 31, 53, 0.9), rgba(245, 158, 11, 0.03));
}

.feedback-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1.25rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border-color);
}

.feedback-icon {
  font-size: 1.75rem;
}

.feedback-header h3 {
  font-size: 1.15rem;
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.feedback-body {
  line-height: 1.75;
}

.feedback-paragraph {
  margin-bottom: 0.75rem;
  color: var(--text-secondary);
  font-size: 0.95rem;
}

.feedback-paragraph:last-child {
  margin-bottom: 0;
}
</style>
