const mediaInput = document.getElementById('media-file');
const drop = document.getElementById('drop');
const video = document.getElementById('video-preview');
const audio = document.getElementById('audio-preview');
const empty = document.getElementById('media-empty');
const fileName = document.getElementById('file-name');
const durationLabel = document.getElementById('file-duration');
const startInput = document.getElementById('segment-start');
const endInput = document.getElementById('segment-end');
const selection = document.getElementById('selection');
const waveform = document.getElementById('waveform');
const timelineStatus = document.getElementById('timeline-status');
const referenceText = document.getElementById('reference-text');
const transcriptStatus = document.getElementById('transcript-status');
const ttsMode = document.getElementById('tts-mode');
const voiceFileName = document.getElementById('voice-file-name');
const referenceFileInput = document.getElementById('speaker-reference-file');
const referenceStartInput = document.getElementById('reference-start');
const referenceSecondsInput = document.getElementById('reference-seconds');
const selectedSpeakerLabel = document.getElementById('selected-speaker');
const analyzeButton = document.getElementById('analyze-btn');
const sentenceBatchButton = document.getElementById('sentence-batch-btn');
const fullBatchButton = document.getElementById('full-batch-btn');

let sourceFile = null;
let referenceSourceFile = null;
let sourceUrl = '';
let mediaDuration = 0;
let segments = [];
let analysisTimeline = null;
let analysisJobId = '';
let selectedIndex = -1;

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function fmt(seconds) {
  seconds = Math.max(0, Number(seconds) || 0);
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function setStatus(node, text, kind = '') {
  node.className = `status${kind ? ` ${kind}` : ''}`;
  node.textContent = text;
}

function markStep(number) {
  const step = document.querySelector(`.step:nth-child(${number})`);
  if (step) step.classList.add('active');
}

async function jsonResponse(response) {
  let result = {};
  try { result = await response.json(); } catch (_) {}
  if (!response.ok) throw Error(result.detail || result.message || `请求失败（${response.status}）`);
  return result;
}

function clampSelection() {
  if (!mediaDuration) return;
  let start = Math.max(0, Math.min(mediaDuration, Number(startInput.value) || 0));
  let end = Math.max(0, Math.min(mediaDuration, Number(endInput.value) || 0));
  if (end <= start) end = Math.min(mediaDuration, start + 0.5);
  startInput.value = start.toFixed(1).replace(/\.0$/, '');
  endInput.value = end.toFixed(1).replace(/\.0$/, '');
  const left = start / mediaDuration * 100;
  const width = (end - start) / mediaDuration * 100;
  selection.style.left = `${left}%`;
  selection.style.width = `${width}%`;
  document.getElementById('selected-chip').textContent = `${fmt(start)} – ${fmt(end)}`;
}

function drawWaveform(values = null) {
  const ctx = waveform.getContext('2d');
  const w = waveform.width;
  const h = waveform.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0c121c';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#23344b';
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();
  const count = Math.floor(w / 5);
  ctx.fillStyle = '#4b759f';
  for (let i = 0; i < count; i++) {
    const value = values?.length
      ? values[i % values.length]
      : (0.13 + Math.abs(Math.sin(i * .73)) * .28 + Math.abs(Math.sin(i * .19)) * .13);
    const bar = Math.max(3, value * h * .75);
    ctx.fillRect(i * 5, (h - bar) / 2, 2, bar);
  }
}

async function drawFileWaveform(file) {
  try {
    const buffer = await file.arrayBuffer();
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) throw Error('浏览器不支持音频解码');
    const context = new AudioContextClass();
    const decoded = await context.decodeAudioData(buffer);
    const data = decoded.getChannelData(0);
    const values = [];
    const block = Math.max(1, Math.floor(data.length / 600));
    for (let i = 0; i < 600; i++) {
      let peak = 0;
      const from = i * block;
      for (let j = from; j < Math.min(data.length, from + block); j += 4) peak = Math.max(peak, Math.abs(data[j]));
      values.push(peak);
    }
    drawWaveform(values);
    await context.close();
  } catch (_) {
    drawWaveform();
  }
}

function resetAnalysis() {
  analysisJobId = '';
  analysisTimeline = null;
  segments = [];
  selectedIndex = -1;
  renderSegments();
  selectedSpeakerLabel.textContent = '还没有选择说话人';
  document.getElementById('selected-chip').textContent = '未选择';
}

function updateMedia(file) {
  if (!file) return;
  sourceFile = file;
  referenceSourceFile = null;
  referenceFileInput.value = '';
  resetAnalysis();
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = URL.createObjectURL(file);
  fileName.textContent = file.name;
  empty.hidden = true;
  video.hidden = true;
  audio.hidden = true;
  const isVideo = file.type.startsWith('video') || /\.(mp4|mov|mkv|webm|avi)$/i.test(file.name);
  const player = isVideo ? video : audio;
  player.src = sourceUrl;
  player.hidden = false;
  player.onloadedmetadata = () => {
    mediaDuration = Number(player.duration) || 0;
    durationLabel.textContent = fmt(mediaDuration);
    document.getElementById('ruler-middle').textContent = fmt(mediaDuration / 2);
    document.getElementById('ruler-end').textContent = fmt(mediaDuration);
    startInput.max = mediaDuration;
    endInput.max = mediaDuration;
    startInput.value = 0;
    endInput.value = Math.min(10, mediaDuration).toFixed(1).replace(/\.0$/, '');
    clampSelection();
    timelineStatus.textContent = '已加载媒体，可以分析整段或选择片段试听。';
  };
  voiceFileName.value = `${file.name}（按选中片段截取）`;
  referenceText.value = '';
  transcriptStatus.textContent = '选择 Zero-shot 后会自动识别参考台词。';
  drawFileWaveform(file);
}

function selectedSegment() {
  return selectedIndex >= 0 ? segments[selectedIndex] : null;
}

function updateSelectedSegment(index, { updateInputs = true } = {}) {
  if (index < 0 || index >= segments.length) return;
  selectedIndex = index;
  const item = segments[index];
  if (updateInputs) {
    startInput.value = item.start_sec;
    endInput.value = item.end_sec;
    document.getElementById('target-text').value = item.dubbing_text || item.text || '';
    const speaker = (analysisTimeline?.speakers || []).find(value => value.speaker_id === item.speaker_id);
    referenceText.value = speaker?.reference_text || '';
    selectedSpeakerLabel.textContent = `${item.speaker_label || item.speaker_id || '未确定说话人'}（${item.speaker_id || 'unknown'}）`;
    referenceStartInput.value = item.start_sec || 0;
    referenceSecondsInput.value = Math.max(1, Math.min(30, Number(item.end_sec) - Number(item.start_sec))).toFixed(1).replace(/\.0$/, '');
    clampSelection();
  }
  renderSegments();
}

function renderSegments() {
  const root = document.getElementById('segments');
  if (!segments.length) {
    root.innerHTML = '<div class="empty">分析后，识别出的说话片段会按时间戳显示在这里。</div>';
    return;
  }
  root.innerHTML = segments.map((item, index) => {
    const selected = index === selectedIndex ? 'selected' : '';
    const speaker = item.speaker_label || item.speaker_id || '未确定说话人';
    const confidence = Number(item.speaker_confidence || 0).toFixed(2);
    const ref = (analysisTimeline?.speakers || []).find(value => value.speaker_id === item.speaker_id);
    const refState = ref?.reference_audio ? '参考音色已绑定' : '待绑定参考音色';
    return `<div class="segment ${selected}" data-index="${index}">
      <span class="segment-time">${fmt(item.start_sec)}<br>${fmt(item.end_sec)}</span>
      <span>
        <span class="segment-text">${esc(speaker)} · ${esc(item.text || '待识别台词')}</span><br>
        <input class="segment-edit" data-index="${index}" value="${esc(item.dubbing_text || item.text || '')}" aria-label="${esc(item.segment_id)} 配音台词">
        <span class="segment-speaker">${esc(refState)} · 覆盖度 ${confidence}${item.overlap ? ' · 有重叠标记' : ''}</span>
      </span>
      <button class="ghost use-segment" type="button">选中</button>
    </div>`;
  }).join('');
}

async function fillTranscript() {
  const file = referenceSourceFile || sourceFile;
  if (!file || ttsMode.value !== 'zero_shot') return;
  transcriptStatus.textContent = '正在截取参考片段并自动识别参考台词…';
  transcriptStatus.className = 'help';
  const start = referenceSourceFile ? Number(referenceStartInput.value) || 0 : Number(startInput.value) || 0;
  const selectedDuration = Math.max(.5, (Number(endInput.value) || 0) - (Number(startInput.value) || 0));
  const seconds = referenceSourceFile ? Math.max(1, Math.min(30, Number(referenceSecondsInput.value) || 10)) : selectedDuration;
  const data = new FormData();
  data.append('reference_media', file);
  data.append('reference_start', start);
  data.append('reference_seconds', seconds);
  data.append('noise_reduction', '0');
  data.append('trim_leading_silence', 'true');
  try {
    const result = await jsonResponse(await fetch('/api/tts/reference-transcript', { method: 'POST', body: data }));
    referenceText.value = result.reference_text || '';
    transcriptStatus.textContent = '已由 Qwen3-ASR 自动填入；请确认或修改后再生成。';
    timelineStatus.textContent = '参考台词已就绪，可以生成片段试音。';
  } catch (error) {
    transcriptStatus.textContent = error.message || String(error);
    transcriptStatus.className = 'help status error';
  }
}

async function startAnalysis() {
  if (!sourceFile) {
    setStatus(timelineStatus, '请先选择视频或音频。', 'warn');
    return;
  }
  const buttons = [analyzeButton, sentenceBatchButton];
  buttons.forEach(button => { if (button) button.disabled = true; });
  setStatus(timelineStatus, '正在提交全片分析任务…');
  const data = new FormData();
  data.append('media', sourceFile);
  data.append('output_dir', 'F:\\voice');
  data.append('speaker_num', document.getElementById('analysis-speaker-num').value || '0');
  data.append('noise_reduction', document.getElementById('analysis-noise-reduction').value || '8');
  data.append('include_overlap', document.getElementById('analysis-overlap').checked ? 'true' : 'false');
  data.append('language', 'Chinese');
  try {
    const job = await jsonResponse(await fetch('/api/dubbing/analyze', { method: 'POST', body: data }));
    analysisJobId = job.job_id;
    segments = [];
    renderSegments();
    markStep(2);
    setStatus(timelineStatus, `任务 ${analysisJobId} 已开始：${job.output_dir || '输出到 F:\\voice'}\n正在等待 3D-Speaker 和 Qwen3-ASR…`);
    await pollJob('analysis');
  } catch (error) {
    setStatus(timelineStatus, error.message || String(error), 'error');
  } finally {
    buttons.forEach(button => { if (button) button.disabled = false; });
  }
}

async function refreshTimeline() {
  analysisTimeline = await jsonResponse(await fetch(`/api/dubbing/jobs/${analysisJobId}/timeline`));
  segments = analysisTimeline.segments || [];
  if (segments.length) updateSelectedSegment(0);
  else renderSegments();
  markStep(3);
}

function showGenerationFiles(job) {
  const root = document.getElementById('generated-links');
  if (!root) return;
  const urls = job.generation_urls || {};
  const links = [];
  if (urls.voice_track) links.push(`<a href="${esc(urls.voice_track)}" target="_blank">试听整段配音人声</a>`);
  if (urls.video) links.push(`<a href="${esc(urls.video)}" target="_blank">打开配音视频</a>`);
  root.innerHTML = links.join(' · ');
  if (urls.video) {
    video.src = `${urls.video}?t=${Date.now()}`;
    video.hidden = false;
    audio.hidden = true;
    empty.hidden = true;
  }
}

async function pollJob(kind) {
  if (!analysisJobId) return;
  const job = await jsonResponse(await fetch(`/api/dubbing/jobs/${analysisJobId}`));
  const prefix = kind === 'analysis' ? '分析' : '配音';
  const statusNode = kind === 'analysis' ? timelineStatus : document.getElementById('batch-status');
  if (job.status === 'failed') {
    setStatus(statusNode, `${prefix}失败：${job.error || '未知错误'}`, 'error');
    return;
  }
  if (job.status === 'done' && (kind === 'analysis' ? !job.generation : true)) {
    if (kind === 'analysis') {
      await refreshTimeline();
      const quality = job.quality || {};
      setStatus(timelineStatus, `分析完成：${job.speaker_count || 0} 个说话人，${job.segment_count || 0} 个台词片段。输出：${job.output_dir || 'F:\\voice'}`);
      setStatus(document.getElementById('batch-status'), '时间轴已生成。逐个选择说话人并绑定参考音色，然后点击“开始批量配音”。');
      if (quality.low_confidence_segment_count) setStatus(document.getElementById('batch-status'), `时间轴已生成，但有 ${quality.low_confidence_segment_count} 个片段覆盖度偏低，建议试听或调整说话人。`, 'warn');
    } else {
      showGenerationFiles(job);
      setStatus(document.getElementById('batch-status'), `批量配音完成：${job.generation_segment_count || job.generation?.segments?.length || 0} 个片段已合成。`, '');
      markStep(5);
    }
    return;
  }
  setStatus(statusNode, `${prefix}中：${job.phase || job.status_label || '处理中'}（${job.progress || 0}%）`);
  window.setTimeout(() => pollJob(kind).catch(error => setStatus(statusNode, error.message || String(error), 'error')), 2000);
}

async function attachReference() {
  if (!analysisJobId || !analysisTimeline) {
    setStatus(document.getElementById('tts-status'), '请先完成全片分析，再绑定说话人参考音色。', 'warn');
    return;
  }
  const item = selectedSegment();
  if (!item || !item.speaker_id || item.speaker_id === 'unknown') {
    setStatus(document.getElementById('tts-status'), '请先在时间轴中选择一个已识别的说话人。', 'warn');
    return;
  }
  const file = referenceSourceFile || sourceFile;
  if (!file) {
    setStatus(document.getElementById('tts-status'), '请选择参考音频或视频。', 'warn');
    return;
  }
  const button = document.getElementById('attach-reference-btn');
  button.disabled = true;
  setStatus(document.getElementById('tts-status'), `正在为 ${item.speaker_label || item.speaker_id} 提取参考音色并识别参考台词…`);
  const data = new FormData();
  data.append('reference_media', file);
  data.append('reference_start', referenceSourceFile ? (referenceStartInput.value || '0') : (startInput.value || '0'));
  data.append('reference_seconds', referenceSourceFile ? (referenceSecondsInput.value || '10') : Math.max(1, Math.min(30, Number(endInput.value) - Number(startInput.value))));
  data.append('noise_reduction', '0');
  data.append('trim_leading_silence', 'true');
  data.append('reference_text', ttsMode.value === 'zero_shot' ? referenceText.value : '');
  try {
    const result = await jsonResponse(await fetch(`/api/dubbing/jobs/${analysisJobId}/speakers/${encodeURIComponent(item.speaker_id)}/reference`, { method: 'POST', body: data }));
    analysisTimeline = await jsonResponse(await fetch(`/api/dubbing/jobs/${analysisJobId}/timeline`));
    segments = analysisTimeline.segments || segments;
    updateSelectedSegment(selectedIndex);
    referenceText.value = result.reference_text || '';
    voiceFileName.value = `${file.name}（${result.speaker_label}）`;
    setStatus(document.getElementById('tts-status'), `${result.speaker_label} 的参考音色已绑定：${result.reference_duration_sec}s。${result.reference_text ? '参考台词已填入。' : '没有识别到参考台词，请手工填写。'}`);
  } catch (error) {
    setStatus(document.getElementById('tts-status'), error.message || String(error), 'error');
  } finally {
    button.disabled = false;
  }
}

async function generatePreview() {
  const file = referenceSourceFile || sourceFile;
  if (!file) {
    setStatus(document.getElementById('tts-status'), '请先选择参考媒体。', 'warn');
    return;
  }
  const button = document.getElementById('preview-btn');
  button.disabled = true;
  setStatus(document.getElementById('tts-status'), '正在抽取参考片段并生成试音…');
  const data = new FormData();
  const start = referenceSourceFile ? Number(referenceStartInput.value) || 0 : Number(startInput.value) || 0;
  const end = Number(endInput.value) || 0;
  const seconds = referenceSourceFile ? Math.max(1, Math.min(30, Number(referenceSecondsInput.value) || 10)) : Math.max(.5, end - start);
  data.append('reference_media', file);
  data.append('output_dir', 'F:\\voice');
  data.append('mode', ttsMode.value);
  data.append('reference_text', ttsMode.value === 'zero_shot' ? referenceText.value : '');
  data.append('prompt_instruction', 'You are a helpful assistant.');
  data.append('target_text', document.getElementById('target-text').value);
  data.append('reference_start', start);
  data.append('reference_seconds', seconds);
  data.append('noise_reduction', '0');
  data.append('trim_leading_silence', 'true');
  data.append('speed', document.getElementById('tts-speed').value);
  data.append('seed', document.getElementById('tts-seed').value);
  data.append('text_frontend', 'true');
  data.append('target_duration', Math.max(.5, end - Number(startInput.value || 0)));
  data.append('fit_duration', document.getElementById('fit-duration').checked ? 'true' : 'false');
  try {
    const result = await jsonResponse(await fetch('/api/tts/direct', { method: 'POST', body: data }));
    setStatus(document.getElementById('tts-status'), result.message || '试音完成');
    document.getElementById('preview-box').hidden = false;
    document.getElementById('tts-preview').src = `${result.output_url}?t=${Date.now()}`;
    document.getElementById('tts-output-path').textContent = result.output_path || '';
    markStep(4);
  } catch (error) {
    setStatus(document.getElementById('tts-status'), error.message || String(error), 'error');
  } finally {
    button.disabled = false;
  }
}

async function startGeneration() {
  if (!analysisJobId || !analysisTimeline || !segments.length) {
    setStatus(document.getElementById('batch-status'), '请先点击“分析整段媒体”，生成可编辑时间轴。', 'warn');
    return;
  }
  segments.forEach((item, index) => {
    const input = document.querySelector(`.segment-edit[data-index="${index}"]`);
    if (input) item.dubbing_text = input.value.trim();
  });
  const button = fullBatchButton;
  button.disabled = true;
  setStatus(document.getElementById('batch-status'), '正在提交批量配音任务…');
  const payload = {
    edits: segments.map(item => ({
      segment_id: item.segment_id,
      text: item.text,
      dubbing_text: item.dubbing_text,
      speaker_id: item.speaker_id,
      start_sec: item.start_sec,
      end_sec: item.end_sec,
    })),
    selected_ids: null,
    mode: ttsMode.value,
    speed: Number(document.getElementById('tts-speed').value) || 1,
    adaptive_speed: document.getElementById('adaptive-speed').checked,
    fit_to_slot: document.getElementById('fit-duration').checked,
    prompt_instruction: 'You are a helpful assistant.',
    text_frontend: true,
    seed: Number(document.getElementById('tts-seed').value) || 0,
    export_video: document.getElementById('export-video').checked,
  };
  try {
    const job = await jsonResponse(await fetch(`/api/dubbing/jobs/${analysisJobId}/generate`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    }));
    markStep(4);
    setStatus(document.getElementById('batch-status'), `批量任务 ${job.generation_id || ''} 已开始，逐句生成中…`);
    await pollJob('generation');
  } catch (error) {
    setStatus(document.getElementById('batch-status'), error.message || String(error), 'error');
  } finally {
    button.disabled = false;
  }
}

startInput.addEventListener('input', clampSelection);
endInput.addEventListener('input', clampSelection);
mediaInput.addEventListener('change', () => updateMedia(mediaInput.files[0]));
referenceFileInput.addEventListener('change', () => {
  referenceSourceFile = referenceFileInput.files[0] || null;
  if (!referenceSourceFile) return;
  voiceFileName.value = `${referenceSourceFile.name}（独立参考文件）`;
  referenceStartInput.value = 0;
  if (ttsMode.value === 'zero_shot') fillTranscript();
});
['dragenter', 'dragover'].forEach(event => drop.addEventListener(event, e => { e.preventDefault(); drop.classList.add('drag'); }));
['dragleave', 'drop'].forEach(event => drop.addEventListener(event, e => { e.preventDefault(); drop.classList.remove('drag'); }));
drop.addEventListener('drop', e => updateMedia(e.dataTransfer.files[0]));
analyzeButton.addEventListener('click', startAnalysis);
sentenceBatchButton.addEventListener('click', startAnalysis);
fullBatchButton.addEventListener('click', startGeneration);
document.getElementById('batch-btn').addEventListener('click', startGeneration);
document.getElementById('transcript-btn').addEventListener('click', fillTranscript);
document.getElementById('attach-reference-btn').addEventListener('click', attachReference);
document.getElementById('select-all-btn').addEventListener('click', () => {
  if (!mediaDuration) return;
  startInput.value = 0;
  endInput.value = mediaDuration.toFixed(1).replace(/\.0$/, '');
  clampSelection();
  setStatus(timelineStatus, '已选择整段媒体；点击“分析整段媒体”会拆分完整时间轴。', 'warn');
});
document.getElementById('use-media-btn').addEventListener('click', () => {
  referenceSourceFile = null;
  referenceFileInput.value = '';
  if (sourceFile) {
    voiceFileName.value = `${sourceFile.name}（按选中片段截取）`;
    fillTranscript();
  }
});
document.getElementById('separate-btn').addEventListener('click', () => setStatus(
  document.getElementById('separate-status'),
  '首版时间轴和配音流程已可用；环境声 / BGM 分离模型暂未启用，批量导出会替换原音轨。',
  'warn',
));
document.getElementById('segments').addEventListener('click', event => {
  const row = event.target.closest('.segment');
  if (!row) return;
  const index = Number(row.dataset.index);
  if (event.target.closest('.segment-edit')) return;
  updateSelectedSegment(index);
});
document.getElementById('segments').addEventListener('input', event => {
  const input = event.target.closest('.segment-edit');
  if (!input) return;
  const item = segments[Number(input.dataset.index)];
  if (item) item.dubbing_text = input.value;
});
ttsMode.addEventListener('change', () => {
  referenceText.disabled = false;
  if (ttsMode.value === 'zero_shot') fillTranscript();
  else transcriptStatus.textContent = '当前模式不需要参考台词；仍可保留已识别文本作记录。';
});
document.getElementById('preview-btn').addEventListener('click', generatePreview);

drawWaveform();
clampSelection();
