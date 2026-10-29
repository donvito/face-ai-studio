const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const state = { samples: null, gallery: [], analyze: null, identify: null, liveness: null, verify: { a: null, b: null, active: 'a', res: null }, video: {} };

function toast(msg) { const t = $('#toast'); t.textContent = msg; t.style.display = 'block'; clearTimeout(t._h); t._h = setTimeout(() => t.style.display = 'none', 5000); }
async function api(url, data, method = 'POST') {
  const opt = { method };
  if (data) { const fd = new FormData(); for (const [k, v] of Object.entries(data)) if (v !== undefined && v !== null) fd.append(k, v); opt.body = fd; }
  const r = await fetch(url, method === 'GET' ? undefined : opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || r.statusText);
  return j;
}
const src = x => x instanceof File ? { file: x } : { sample: x };
const fmt = (v, d = 2) => v == null ? '-' : Number(v).toFixed(d);
const titleCase = s => s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

// Tabs
$$('#nav button').forEach(b => b.onclick = () => {
  $$('#nav button').forEach(x => x.classList.toggle('active', x === b));
  $$('.tab').forEach(t => t.classList.toggle('active', t.id === 'tab-' + b.dataset.tab));
  location.hash = b.dataset.tab;
});

// Sample picker
function buildPicker(el, onPick) {
  const def = el.dataset.default || 'all';
  el.innerHTML = `<div class="picker-tabs">${['all', 'people', 'groups'].map(g => `<button data-g="${g}" class="${g === def ? 'on' : ''}">${g === 'all' ? 'All samples' : g === 'people' ? 'Individuals' : 'Group photos'}</button>`).join('')}
    <label class="upload"><button type="button">Upload photo</button><input type="file" accept="image/*" hidden></label></div><div class="strip"></div>`;
  const strip = $('.strip', el);
  const render = g => {
    strip.innerHTML = state.samples.photos.filter(p => g === 'all' || p.group === g).map(p => `<img src="${p.thumb}" data-path="${p.path}" title="${p.title} (${p.license})">`).join('');
    $$('img', strip).forEach(img => img.onclick = () => { $$('img', strip).forEach(i => i.classList.remove('sel')); img.classList.add('sel'); onPick(img.dataset.path); });
  };
  $$('.picker-tabs button[data-g]', el).forEach(b => b.onclick = () => { $$('.picker-tabs button[data-g]', el).forEach(x => x.classList.toggle('on', x === b)); render(b.dataset.g); });
  const input = $('input[type=file]', el);
  $('.upload button', el).onclick = () => input.click();
  input.onchange = () => input.files[0] && onPick(input.files[0]);
  el._pick = onPick;
  render(def);
}

// Canvas drawing
function loadImg(url) { return new Promise(r => { const i = new Image(); i.onload = () => r(i); i.src = url; }); }
async function drawFaces(canvas, res, opts) {
  const img = res._img || (res._img = await loadImg(res.image));
  canvas.width = img.width; canvas.height = img.height;
  const c = canvas.getContext('2d'); c.drawImage(img, 0, 0);
  const s = Math.max(1, img.width / 900);
  res.faces.forEach((f, i) => {
    const [x1, y1, x2, y2] = f.bbox; const st = opts.style(f, i);
    if (opts.layers.box !== false) { c.lineWidth = 2.5 * s; c.strokeStyle = st.color; c.strokeRect(x1, y1, x2 - x1, y2 - y1); }
    if (opts.layers.lmk106 && f.landmark_2d_106) { c.fillStyle = '#2ee6a6'; f.landmark_2d_106.forEach(([x, y]) => { c.beginPath(); c.arc(x, y, 1.3 * s, 0, 7); c.fill(); }); }
    if (opts.layers.lmk68 && f.landmark_3d_68) { c.fillStyle = '#ff8a3d'; f.landmark_3d_68.forEach(([x, y]) => { c.beginPath(); c.arc(x, y, 1.5 * s, 0, 7); c.fill(); }); }
    if (opts.layers.kps && f.kps) { f.kps.forEach(([x, y], k) => { c.fillStyle = k === 2 ? '#ffd23f' : '#ff4d6d'; c.beginPath(); c.arc(x, y, 3 * s, 0, 7); c.fill(); }); }
    if (opts.layers.label !== false && st.label) {
      c.font = `600 ${13 * s}px -apple-system,sans-serif`; const w = c.measureText(st.label).width + 10 * s, h = 19 * s;
      const ty = y1 - h - 2 * s < 0 ? y2 + 2 * s : y1 - h - 2 * s;
      c.fillStyle = st.color; c.fillRect(x1 - 1, ty, w, h); c.fillStyle = '#0b0d12'; c.fillText(st.label, x1 + 5 * s, ty + 14 * s);
    }
  });
}
async function runOn(stageSel, fn) { const st = $(stageSel); st.classList.add('busy'); try { return await fn(); } catch (e) { toast(e.message); } finally { st.classList.remove('busy'); } }

// 01 Analyze
const anLayers = () => Object.fromEntries($$('#tab-analyze [data-layer]').map(i => [i.dataset.layer, i.checked]));
function renderAnalyze() {
  const r = state.analyze; if (!r) return;
  $('#tab-analyze .stage').classList.add('has');
  drawFaces($('#an-canvas'), r, { layers: anLayers(), style: f => ({ color: '#5b8cff', label: f.age != null ? `#${f.index + 1} ${f.sex} ${f.age}` : `#${f.index + 1} ${fmt(f.det_score)}` }) });
  $('#an-stat').textContent = `${r.faces.length} faces | ${r.elapsed_ms} ms | ${r.model} | ${r.provider.replace('ExecutionProvider', '')} | ${r.width}x${r.height}`;
  $('#an-cards').innerHTML = r.faces.map(f => {
    const [p, y, ro] = f.pose || [];
    const m = f.matches && f.matches[0];
    const emb = f.embedding_preview ? `<div class="emb">${f.embedding_preview.map(v => `<i style="height:${Math.min(100, Math.abs(v) * 900)}%;background:${v < 0 ? '#ff5470' : '#5b8cff'}"></i>`).join('')}</div>` : '';
    return `<div class="card"><img src="${f.crop}"><div><h3>Face #${f.index + 1}${m && m.score >= 0.3 ? `<span class="badge ok">${m.name} ${fmt(m.score)}</span>` : ''}</h3>
      <div class="kv"><span>detection</span><b>${fmt(f.det_score, 3)}</b>
      ${f.age != null ? `<span>age / gender</span><b>${f.age} / ${f.sex === 'M' ? 'male' : 'female'}</b>` : ''}
      ${f.pose ? `<span>yaw / pitch / roll</span><b>${fmt(y, 0)}° / ${fmt(p, 0)}° / ${fmt(ro, 0)}°</b>` : ''}
      ${f.liveness ? `<span>liveness</span><b>${f.liveness.status === 'ok' ? `${f.liveness.is_live ? 'live' : 'spoof'} ${fmt(f.liveness.live_score, 3)}` : 'input rejected'}</b>` : ''}
      ${f.embedding_dim ? `<span>embedding</span><b>${f.embedding_dim}-d, norm ${f.embedding_norm}</b>` : ''}
      <span>landmarks</span><b>5${f.landmark_2d_106 ? ' + 106' : ''}${f.landmark_3d_68 ? ' + 68 (3D)' : ''}</b></div></div>${emb}</div>`;
  }).join('') || '<div class="empty">No faces found</div>';
}
async function analyze(x) {
  state.analyze = await runOn('#tab-analyze .stage', () => api('/api/analyze', { ...src(x), model: $('#an-model').value }));
  state.analyzeSrc = x; renderAnalyze();
}
$$('#tab-analyze [data-layer]').forEach(i => i.onchange = renderAnalyze);
$('#an-model').onchange = () => state.analyzeSrc && analyze(state.analyzeSrc);

// 02 Verify
const PAIRS = {
  same: ['photos/people/christina_koch/christina_m_hammock_portrait.jpg', 'photos/people/christina_koch/jsc2023e0016435_alt.jpg'],
  hard: ['photos/people/victor_glover/victor_j_glover_official_portrait.jpg', 'photos/people/victor_glover/victor_glover_after_crew_1_splashdown.jpg'],
  diff: ['photos/people/reid_wiseman/jsc2023e0016434_alt.jpg', 'photos/people/jeremy_hansen/jsc2023e0016436_alt2.jpg'],
};
function setSlot(k, x) {
  state.verify[k] = x; const el = $(`.slot[data-slot=${k}]`); el.classList.add('has');
  $('img', el).src = x instanceof File ? URL.createObjectURL(x) : '/' + x; $('.crop', el)?.remove();
}
function activeSlot(k) { state.verify.active = k; $$('.slot').forEach(s => s.classList.toggle('active', s.dataset.slot === k)); }
$$('.slot').forEach(s => s.onclick = () => activeSlot(s.dataset.slot));
function renderVerdict() {
  const r = state.verify.res, th = +$('#vf-th').value, v = $('#vf-verdict');
  $('#vf-thv').textContent = th.toFixed(2); $('.gauge-mark', v).style.left = th * 100 + '%';
  if (!r) return;
  const same = r.similarity >= th;
  v.classList.toggle('same', same); v.classList.toggle('diff', !same);
  $('.vf-score', v).textContent = r.similarity.toFixed(3);
  $('.vf-label', v).textContent = same ? 'Same person' : 'Different people';
  const g = $('.gauge-fill', v); g.style.width = Math.max(0, r.similarity) * 100 + '%'; g.style.background = same ? 'var(--acc2)' : 'var(--bad)';
}
async function verify() {
  const { a, b } = state.verify; if (!a || !b) return;
  $('.vf-label').textContent = 'comparing...';
  try {
    const d = {}; if (a instanceof File) d.file_a = a; else d.sample_a = a; if (b instanceof File) d.file_b = b; else d.sample_b = b;
    const r = state.verify.res = await api('/api/compare', d);
    ['a', 'b'].forEach(k => { const el = $(`.slot[data-slot=${k}]`); $('.crop', el)?.remove(); el.insertAdjacentHTML('beforeend', `<img class="crop" src="${r[k].crop}" title="aligned 112x112 crop">`); });
    renderVerdict();
  } catch (e) { toast(e.message); }
}
function pickVerify(x) { const k = state.verify.active; setSlot(k, x); activeSlot(k === 'a' ? 'b' : 'a'); verify(); }
$$('[data-pair]').forEach(b => b.onclick = () => { const [a, c] = PAIRS[b.dataset.pair]; setSlot('a', a); setSlot('b', c); activeSlot('a'); verify(); });
$('#vf-th').oninput = renderVerdict;

// 03 Identify
async function loadGallery() {
  state.gallery = await (await fetch('/api/gallery')).json();
  $('#id-gallery').innerHTML = state.gallery.map(g => `<div class="person"><span class="x" data-del="${g.id}" title="remove">x</span><div class="faces">${g.crops.map(c => `<img src="${c}">`).join('')}</div><div class="nm">${g.name}</div><div class="ct">${g.count} reference photos</div></div>`).join('');
  $$('[data-del]').forEach(x => x.onclick = async () => { if (!confirm(`Remove ${x.closest('.person').querySelector('.nm').textContent} from the gallery? Their photos move to work/deleted/.`)) return; await fetch('/api/gallery/' + x.dataset.del, { method: 'DELETE' }); await loadGallery(); if (state.identifySrc) identify(state.identifySrc); });
  const chips = id => state.gallery.map(g => `<label><input type="checkbox" value="${g.id}" checked><img src="${g.crops[0]}">${g.name}</label>`).join('');
  $('#pf-people').innerHTML = chips(); $('#pa-people').innerHTML = chips();
  $$('#pf-people input').forEach(i => i.checked = false);
  syncPfPeople();
}
$('#enroll').onsubmit = async e => {
  e.preventDefault(); const f = e.target; const fd = new FormData(f);
  try { const r = await fetch('/api/gallery/enroll', { method: 'POST', body: fd }); const j = await r.json(); if (!r.ok) throw new Error(j.detail); f.reset(); toast(`Saved ${j.name}: ${j.added} photo(s) added${j.skipped ? `, ${j.skipped} skipped (no face)` : ''}`); await loadGallery(); if (state.identifySrc) identify(state.identifySrc); }
  catch (err) { toast(err.message); }
};
const COLORS = ['#ff8c42', '#2ee6a6', '#e65ac8', '#50befa', '#ffd23f', '#a78bfa'];
function renderIdentify() {
  const r = state.identify; if (!r) return; const th = +$('#id-th').value; $('#id-thv').textContent = th.toFixed(2);
  $('#tab-identify .stage').classList.add('has');
  const ids = state.gallery.map(g => g.id);
  const who = f => { const m = f.matches[0]; return m && m.score >= th ? m : null; };
  drawFaces($('#id-canvas'), r, { layers: { box: true, label: true }, style: f => { const m = who(f); return m ? { color: COLORS[ids.indexOf(m.id) % COLORS.length], label: `${m.name} ${fmt(m.score)}` } : { color: '#8a93a6', label: 'unknown' }; } });
  const found = r.faces.filter(who).length;
  $('#id-stat').textContent = `${r.faces.length} faces, ${found} identified | ${r.elapsed_ms} ms`;
  $('#id-cards').innerHTML = r.faces.map(f => { const m = who(f); return `<div class="card ${m ? 'hl' : ''}"><img src="${f.crop}"><div><h3>${m ? m.name : 'Unknown'}<span class="badge ${m ? 'ok' : ''}">${m ? 'match' : 'no match'}</span></h3>
    ${f.matches.map(x => `<div class="kv"><span>${x.name}</span><b>${fmt(x.score, 3)}</b></div><div class="bar"><i style="width:${Math.max(0, x.score) * 100}%;background:${x.score >= th ? 'var(--acc2)' : '#4a5367'}"></i><u style="left:${th * 100}%"></u></div>`).join('')}</div></div>`; }).join('');
}
async function identify(x) { state.identifySrc = x; state.identify = await runOn('#tab-identify .stage', () => api('/api/analyze', { ...src(x), model: 'buffalo_l' })); renderIdentify(); }
$('#id-th').oninput = renderIdentify;

// 04 Liveness
function renderLiveness() {
  const r = state.liveness; if (!r) return; const th = +$('#lv-th').value; $('#lv-thv').textContent = th.toFixed(2);
  $('#tab-liveness .stage').classList.add('has');
  const cls = f => !f.liveness ? null : f.liveness.status !== 'ok' ? 'rejected' : f.liveness.live_score >= th ? 'live' : 'spoof';
  drawFaces($('#lv-canvas'), r, { layers: { box: true, label: true }, style: f => { const c = cls(f); return { color: c === 'live' ? '#2ee6a6' : c === 'spoof' ? '#ff5470' : '#ffb020', label: c === 'rejected' ? 'input rejected' : `${c} ${fmt(f.liveness?.live_score, 3)}` }; } });
  const n = r.faces.map(cls);
  $('#lv-stat').textContent = `${n.filter(c => c === 'live').length} live, ${n.filter(c => c === 'spoof').length} spoof, ${n.filter(c => c === 'rejected').length} rejected | ${r.elapsed_ms} ms`;
  $('#lv-cards').innerHTML = r.faces.map(f => { const c = cls(f); const s = f.liveness?.live_score; return `<div class="card"><img src="${f.crop}"><div><h3>Face #${f.index + 1}<span class="badge ${c === 'live' ? 'ok' : c === 'spoof' ? 'bad' : 'warn'}">${c}</span></h3>
    ${s != null ? `<div class="kv"><span>live score</span><b>${fmt(s, 4)}</b></div><div class="bar"><i style="width:${s * 100}%;background:${c === 'live' ? 'var(--acc2)' : 'var(--bad)'}"></i><u style="left:${th * 100}%"></u></div>
    <div class="kv" style="margin-top:6px"><span>normal mode</span><b>${c === 'live' ? 'recognition runs' : 'recognition skipped'}</b></div>` : `<div class="kv"><span>reason</span><b>${f.liveness?.reason || '-'}</b></div>`}</div></div>`; }).join('');
}
async function liveness(x) { state.livenessSrc = x; state.liveness = await runOn('#tab-liveness .stage', () => api('/api/analyze', { ...src(x), model: 'buffalo_l', spoof: $('#lv-spoof').checked })); renderLiveness(); }
$('#lv-th').oninput = renderLiveness;
$('#lv-spoof').onchange = () => state.livenessSrc && liveness(state.livenessSrc);

// 05 Cluster
let clusterData = null;
async function cluster() {
  const btn = $('#cl-run'); btn.disabled = true; btn.textContent = 'Embedding every face...';
  try {
    const th = $('#cl-th').value; const r = clusterData = await api('/api/cluster', { threshold: th });
    $('#cl-stat').textContent = `${r.faces} faces -> ${r.clusters.length} clusters (${r.clusters.filter(c => c.size > 1).length} with 2+ faces) | embed ${r.embed_ms} ms, total ${r.cluster_ms} ms`;
    $('#cl-out').innerHTML = r.clusters.map((c, i) => `<div class="cluster"><h3>${c.guess || (c.size > 1 ? 'Person ' + (i + 1) : 'Single face')}<span class="badge">${c.size} faces in ${c.photos} photos</span>${c.guess ? '<span class="badge ok">matches gallery</span>' : ''}</h3>
      <div class="faces">${c.faces.map(f => `<img src="${f.crop}" title="${f.photo}">`).join('')}</div></div>`).join('');
  } catch (e) { toast(e.message); } finally { btn.disabled = false; btn.textContent = 'Cluster the album'; }
}
$('#cl-run').onclick = cluster;
$('#cl-th').oninput = () => $('#cl-thv').textContent = (+$('#cl-th').value).toFixed(2);
$('#cl-th').onchange = () => clusterData && cluster();

// Videos
function buildVideos(el, key) {
  el.innerHTML = state.samples.videos.map((v, i) => `<div class="vid ${i === 0 ? 'on' : ''}" data-path="${v.path}"><img src="${v.poster}"><div>${titleCase(v.name)}</div></div>`).join('') +
    `<label class="vid upload"><span>Upload video</span><input type="file" accept="video/*" hidden></label>`;
  state.video[key] = state.samples.videos[0]?.path;
  $$('.vid[data-path]', el).forEach(v => v.onclick = () => { $$('.vid', el).forEach(x => x.classList.toggle('on', x === v)); state.video[key] = v.dataset.path; });
  const inp = $('input', el); inp.onchange = () => { if (inp.files[0]) { $$('.vid', el).forEach(x => x.classList.remove('on')); inp.parentElement.classList.add('on'); $('span', inp.parentElement).textContent = inp.files[0].name; state.video[key] = inp.files[0]; } };
}
async function pollJob(job, prog, onDone) {
  const bar = $(prog); bar.classList.add('on');
  while (true) {
    const j = await (await fetch('/api/jobs/' + job.id)).json();
    $('div', bar).style.width = (j.progress * 100).toFixed(1) + '%';
    $('span', bar).textContent = `${j.stage} ${(j.progress * 100).toFixed(0)}% | ${((Date.now() / 1000) - j.started).toFixed(0)}s`;
    if (j.status === 'done') { $('span', bar).textContent = `done in ${j.elapsed_s}s`; return onDone(j); }
    if (j.status === 'error') { $('span', bar).textContent = 'failed'; throw new Error(j.error); }
    await new Promise(r => setTimeout(r, 700));
  }
}
const segVal = id => $(`#${id} .on`).dataset.v;
$$('.seg').forEach(s => $$('button', s).forEach(b => b.onclick = () => { $$('button', s).forEach(x => x.classList.toggle('on', x === b)); syncPfPeople(); }));
function syncPfPeople() { $('#pf-people').classList.toggle('disabled', segVal('pf-mode') === 'all'); }
const checked = id => $$(`#${id} input:checked`).map(i => i.value).join(',');
const vsrc = v => v instanceof File ? { file: v } : { sample: v };

$('#pf-run').onclick = async () => {
  if (segVal('pf-mode') !== 'all' && !checked('pf-people')) return toast('Select at least one person for this mode');
  const btn = $('#pf-run'); btn.disabled = true; $('#pf-out').innerHTML = '';
  try {
    const job = await api('/api/video/privateframe', { ...vsrc(state.video.privacy), mode: segVal('pf-mode'), method: segVal('pf-method'), people: checked('pf-people') });
    await pollJob(job, '#pf-prog', j => {
      const r = j.result, s = r.summary || {};
      const modeTxt = { all: 'every face blurred', blur_only: 'blur only: ' + r.people.join(', '), exempt: 'kept visible: ' + r.people.join(', ') }[r.mode];
      $('#pf-out').innerHTML = `<div class="summary">${[['face tracks', s.face_tracks], ['face regions', s.face_regions], ['redacted', s.redacted_face_regions], ['kept visible', s.kept_face_regions], ['matched tracks', s.matched_tracks], ['frames', s.frame_count], ['time', j.elapsed_s + 's']].filter(x => x[1] != null).map(([k, v]) => `<div class="metric"><b>${v}</b><span>${k}</span></div>`).join('')}</div>
        <figure><video src="${r.source}" controls muted loop></video><figcaption>Original</figcaption></figure>
        <figure><video src="${r.video}" controls muted loop></video><figcaption>PrivateFrame (${r.method}, ${modeTxt}) | <a href="${r.json}" target="_blank">editable analysis JSON</a></figcaption></figure>`;
      const vids = $$('#pf-out video'); vids.forEach(v => v.play()); vids[1].onplay = () => { vids[0].currentTime = vids[1].currentTime; };
    });
  } catch (e) { toast(e.message); } finally { btn.disabled = false; }
};

$('#pa-run').onclick = async () => {
  if (!checked('pa-people')) return toast('Select at least one person to track');
  const btn = $('#pa-run'); btn.disabled = true; $('#pa-out').innerHTML = '';
  try {
    const job = await api('/api/video/person', { ...vsrc(state.video.person), people: checked('pa-people'), analysis_fps: $('#pa-fps').value });
    await pollJob(job, '#pa-prog', j => {
      const r = j.result; const ids = state.gallery.map(g => g.id);
      const rows = r.appearances.map(a => `<div class="tl-row"><b>${a.name}</b><div class="tl-track">${a.times.map(t => `<i style="left:${t / r.duration * 100}%;background:${COLORS[ids.indexOf(a.id) % COLORS.length]}"></i>`).join('')}</div><span>${a.seconds}s</span></div>`).join('');
      $('#pa-out').innerHTML = `<div class="pa-grid"><figure><video src="${r.video}" controls autoplay muted loop></video><figcaption>Green-ish boxes: matched people (face or body ReID). Orange: unknown.</figcaption></figure>
        <div class="timeline"><h3 style="margin:0 0 6px">Who appears when (${r.duration}s clip)</h3>${rows || '<div class="empty">nobody matched</div>'}
        <div class="summary" style="margin-top:14px">${[['face matches', r.stats.face], ['body (ReID) matches', r.stats.body], ['unmatched', r.stats.unmatched], ['frames analyzed', r.analyzed_frames], ['ms / frame', r.avg_ms_per_frame]].map(([k, v]) => `<div class="metric"><b>${v}</b><span>${k}</span></div>`).join('')}</div>
        <h3 style="margin:14px 0 6px;font-size:13px">Registered references</h3>${r.registrations.map(x => `<div class="kv"><span>${x.name}</span><b>${x.accepted} accepted${x.rejected.length ? ', ' + x.rejected.length + ' rejected (' + x.rejected.join(', ') + ')' : ''}</b></div>`).join('')}</div></div>`;
    });
  } catch (e) { toast(e.message); } finally { btn.disabled = false; }
};

// Drag and drop onto the active tab
let dragDepth = 0;
addEventListener('dragenter', e => { e.preventDefault(); dragDepth++; document.body.classList.add('dragging'); });
addEventListener('dragleave', () => { if (--dragDepth <= 0) document.body.classList.remove('dragging'); });
addEventListener('dragover', e => e.preventDefault());
addEventListener('drop', e => {
  e.preventDefault(); dragDepth = 0; document.body.classList.remove('dragging');
  const f = e.dataTransfer.files[0]; if (!f || !f.type.startsWith('image/')) return;
  const p = $('.tab.active .picker'); if (p && p._pick) p._pick(f);
});

// Boot
(async () => {
  state.samples = await (await fetch('/api/samples')).json();
  buildPicker($('[data-picker=analyze]'), analyze);
  buildPicker($('[data-picker=verify]'), pickVerify);
  buildPicker($('[data-picker=identify]'), identify);
  buildPicker($('[data-picker=liveness]'), liveness);
  buildVideos($('[data-videos=privacy]'), 'privacy');
  buildVideos($('[data-videos=person]'), 'person');
  await loadGallery();
  const info = await (await fetch('/api/info')).json();
  $('#runtime').innerHTML = `InsightFace <b>${info.insightface}</b><br>Runtime <b>${info.provider.replace('ExecutionProvider', '')}</b> (Apple Silicon)<br>${Object.entries(info.models).map(([k, v]) => `${k}: ${v}`).join('<br>')}`;
  const tab = location.hash.slice(1); if (tab) $(`#nav [data-tab=${tab}]`)?.click();
  const first = 'photos/groups/artemis_2_crew_portrait.jpg';
  analyze(first); identify(first);
})();
