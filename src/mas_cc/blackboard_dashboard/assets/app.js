(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = { timeline: null, snapshot: null, staticMode: false, staticBundle: null, busy: false };
  const embedded = $('dashboard-data').textContent.trim();
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const json = value => JSON.stringify(value ?? null, null, 2);
  const kv = (name, value) => `<div class="kv"><span>${esc(name)}</span><span>${esc(Array.isArray(value) ? value.join(', ') : value)}</span></div>`;

  async function get(path) {
    const response = await fetch(path, {cache: 'no-store'});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    return payload;
  }

  function setStatus(run) {
    const node = document.querySelector('.status');
    node.className = `status ${run.status || ''}`;
    $('status-text').textContent = `${run.status || 'waiting'} · ${run.completed_updates}/${run.expected_updates ?? '?'} updates`;
    $('refreshed').textContent = `refreshed ${new Date().toLocaleTimeString()}`;
  }

  function populateTimeline(timeline) {
    state.timeline = timeline;
    const round = $('round');
    const previousRound = round.value;
    round.replaceChildren(...timeline.rounds.map(item => new Option(`Round ${item.round_index + 1}`, item.round_index)));
    if (timeline.rounds.some(item => String(item.round_index) === previousRound)) round.value = previousRound;
    const agent = $('agent');
    const previousAgent = agent.value;
    agent.replaceChildren(...timeline.agents.map(id => new Option(id.replace('agent_', 'Agent '), id)));
    if (timeline.agents.includes(previousAgent)) agent.value = previousAgent;
    updateStepRange();
  }

  function updateStepRange() {
    if (!state.timeline) return;
    const selected = state.timeline.rounds.find(item => String(item.round_index) === $('round').value) || state.timeline.rounds.at(-1);
    if (!selected) return;
    $('step').max = selected.available_steps;
    if (+$('step').value > selected.available_steps || $('follow').checked) $('step').value = selected.available_steps;
    $('step-value').value = $('step').value;
  }

  function renderCards(snapshot) {
    const p = snapshot.population;
    const cards = [
      ['Round / update', `${snapshot.cursor.round_index + 1} / ${snapshot.cursor.step}`],
      ['Truth share', p.truth_vote_share == null ? '—' : p.truth_vote_share.toFixed(3)],
      ['Live messages', p.blackboard_live_size],
      ['Mean active', Number(p.mean_active).toFixed(2)],
      ['Mean historical', Number(p.mean_historical).toFixed(2)],
      ['Acquisitions', p.exact_acquisitions],
      ['Refreshes', p.refreshes],
      ['Prompt attempts', snapshot.run.prompt_attempts]
    ];
    $('cards').innerHTML = cards.map(([name,value]) => `<div class="card"><span>${esc(name)}</span><strong>${esc(value)}</strong></div>`).join('');
  }

  function renderOverview(snapshot) {
    renderCards(snapshot);
    const counts = snapshot.population.vote_counts;
    const total = Object.values(counts).reduce((a,b) => a+b, 0) || 1;
    $('vote-bars').innerHTML = Object.entries(counts).map(([vote,count]) => `<div class="bar-row ${vote === snapshot.population.correct_answer ? 'truth' : ''}"><span>${esc(vote)}${vote === snapshot.population.correct_answer ? ' ✓' : ''}</span><div class="bar"><i style="width:${100*count/total}%"></i></div><b>${count}</b></div>`).join('');
    const focal = snapshot.agent;
    $('activity').innerHTML = kv('Selected agent', focal.agent_id) + kv('Focal now', focal.is_focal_at_cursor ? 'yes' : 'no') + kv('Current vote', focal.vote) + kv('Latest action', focal.parsed_response?.public_message?.type || '—') + kv('Controller', snapshot.controller.action || '—') + kv('Invalid attempts', snapshot.run.invalid_attempts);
  }

  function renderMessages(snapshot) {
    const enabled = new Set([...document.querySelectorAll('.message-filter:checked')].map(node => node.value));
    const showExpired = $('expired').checked;
    const messages = snapshot.blackboard.filter(message => enabled.has(message.message_type) && (showExpired || message.live));
    $('messages').innerHTML = messages.length ? messages.map(message => `<article class="message ${esc(message.message_type)} ${message.live ? '' : 'expired'} ${message.new_at_cursor ? 'new' : ''}" data-agent="${esc(message.author_id)}"><div class="message-head"><span class="badge">${esc(message.message_type)}</span><strong>${esc(message.author_id)}</strong><span class="meta">${esc(message.message_id)} · round ${Number(message.round_created)+1} · expires ${Number(message.expires_after_round)+1}</span></div><p>${esc(message.text)}</p><div class="meta">evidence: ${esc(message.shared_fact_id || 'none')} · reply: ${esc(message.reply_to || 'none')} · ${message.live ? 'LIVE' : 'EXPIRED'}</div></article>`).join('') : '<div class="panel">No matching messages at this cursor.</div>';
    document.querySelectorAll('.message[data-agent]').forEach(node => node.addEventListener('click', () => selectAgent(node.dataset.agent)));
  }

  function renderCoverage(snapshot) {
    const mode = $('memory-mode').value;
    const latents = snapshot.coverage.latents;
    const rows = snapshot.coverage.agents.map(agent => {
      const present = new Set(agent[`${mode}_latent_ids`]);
      return `<tr><td data-agent="${esc(agent.agent_id)}">${esc(agent.agent_id)} · ${esc(agent.vote)}</td>${latents.map(latent => `<td class="cell ${present.has(latent) ? `on ${mode}` : ''}" title="${esc(agent.agent_id)} / ${esc(latent)}">${present.has(latent) ? '●' : ''}</td>`).join('')}</tr>`;
    }).join('');
    $('matrix').innerHTML = `<thead><tr><th>Agent / vote</th>${latents.map(value => `<th>${esc(value)}</th>`).join('')}</tr></thead><tbody>${rows}</tbody>`;
    document.querySelectorAll('#matrix [data-agent]').forEach(node => node.addEventListener('click', () => selectAgent(node.dataset.agent)));
  }

  function renderAgent(snapshot) {
    const a = snapshot.agent;
    $('agent-title').textContent = `${a.agent_id || 'Agent'} decision context`;
    $('agent-state').innerHTML = kv('Vote', a.vote) + kv('Focal at cursor', a.is_focal_at_cursor ? 'yes' : 'no') + kv('Latest decision update', a.latest_decision_global_update == null ? 'initialization' : Number(a.latest_decision_global_update)+1) + kv('Attempt', a.attempt || '—') + kv('Valid', a.valid == null ? '—' : a.valid) + kv('Active evidence', a.active_fact_ids || []) + kv('Historical evidence', a.historical_fact_ids || []);
    $('agent-timeline').innerHTML = (a.timeline || []).map(item => `<div class="timeline-item"><b>R${Number(item.round_index)+1}.${item.step}</b> ${esc(item.vote_before)} → ${esc(item.vote_after)} · ${esc(item.message_type || 'NONE')}</div>`).join('') || 'No social update for this agent yet.';
    $('visible-state').textContent = json(a.visible_state);
    $('compiled-prompt').innerHTML = (a.compiled_messages || []).map(message => `<div class="prompt-message"><div class="prompt-role">${esc(message.role)}</div><pre>${esc(message.content)}</pre></div>`).join('') || 'No recorded prompt yet.';
    $('raw-response').textContent = a.raw_response || 'No recorded response yet.';
    $('parsed-response').textContent = json(a.parsed_response);
  }

  function renderController(snapshot) {
    const c = snapshot.controller;
    $('controller-state').innerHTML = kv('Enabled', c.enabled) + kv('Action', c.action || '—') + kv('Target', c.target || '—') + kv('Intervention probability', c.probability ?? '—') + kv('Sampled action', c.sampled_action ?? '—') + kv('Controlled positions', c.controlled_positions || []) + kv('Directive IDs', c.directive_ids || []) + kv('Direct replies', c.direct_replies ?? '—') + kv('Unique readers', c.unique_readers ?? '—') + '<h2 style="margin-top:20px">Sensor observation</h2><pre>' + esc(json(c.sensor)) + '</pre>';
  }

  function render(snapshot) {
    state.snapshot = snapshot;
    setStatus(snapshot.run);
    renderOverview(snapshot); renderMessages(snapshot); renderCoverage(snapshot); renderAgent(snapshot); renderController(snapshot);
  }

  async function refresh(forceEdge = false) {
    if (state.busy) return;
    state.busy = true;
    try {
      if (state.staticMode) {
        const key = `${$('round').value}:${$('step').value}`;
        const base = state.staticBundle.snapshots[key];
        if (!base) throw new Error('No exported snapshot exists at this cursor');
        render({...base, agent: state.staticBundle.agents[key][$('agent').value]});
        return;
      }
      const timeline = await get('/api/timeline');
      populateTimeline(timeline);
      if ($('follow').checked || forceEdge) {
        const edge = timeline.available_cursors.at(-1);
        if (edge) { $('round').value = edge.round_index; updateStepRange(); $('step').value = edge.step; $('step-value').value = edge.step; }
      }
      const query = new URLSearchParams({round: $('round').value, step: $('step').value, agent: $('agent').value});
      render(await get(`/api/snapshot?${query}`));
    } catch (error) {
      $('status-text').textContent = `error · ${error.message}`;
    } finally { state.busy = false; }
  }

  function selectAgent(agent) {
    if (!agent) return;
    $('agent').value = agent;
    document.querySelector('[data-view="agent-view"]').click();
    refresh();
  }

  document.querySelectorAll('#tabs button').forEach(button => button.addEventListener('click', () => {
    document.querySelectorAll('#tabs button,.view').forEach(node => node.classList.remove('active'));
    button.classList.add('active'); $(button.dataset.view).classList.add('active');
  }));
  $('round').addEventListener('change', () => { $('follow').checked = false; updateStepRange(); refresh(); });
  $('step').addEventListener('input', () => { $('follow').checked = false; $('step-value').value = $('step').value; refresh(); });
  $('agent').addEventListener('change', () => refresh());
  $('follow').addEventListener('change', () => refresh(true));
  $('expired').addEventListener('change', () => renderMessages(state.snapshot));
  $('memory-mode').addEventListener('change', () => renderCoverage(state.snapshot));
  document.querySelectorAll('.message-filter').forEach(node => node.addEventListener('change', () => renderMessages(state.snapshot)));
  document.addEventListener('keydown', event => {
    if (!state.timeline || ['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)) return;
    if (event.key === ' ') { event.preventDefault(); $('follow').checked = !$('follow').checked; refresh(true); return; }
    if (!['ArrowLeft','ArrowRight'].includes(event.key)) return;
    event.preventDefault(); $('follow').checked = false;
    const delta = event.key === 'ArrowRight' ? 1 : -1;
    if (event.shiftKey) { $('round').selectedIndex = Math.max(0, Math.min($('round').options.length-1, $('round').selectedIndex + delta)); updateStepRange(); }
    else $('step').value = Math.max(1, Math.min(+$('step').max, +$('step').value + delta));
    $('step-value').value = $('step').value; refresh();
  });

  if (embedded) {
    state.staticMode = true; state.staticBundle = JSON.parse(embedded);
    populateTimeline(state.staticBundle.timeline);
    const edge = state.staticBundle.timeline.available_cursors.at(-1);
    $('round').value = edge.round_index; updateStepRange(); $('step').value = edge.step; $('step-value').value = edge.step;
    $('follow').checked = false;
    refresh();
  } else {
    refresh(true); setInterval(() => { if ($('follow').checked) refresh(true); }, 2000);
  }
})();
