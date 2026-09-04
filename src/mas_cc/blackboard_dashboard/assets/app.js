(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = { timeline: null, snapshot: null, staticMode: false, staticBundle: null, busy: false, mode: 'episode', study: null, cell: null, cellId: null, episodeId: null, cellTab: 'cell-episodes', selectedTrajectories: new Set(), episodeCache: new Map() };
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

  const unavailable = value => value == null || value === '' ? 'Unavailable' : value;
  const statusBadge = value => `<span class="status-badge ${esc(value)}">${esc(value)}</span>`;
  function sparkline(points, key = 'truth_share', width = 150, height = 42) {
    const values = points.map((point, index) => [index, point[key]]).filter(item => item[1] != null);
    if (!values.length) return '<span class="unavailable">Unavailable</span>';
    const left = 30, right = 8, top = 8, bottom = 18;
    const plotWidth = width - left - right, plotHeight = height - top - bottom;
    const denominator = Math.max(1, points.length - 1);
    const x = index => left + (index / denominator) * plotWidth;
    const y = value => top + (1 - Number(value)) * plotHeight;
    const coordinates = values.map(([index, value]) => `${x(index)},${y(value)}`).join(' ');
    const grid = [1, 0.5, 0].map(value => `<line class="plot-grid${value === 0 ? ' zero' : ''}" x1="${left}" y1="${y(value)}" x2="${width-right}" y2="${y(value)}"></line><text class="axis-label" x="${left-5}" y="${y(value)+3}" text-anchor="end">${value.toFixed(value === 0.5 ? 1 : 0)}</text>`).join('');
    const dots = values.map(([index, value]) => { const point = points[index], update = point.global_update_index == null ? index + 1 : Number(point.global_update_index) + 1; return `<circle class="plot-point" cx="${x(index)}" cy="${y(value)}" r="2.5"><title>Update ${update}: ${Number(value).toFixed(3)}</title></circle>`; }).join('');
    return `<svg class="sparkline" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(key)} trajectory">${grid}<polyline points="${coordinates}"></polyline>${dots}<text class="axis-label" x="${left}" y="${height-3}">1</text><text class="axis-label" x="${width-right}" y="${height-3}" text-anchor="end">${points.length}</text></svg>`;
  }

  function showShell(name) {
    state.mode = name;
    $('study-shell').hidden = name === 'episode';
    $('episode-shell').hidden = name !== 'episode';
    $('study-view').hidden = name !== 'study';
    $('cell-view').hidden = name !== 'cell';
    $('breadcrumbs').hidden = state.staticMode || !state.study;
  }

  function renderBreadcrumbs(cell = null, episode = null) {
    $('breadcrumbs').innerHTML = `<button data-crumb="study">${esc(state.study?.study_id || 'Study')}</button>${cell ? `<span>›</span><button data-crumb="cell">${esc(cell.config_name)} / ${esc(cell.cell_id)}</button>` : ''}${episode ? `<span>›</span><span>${esc(episode)}</span>` : ''}`;
    $('breadcrumbs').querySelector('[data-crumb="study"]')?.addEventListener('click', () => { state.cell = null; state.cellId = null; state.episodeId = null; history.replaceState({}, '', location.pathname); renderStudy(); });
    $('breadcrumbs').querySelector('[data-crumb="cell"]')?.addEventListener('click', () => openCell(cell.qualified_id));
  }

  function filterOptions(id, values) {
    const select = $(id), previous = select.value;
    select.replaceChildren(new Option('All', ''), ...[...new Set(values.filter(value => value != null))].sort().map(value => new Option(value, value)));
    select.value = [...select.options].some(option => option.value === previous) ? previous : '';
  }

  function renderStudy() {
    showShell('study'); renderBreadcrumbs();
    const study = state.study;
    document.querySelector('h1').textContent = study.study_id;
    const totals = study.episode_outcomes, activity = study.episode_activity;
    $('status-text').textContent = `${totals.completed} durable episodes complete · ${activity.running + activity.advancing} running · ${study.active_scheduler_tasks} SLURM cell tasks active`;
    document.querySelector('.status').className = `status ${study.live ? 'running' : 'completed'}`;
    $('study-refreshed').textContent = `refreshed ${new Date(study.refreshed_at).toLocaleTimeString()}`;
    const cards = [
      ['Cells', `${study.discovered_cell_count}/${study.expected_cell_count}`], ['Episodes expected', study.expected_episode_count],
      ['Not started', activity.not_started], ['Running', activity.running + activity.advancing], ['Durable complete', totals.completed], ['Failed', totals.failed + totals.aborted],
      ['Unknown', totals.unknown], ['SLURM active', study.scheduler.available ? study.active_scheduler_tasks : 'Unavailable']
    ];
    $('study-cards').innerHTML = cards.map(([name,value]) => `<div class="card"><span>${esc(name)}</span><strong>${esc(value)}</strong></div>`).join('');
    filterOptions('filter-block', study.cells.map(cell => cell.parameters.experiment_block));
    filterOptions('filter-controller', study.cells.map(cell => cell.parameters.controller_condition));
    filterOptions('filter-rho', study.cells.map(cell => cell.parameters.rho).sort((a,b) => Number(a)-Number(b)));
    filterOptions('filter-status', study.cells.map(cell => cell.status));
    renderCellTable();
  }

  function renderCellTable() {
    let cells = state.study.cells.filter(cell =>
      (!$('filter-block').value || String(cell.parameters.experiment_block) === $('filter-block').value) &&
      (!$('filter-controller').value || String(cell.parameters.controller_condition) === $('filter-controller').value) &&
      (!$('filter-rho').value || Number(cell.parameters.rho) === Number($('filter-rho').value)) &&
      (!$('filter-status').value || cell.status === $('filter-status').value) &&
      true
    );
    const sort = $('cell-sort').value;
    cells.sort((a,b) => sort === 'status' ? a.status.localeCompare(b.status) : sort === 'progress' ? b.outcome_counts.completed - a.outcome_counts.completed : a.qualified_id.localeCompare(b.qualified_id));
    $('cell-table').innerHTML = `<thead><tr><th>Config / cell</th><th>Condition</th><th>ρ</th><th>b</th><th>Scientific outcomes</th><th>Live activity</th><th>SLURM task</th><th></th></tr></thead><tbody>${cells.map(cell => `<tr><td><b>${esc(cell.config_name)}</b><br><span class="meta">${esc(cell.cell_id)}</span></td><td>${esc(unavailable(cell.parameters.controller_condition))}</td><td>${esc(unavailable(cell.parameters.rho))}</td><td>${esc(unavailable(cell.parameters.b))}</td><td>${cell.outcome_counts.completed}/${cell.expected_episodes} durable complete<br><span class="meta">${cell.outcome_counts.failed} failed · ${cell.outcome_counts.aborted} aborted · ${cell.outcome_counts.unknown} unknown</span></td><td>${cell.activity_counts.running + cell.activity_counts.advancing} running<br><span class="meta">${cell.activity_counts.started_unchanged} inactive stream</span></td><td>${cell.scheduler ? `${statusBadge(cell.scheduler.state)}<br><span class="meta">array ${cell.scheduler.array_index} · ${esc(unavailable(cell.scheduler.node))} · ${esc(unavailable(cell.scheduler.elapsed))}</span>` : '<span class="unavailable">Unavailable</span>'}</td><td><button class="open-cell" data-cell="${esc(cell.qualified_id)}">Open</button></td></tr>`).join('')}</tbody>`;
    document.querySelectorAll('.open-cell').forEach(button => button.addEventListener('click', () => openCell(button.dataset.cell)));
  }

  function setCellTab(id) {
    state.cellTab = id;
    document.querySelectorAll('#cell-tabs button').forEach(button => button.classList.toggle('active', button.dataset.cellView === id));
    document.querySelectorAll('.cell-subview').forEach(view => { view.hidden = view.id !== id; });
    updateHash();
  }

  function updateHash() {
    if (!state.study) return;
    const params = new URLSearchParams();
    if (state.cellId) params.set('cell', state.cellId);
    if (state.episodeId) params.set('episode', state.episodeId);
    if (state.cellTab) params.set('cellTab', state.cellTab);
    if ($('all-parameters').open) params.set('parameters', 'open');
    if (state.selectedTrajectories.size) params.set('trajectories', [...state.selectedTrajectories].join(','));
    const activeTab = document.querySelector('#tabs button.active')?.dataset.view;
    if (state.episodeId && activeTab) params.set('episodeTab', activeTab);
    if (state.episodeId) { params.set('round', $('round').value); params.set('step', $('step').value); params.set('agent', $('agent').value); params.set('follow', $('follow').checked ? '1' : '0'); }
    if ($('filter-rho').value) params.set('rho', $('filter-rho').value);
    history.replaceState({}, '', `#${params}`);
  }

  function renderCell(cell) {
    showShell('cell'); renderBreadcrumbs(cell);
    const c = cell.outcome_counts, a = cell.activity_counts;
    $('cell-cards').innerHTML = [['Durable complete', c.completed], ['Failed / aborted', c.failed + c.aborted], ['Incomplete / unknown', c.incomplete + c.unknown], ['Running', a.running + a.advancing], ['Inactive stream', a.started_unchanged], ['Not started', a.not_started]].map(([name,value]) => `<div class="card"><span>${esc(name)}</span><strong>${esc(value)}</strong></div>`).join('');
    const primary = [['Controller condition', cell.parameters.controller_condition], ['ρ', cell.parameters.rho], ['b', cell.parameters.b], ['Task', cell.parameters.task_id], ['Population', cell.parameters.population_size], ['Rounds', cell.parameters['game.options.rounds']], ['Controller target', cell.parameters.controller_target], ['Truth', cell.parameters.ground_truth]];
    $('primary-parameters').innerHTML = primary.filter(([,value]) => value != null).map(([name,value]) => kv(name, value)).join('');
    $('cell-parameters').innerHTML = Object.entries(cell.parameters).sort(([a],[b]) => a.localeCompare(b)).map(([name,value]) => kv(name, unavailable(typeof value === 'object' ? json(value) : value))).join('');
    $('mean-label').textContent = `Descriptive live mean · ${cell.descriptive_mean.label}. Missing rounds are not interpolated.`;
    $('episode-table').innerHTML = `<thead><tr><th>Repetition</th><th>Episode</th><th>Seed</th><th>Durable outcome</th><th>Live activity</th><th>Progress</th><th>Last update / elapsed</th><th></th></tr></thead><tbody>${cell.episodes.map(episode => `<tr><td>${episode.repetition_index}</td><td>${esc(episode.episode_id)}</td><td>${esc(unavailable(episode.seed))}</td><td>${statusBadge(episode.durable_status)}${episode.status_reason ? `<br><span class="meta">${esc(episode.status_reason)}</span>` : ''}</td><td>${statusBadge(episode.activity_status)}</td><td>${episode.current_round == null ? 'Unavailable' : `round ${episode.current_round + 1}`}${episode.current_update == null ? '' : ` / update ${episode.current_update + 1}`}</td><td>${episode.last_update_at ? esc(new Date(episode.last_update_at).toLocaleTimeString()) : episode.elapsed_seconds == null ? 'Unavailable' : `${episode.elapsed_seconds.toFixed(1)} s`}</td><td>${episode.detail_available ? `<button class="open-episode" data-episode="${esc(episode.qualified_id)}">Inspect episode</button>` : `<span class="unavailable" title="${esc(episode.detail_reason)}">${esc(episode.detail_reason)}</span>`}</td></tr>`).join('')}</tbody>`;
    if (!state.selectedTrajectories.size) cell.episodes.forEach(episode => { if (cell.vote_series[episode.qualified_id]?.points.length) state.selectedTrajectories.add(episode.qualified_id); });
    $('trajectory-controls').innerHTML = cell.episodes.filter(episode => cell.vote_series[episode.qualified_id]?.points.length).map(episode => `<label><input type="checkbox" data-trajectory="${esc(episode.qualified_id)}" ${state.selectedTrajectories.has(episode.qualified_id) ? 'checked' : ''}> repetition ${episode.repetition_index} <button class="trajectory-inspect" data-episode="${esc(episode.qualified_id)}" ${episode.detail_available ? '' : 'disabled'}>Inspect</button></label>`).join('');
    renderTrajectories(); setCellTab(state.cellTab);
    document.querySelectorAll('.open-episode,.trajectory-inspect').forEach(button => button.addEventListener('click', () => openEpisode(button.dataset.episode)));
    document.querySelectorAll('[data-trajectory]').forEach(input => input.addEventListener('change', () => { input.checked ? state.selectedTrajectories.add(input.dataset.trajectory) : state.selectedTrajectories.delete(input.dataset.trajectory); renderTrajectories(); updateHash(); }));
  }

  function renderTrajectories() {
    const cell = state.cell;
    const individual = cell.episodes.filter(episode => state.selectedTrajectories.has(episode.qualified_id)).map(episode => `<div class="trajectory-row"><span>Repetition ${episode.repetition_index}</span>${sparkline(cell.vote_series[episode.qualified_id].points, 'truth_share', 520, 90)}${sparkline(cell.vote_series[episode.qualified_id].points, 'controller_target_share', 520, 90)}</div>`).join('');
    $('cell-votes').innerHTML = `<div class="plot-legend"><span class="truth-line">Truth</span><span class="target-line">Controller target</span></div><h3>Descriptive mean</h3>${sparkline(cell.mean_vote_series, 'truth_share', 520, 120)}${sparkline(cell.mean_vote_series, 'controller_target_share', 520, 120)}<h3>Selected repetitions</h3>${individual || '<span class="unavailable">No trajectories selected</span>'}`;
  }

  async function openCell(id, polling = false) {
    try {
      const disclosureOpen = $('all-parameters').open;
      state.cellId = id; state.cell = await get(`/api/study/cell/${encodeURIComponent(id)}`);
      if (!polling) state.episodeId = null;
      renderCell(state.cell); $('all-parameters').open = disclosureOpen;
      updateHash();
    } catch (error) { $('status-text').textContent = `error · ${error.message}`; }
  }

  async function openEpisode(id) {
    state.episodeId = id; showShell('episode'); renderBreadcrumbs(state.cell, id);
    $('episode-nav').hidden = false;
    updateHash();
    const cached = state.episodeCache.get(id);
    if (cached) {
      state.episodeCache.delete(id); state.episodeCache.set(id, cached);
      populateTimeline(cached.timeline); render(cached.snapshot);
      return;
    }
    $('status-text').textContent = 'Loading episode detail…';
    try {
      const detail = await get(`/api/study/episode/${encodeURIComponent(id)}/detail`);
      if (state.episodeId !== id) return;
      state.episodeCache.set(id, detail);
      while (state.episodeCache.size > 8) state.episodeCache.delete(state.episodeCache.keys().next().value);
      populateTimeline(detail.timeline); render(detail.snapshot);
    } catch (error) { $('status-text').textContent = `error · ${error.message}`; }
  }

  function adjacentEpisode(delta) {
    if (!state.cell || !state.episodeId) return;
    const available = state.cell.episodes.filter(episode => episode.detail_available);
    const index = available.findIndex(episode => episode.qualified_id === state.episodeId);
    const target = available[index + delta];
    if (target) openEpisode(target.qualified_id);
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
    renderEpisodeTrajectory(timeline);
  }

  function renderEpisodeTrajectory(timeline) {
    const points = timeline.time_series || [];
    const cursor = state.snapshot?.cursor?.global_update_index;
    const marker = cursor == null ? '' : `<p class="meta">Selected update: ${Number(cursor) + 1}</p>`;
    $('episode-votes').innerHTML = `<div class="plot-legend"><span class="truth-line">Truth share</span><span class="target-line">Controller-target share</span></div>${marker}<h3>Truth</h3>${sparkline(points, 'truth_share', 900, 180)}<h3>Controller target</h3>${sparkline(points, 'controller_target_share', 900, 180)}`;
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
    const promptReason = snapshot.capabilities?.prompts?.reason;
    const responseReason = snapshot.capabilities?.raw_response?.reason;
    $('agent-title').textContent = `${a.agent_id || 'Agent'} decision context`;
    $('agent-state').innerHTML = kv('Vote', a.vote) + kv('Focal at cursor', a.is_focal_at_cursor ? 'yes' : 'no') + kv('Latest decision update', a.latest_decision_global_update == null ? 'initialization' : Number(a.latest_decision_global_update)+1) + kv('Attempt', a.attempt || '—') + kv('Valid', a.valid == null ? '—' : a.valid) + kv('Active evidence', a.active_fact_ids || []) + kv('Historical evidence', a.historical_fact_ids || []);
    $('agent-timeline').innerHTML = (a.timeline || []).map(item => `<div class="timeline-item"><b>R${Number(item.round_index)+1}.${item.step}</b> ${esc(item.vote_before)} → ${esc(item.vote_after)} · ${esc(item.message_type || 'NONE')}</div>`).join('') || 'No social update for this agent yet.';
    $('visible-state').textContent = json(a.visible_state);
    $('compiled-prompt').innerHTML = promptReason ? `<span class="unavailable">${esc(promptReason)}</span>` : (a.compiled_messages || []).map(message => `<div class="prompt-message"><div class="prompt-role">${esc(message.role)}</div><pre>${esc(message.content)}</pre></div>`).join('') || 'No recorded prompt yet.';
    $('raw-response').textContent = responseReason || a.raw_response || 'No recorded response yet.';
    $('parsed-response').textContent = json(a.parsed_response);
  }

  function renderController(snapshot) {
    const c = snapshot.controller;
    $('controller-state').innerHTML = kv('Enabled', c.enabled) + kv('Action', c.action || '—') + kv('Target', c.target || '—') + kv('Intervention probability', c.probability ?? '—') + kv('Sampled action', c.sampled_action ?? '—') + kv('Controlled positions', c.controlled_positions || []) + kv('Directive IDs', c.directive_ids || []) + kv('Direct replies', c.direct_replies ?? '—') + kv('Unique readers', c.unique_readers ?? '—') + '<h2 style="margin-top:20px">Sensor observation</h2><pre>' + esc(json(c.sensor)) + '</pre>';
  }

  function render(snapshot) {
    state.snapshot = snapshot;
    setStatus(snapshot.run);
    if (!snapshot.cursor) {
      $('cards').innerHTML = '<div class="card"><span>Status</span><strong>Waiting for the first retained update</strong></div>';
      $('messages').innerHTML = '<div class="panel">No blackboard records are available yet.</div>';
      $('matrix').innerHTML = '';
      return;
    }
    renderOverview(snapshot); renderMessages(snapshot); renderCoverage(snapshot); renderAgent(snapshot); renderController(snapshot);
    if (state.timeline) renderEpisodeTrajectory(state.timeline);
  }

  async function refresh(forceEdge = false) {
    if (state.busy) return;
    state.busy = true;
    try {
      if (state.staticMode) {
        const key = `round:${$('round').value}:${$('step').value}`;
        const base = state.staticBundle.snapshots[key];
        if (!base) throw new Error('No exported snapshot exists at this cursor');
        render({...base, agent: state.staticBundle.agents[key][$('agent').value]});
        return;
      }
      const prefix = state.study && state.episodeId ? `/api/study/episode/${encodeURIComponent(state.episodeId)}` : '/api';
      const timeline = await get(`${prefix}/timeline`);
      populateTimeline(timeline);
      if ($('follow').checked || forceEdge) {
        const edge = timeline.available_cursors.at(-1);
        if (edge) { $('round').value = edge.round_index; updateStepRange(); $('step').value = edge.step; $('step-value').value = edge.step; }
      }
      const query = new URLSearchParams({round: $('round').value, step: $('step').value, agent: $('agent').value});
      render(await get(`${prefix}/snapshot?${query}`));
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
    updateHash();
  }));
  $('round').addEventListener('change', () => { $('follow').checked = false; updateStepRange(); updateHash(); refresh(); });
  $('step').addEventListener('input', () => { $('follow').checked = false; $('step-value').value = $('step').value; updateHash(); refresh(); });
  $('agent').addEventListener('change', () => { updateHash(); refresh(); });
  $('follow').addEventListener('change', () => { updateHash(); refresh(true); });
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

  async function startStudy() {
    try {
      state.study = await get('/api/study'); state.mode = 'study'; renderStudy();
      const restored = new URLSearchParams(location.hash.slice(1));
      if (restored.get('rho')) { $('filter-rho').value = restored.get('rho'); renderCellTable(); }
      if (restored.get('cellTab')) state.cellTab = restored.get('cellTab');
      if (restored.get('trajectories')) state.selectedTrajectories = new Set(restored.get('trajectories').split(','));
      if (restored.get('cell')) await openCell(restored.get('cell'));
      $('all-parameters').open = restored.get('parameters') === 'open';
      if (restored.get('episode')) {
        $('follow').checked = restored.get('follow') !== '0';
        await openEpisode(restored.get('episode'));
        if (!$('follow').checked) {
          $('round').value = restored.get('round') || $('round').value; updateStepRange();
          $('step').value = restored.get('step') || $('step').value; $('step-value').value = $('step').value;
          $('agent').value = restored.get('agent') || $('agent').value;
          await refresh(false);
        }
        document.querySelector(`[data-view="${restored.get('episodeTab') || 'overview'}"]`)?.click();
      }
      setInterval(async () => {
        if (state.episodeId) { if ($('follow').checked) refresh(true); return; }
        if (state.mode === 'cell' && state.cellId) { await openCell(state.cellId, true); return; }
        if (state.mode === 'study') { state.study = await get('/api/study'); renderStudy(); }
      }, 2000);
    } catch (error) { $('status-text').textContent = `error · ${error.message}`; }
  }
  ['filter-block','filter-controller','filter-rho','filter-status','cell-sort'].forEach(id => $(id).addEventListener('change', () => { renderCellTable(); updateHash(); }));
  document.querySelectorAll('#cell-tabs button').forEach(button => button.addEventListener('click', () => setCellTab(button.dataset.cellView)));
  $('all-parameters').addEventListener('toggle', updateHash);
  $('previous-episode').addEventListener('click', () => adjacentEpisode(-1));
  $('next-episode').addEventListener('click', () => adjacentEpisode(1));

  if (embedded) {
    state.staticMode = true; state.staticBundle = JSON.parse(embedded);
    populateTimeline(state.staticBundle.timeline);
    const edge = state.staticBundle.timeline.available_cursors.at(-1);
    if (edge) { $('round').value = edge.round_index; updateStepRange(); $('step').value = edge.step; $('step-value').value = edge.step; }
    $('follow').checked = false;
    refresh();
  } else {
    get('/api/study').then(() => startStudy()).catch(() => { refresh(true); setInterval(() => { if ($('follow').checked) refresh(true); }, 2000); });
  }
})();
