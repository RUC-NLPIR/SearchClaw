/**
 * SearchClaw — WebSocket client
 *
 * Conversation UI: inline tool blocks interleaved with streamed text,
 * natural scroll, sources at the end of the response.
 */

// ── State ──
let ws = null;
let isResearching = false;
let citationCount = 0;
let citations = [];
let currentAssistantEl = null;
let currentProseEl = null;
let userWantsAutoScroll = true;
let pendingNewChat = false;
let currentSessionId = null;
let authKey = sessionStorage.getItem('authKey') || '';

// ── DOM ──
const conversation = document.getElementById('conversation');
const welcomeScreen = document.getElementById('welcomeScreen');
const messagesEl = document.getElementById('messages');
const queryInput = document.getElementById('queryInput');
const submitBtn = document.getElementById('submitBtn');
const statusIndicator = document.getElementById('statusIndicator');
const statusLabel = document.getElementById('statusLabel');
const sidebarHistory = document.getElementById('sidebarHistory');

// ── Auth helpers ──
function authHeaders() {
    return authKey ? { 'Authorization': `Bearer ${authKey}` } : {};
}

// ── WebSocket ──
function connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const authParam = authKey ? `?api_key=${encodeURIComponent(authKey)}` : '';
    ws = new WebSocket(`${proto}//${location.host}/ws/search${authParam}`);

    ws.onopen = () => {
        statusIndicator.classList.add('connected');
        statusLabel.textContent = 'Connected';
        loadSessions();
    };
    ws.onclose = () => {
        statusIndicator.classList.remove('connected');
        statusLabel.textContent = 'Disconnected';
        isResearching = false;
        refreshBtn();
        setTimeout(connect, 3000);
    };
    ws.onerror = () => { statusLabel.textContent = 'Error'; };
    ws.onmessage = (ev) => {
        try { handleEvent(JSON.parse(ev.data)); }
        catch (e) { console.error('Parse error:', e); }
    };
}

// ── Events ──
function handleEvent(ev) {
    const h = {
        text_delta: onText,
        tool_use: onToolUse,
        tool_result: onToolResult,
        citation: onCitation,
        plan_update: onPlanUpdate,
        status: onStatus,
        error: onError,
        done: onDone,
        user_question: onUserQuestion,
    };
    (h[ev.type] || (() => {}))(ev.data);
}

function onText(d) {
    ensureAssistant();
    ensureProse();
    removeCursor();

    currentProseEl._raw = (currentProseEl._raw || '') + (d.text || '');
    currentProseEl.innerHTML = mdLive(currentProseEl._raw);
    addCursor();
    autoScroll();
}

function onToolUse(d) {
    ensureAssistant();
    currentProseEl = null;
    removeCursor();

    const el = document.createElement('div');
    el.className = 'tool-block';
    el.id = `tool-${d.tool_use_id}`;

    const cls = iconClass(d.tool_name);
    const emoji = iconEmoji(d.tool_name);
    const preview = toolPreview(d);

    el.innerHTML = `
        <div class="tool-block-header" onclick="toggleTool(this)">
            <span class="tool-icon ${cls}">${emoji}</span>
            <span class="tool-name">${esc(fmtName(d.tool_name))}</span>
            <span class="tool-query">${esc(preview)}</span>
            <div class="tool-spinner"></div>
            <span class="tool-chevron">&#9654;</span>
        </div>
        <div class="tool-block-body"></div>
    `;
    currentAssistantEl.appendChild(el);
    autoScroll();
}

function onToolResult(d) {
    const el = document.getElementById(`tool-${d.tool_use_id}`);
    if (!el) return;
    const sp = el.querySelector('.tool-spinner');
    if (sp) {
        sp.outerHTML = d.is_error
            ? '<span class="tool-error-mark">&#10007;</span>'
            : '<span class="tool-check">&#10003;</span>';
    }
    const body = el.querySelector('.tool-block-body');
    if (body && d.result) body.textContent = d.result;
}

function onCitation(d) {
    citationCount++;
    citations.push({ ...d, num: citationCount });
}

function onPlanUpdate(d) {
    ensureAssistant();
    currentProseEl = null;
    removeCursor();

    // Update in place — remove previous plan checklist in this assistant block
    const existing = currentAssistantEl.querySelector('.plan-checklist');
    if (existing) existing.remove();

    const tasks = d.tasks || [];
    if (tasks.length === 0) return;

    const el = document.createElement('div');
    el.className = 'plan-checklist';

    const completedCount = d.completed_count || 0;
    const totalCount = d.total_count || tasks.length;
    const progressPct = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0;

    let html = `
        <div class="plan-header">
            <span class="plan-title">Research Plan</span>
            <span class="plan-progress">${completedCount}/${totalCount} completed</span>
        </div>
        <div class="plan-progress-bar">
            <div class="plan-progress-fill" style="width: ${progressPct}%"></div>
        </div>
        <ul class="plan-tasks">
    `;

    for (const task of tasks) {
        const statusCls = `plan-task-${task.status}`;
        const icon = { pending: '○', in_progress: '◉', completed: '●' }[task.status] || '○';
        html += `
            <li class="plan-task ${statusCls}">
                <span class="plan-task-icon">${icon}</span>
                <span class="plan-task-title">${esc(task.title)}</span>
                ${task.findings ? `<span class="plan-task-findings">${esc(task.findings.slice(0, 120))}</span>` : ''}
            </li>
        `;
    }

    html += '</ul>';
    el.innerHTML = html;
    currentAssistantEl.appendChild(el);
    autoScroll();
}

function onStatus(d) {
    const msg = d.message || '';
    if (msg.includes('Research started')) return;

    ensureAssistant();
    currentProseEl = null;
    removeCursor();

    const el = document.createElement('div');
    el.className = 'status-inline';
    el.innerHTML = `<span class="status-pulse"></span>${esc(msg)}`;
    currentAssistantEl.appendChild(el);
    autoScroll();
}

function onError(d) {
    ensureAssistant();
    currentProseEl = null;
    removeCursor();

    const el = document.createElement('div');
    el.className = 'error-block';
    el.innerHTML = `<span>&#9888;</span><span>${esc(d.message || 'An error occurred')}</span>`;
    currentAssistantEl.appendChild(el);
    isResearching = false;
    refreshBtn();
}

function onDone(d) {
    removeCursor();

    // Track session_id for multi-turn continuity across reconnects
    if (d.session_id) {
        currentSessionId = d.session_id;
    }

    // Final markdown render — apply to ALL prose blocks, not just the last one
    if (currentAssistantEl) {
        for (const proseEl of currentAssistantEl.querySelectorAll('.assistant-prose')) {
            if (proseEl._raw) {
                proseEl.innerHTML = mdFinal(proseEl._raw);
            }
        }
    }

    // Stats + copy button
    if (currentAssistantEl) {
        const citedCount = citations.filter(c => c.cited).length;
        const s = document.createElement('div');
        s.className = 'stats-bar';

        const statsText = document.createElement('span');
        statsText.textContent = `${d.turn_count || 0} turns`;
        s.appendChild(statsText);

        const sourcesText = document.createElement('span');
        sourcesText.textContent = `${citedCount} sources`;
        s.appendChild(sourcesText);

        // Copy button — copies only the final prose (raw markdown), not tool/status blocks
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-btn';
        copyBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg><span>Copy</span>';
        // Capture the assistant element for the closure
        const assistantEl = currentAssistantEl;
        copyBtn.onclick = () => copyResponseMarkdown(assistantEl, copyBtn);
        s.appendChild(copyBtn);

        currentAssistantEl.appendChild(s);
    }

    currentAssistantEl = null;
    currentProseEl = null;
    isResearching = false;
    refreshBtn();
    autoScroll();
    loadSessions();
}

function onUserQuestion(d) {
    ensureAssistant();
    currentProseEl = null;
    removeCursor();

    const el = document.createElement('div');
    el.className = 'user-question-block';

    let html = `<div class="uq-question">${esc(d.question)}</div>`;
    html += '<div class="uq-options">';
    for (const opt of (d.options || [])) {
        // Use data attribute for the label to avoid XSS in onclick
        html += '<button class="uq-option">';
        html += `<span class="uq-option-label">${esc(opt.label)}</span>`;
        if (opt.description) {
            html += `<span class="uq-option-desc">${esc(opt.description)}</span>`;
        }
        html += '</button>';
    }
    html += '</div>';
    el.innerHTML = html;

    // Attach click handlers after innerHTML is set (avoids inline onclick XSS issues)
    const buttons = el.querySelectorAll('.uq-option');
    const options = d.options || [];
    buttons.forEach((btn, i) => {
        btn.addEventListener('click', () => answerQuestion(btn, options[i].label));
    });

    currentAssistantEl.appendChild(el);
    autoScroll();
}

function answerQuestion(btn, answer) {
    // Disable all option buttons in this question block
    const block = btn.closest('.user-question-block');
    for (const b of block.querySelectorAll('.uq-option')) {
        b.disabled = true;
        b.classList.remove('selected');
    }
    btn.classList.add('selected');

    // Send answer to backend
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'user_answer', answer }));
    }
}

function copyResponseMarkdown(assistantEl, btn) {
    // Copy only the LAST prose block — the final synthesized answer.
    // Earlier prose blocks contain intermediate reasoning ("Let me search...",
    // "Now let me read...") which is not part of the final response.
    const proseEls = assistantEl.querySelectorAll('.assistant-prose');
    let markdown = '';
    for (let i = proseEls.length - 1; i >= 0; i--) {
        const raw = (proseEls[i]._raw || '').trim();
        if (raw) { markdown = raw; break; }
    }
    if (!markdown) return;

    navigator.clipboard.writeText(markdown).then(() => {
        btn.classList.add('copied');
        btn.querySelector('span').textContent = 'Copied';
        setTimeout(() => {
            btn.classList.remove('copied');
            btn.querySelector('span').textContent = 'Copy';
        }, 2000);
    }).catch(() => {
        // Fallback for older browsers / non-HTTPS
        const ta = document.createElement('textarea');
        ta.value = markdown;
        ta.style.cssText = 'position:fixed;left:-9999px';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        btn.classList.add('copied');
        btn.querySelector('span').textContent = 'Copied';
        setTimeout(() => {
            btn.classList.remove('copied');
            btn.querySelector('span').textContent = 'Copy';
        }, 2000);
    });
}

// ── Submission ──
function handleSubmit(ev) {
    ev.preventDefault();
    const q = queryInput.value.trim();
    if (!q || isResearching || !ws || ws.readyState !== WebSocket.OPEN) return;
    submitQuery(q);
}

function submitQuery(query) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    welcomeScreen.style.display = 'none';
    messagesEl.style.display = 'block';

    citationCount = 0;
    citations = [];
    currentAssistantEl = null;
    currentProseEl = null;

    const el = document.createElement('div');
    el.className = 'msg-user';
    el.innerHTML = `<div class="msg-user-bubble">${esc(query)}</div>`;
    messagesEl.appendChild(el);

    const options = {};
    if (pendingNewChat) {
        options.new_chat = true;
        pendingNewChat = false;
        currentSessionId = null;
    }
    if (currentSessionId) {
        options.session_id = currentSessionId;
    }
    ws.send(JSON.stringify({ query, options }));

    isResearching = true;
    queryInput.value = '';
    queryInput.style.height = 'auto';
    refreshBtn();
    userWantsAutoScroll = true;
    autoScroll();
}

function submitExample(btn) { submitQuery(btn.textContent.trim()); }

function startNewChat() {
    messagesEl.innerHTML = '';
    messagesEl.style.display = 'none';
    welcomeScreen.style.display = 'flex';
    currentAssistantEl = null;
    currentProseEl = null;
    citationCount = 0;
    citations = [];
    pendingNewChat = true;  // Signal backend to clear history on next query
    currentSessionId = null;
    // Deselect active sidebar item
    const active = sidebarHistory.querySelector('.history-item.active');
    if (active) active.classList.remove('active');
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('collapsed');
}

// ── Helpers ──
function ensureAssistant() {
    if (currentAssistantEl) return;
    currentAssistantEl = document.createElement('div');
    currentAssistantEl.className = 'msg-assistant';
    currentAssistantEl.innerHTML = `
        <div class="assistant-header">
            <div class="assistant-avatar">S</div>
            <span class="assistant-name">SearchClaw</span>
        </div>
    `;
    messagesEl.appendChild(currentAssistantEl);
}

function ensureProse() {
    if (currentProseEl) return;
    currentProseEl = document.createElement('div');
    currentProseEl.className = 'assistant-prose';
    currentProseEl._raw = '';
    currentAssistantEl.appendChild(currentProseEl);
}

function removeCursor() {
    const c = messagesEl?.querySelector('.streaming-cursor');
    if (c) c.remove();
}

function addCursor() {
    if (!currentProseEl) return;
    const c = document.createElement('span');
    c.className = 'streaming-cursor';
    currentProseEl.appendChild(c);
}

function refreshBtn() {
    submitBtn.disabled = isResearching || !queryInput.value.trim();
}

function autoScroll() {
    if (!userWantsAutoScroll) return;
    requestAnimationFrame(() => { conversation.scrollTop = conversation.scrollHeight; });
}

conversation.addEventListener('scroll', () => {
    const gap = conversation.scrollHeight - conversation.scrollTop - conversation.clientHeight;
    userWantsAutoScroll = gap < 80;
});

// ── Markdown ──
function mdLive(raw) {
    let h = esc(raw.replace(/^\n+/, ''));
    h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    h = h.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');
    h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
    h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, text, url) => safeLink(text, url));
    h = h.replace(/\n/g, '<br>');
    return h;
}

function mdFinal(raw) {
    // Process code blocks first (protect them from other transformations)
    const codeBlocks = [];
    let h = raw.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        const idx = codeBlocks.length;
        codeBlocks.push(`<pre><code>${esc(code.trim())}</code></pre>`);
        return `\x00CB${idx}\x00`;
    });

    // Escape HTML in remaining text
    h = esc(h);

    // Restore code blocks
    h = h.replace(/\x00CB(\d+)\x00/g, (_, i) => codeBlocks[i]);

    // Horizontal rules
    h = h.replace(/^---+$/gm, '<hr>');

    // Headings
    h = h.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    h = h.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    h = h.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Bold & italic (use [\s\S] style to handle CJK & multiline)
    h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    h = h.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');

    // Inline code
    h = h.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Links — sanitize href to prevent javascript: URIs
    h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, text, url) => safeLink(text, url));

    // Blockquotes
    h = h.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

    // Unordered lists
    h = h.replace(/^- (.+)$/gm, '<li>$1</li>');

    // Ordered lists
    h = h.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Wrap consecutive <li> in <ul>
    h = h.replace(/((?:<li>[\s\S]*?<\/li>\s*)+)/g, '<ul>$1</ul>');

    // Simple markdown tables
    h = h.replace(/((?:^\|.+\|$\n?)+)/gm, (tableBlock) => {
        const rows = tableBlock.trim().split('\n').filter(r => r.trim());
        if (rows.length < 2) return tableBlock;
        let html = '<table>';
        for (let ri = 0; ri < rows.length; ri++) {
            const row = rows[ri].trim();
            // Skip separator row (|---|---|)
            if (/^\|[\s\-:]+\|$/.test(row) || /^\|(\s*-+\s*\|)+$/.test(row)) continue;
            const cells = row.split('|').filter((_, i, a) => i > 0 && i < a.length - 1);
            const tag = ri === 0 ? 'th' : 'td';
            html += '<tr>' + cells.map(c => `<${tag}>${c.trim()}</${tag}>`).join('') + '</tr>';
        }
        html += '</table>';
        return html;
    });

    // Paragraphs: double newlines become paragraph breaks
    h = h.replace(/\n{2,}/g, '</p><p>');
    h = '<p>' + h + '</p>';
    h = h.replace(/<p>\s*<\/p>/g, '');

    // Single newlines to <br> (but not inside tags)
    h = h.replace(/(?<!\>)\n(?!\<)/g, '<br>');

    // Clean up: remove <br> right after block elements
    h = h.replace(/(<\/h[1-6]>)<br>/g, '$1');
    h = h.replace(/(<\/li>)<br>/g, '$1');
    h = h.replace(/(<\/blockquote>)<br>/g, '$1');
    h = h.replace(/(<\/pre>)<br>/g, '$1');
    h = h.replace(/<br>(<h[1-6]>)/g, '$1');
    h = h.replace(/<br>(<ul>)/g, '$1');
    h = h.replace(/<br>(<\/ul>)/g, '$1');
    h = h.replace(/<p>(<h[1-6]>)/g, '$1');
    h = h.replace(/(<\/h[1-6]>)<\/p>/g, '$1');
    h = h.replace(/<p>(<ul>)/g, '$1');
    h = h.replace(/(<\/ul>)<\/p>/g, '$1');
    h = h.replace(/<p>(<pre>)/g, '$1');
    h = h.replace(/(<\/pre>)<\/p>/g, '$1');
    h = h.replace(/<p>(<hr>)<\/p>/g, '$1');
    h = h.replace(/<p>(<table>)/g, '$1');
    h = h.replace(/(<\/table>)<\/p>/g, '$1');
    h = h.replace(/<br>(<hr>)/g, '$1');
    h = h.replace(/(<hr>)<br>/g, '$1');
    h = h.replace(/<br>(<table>)/g, '$1');
    h = h.replace(/(<\/table>)<br>/g, '$1');

    return h;
}

function esc(t) {
    const d = document.createElement('div');
    d.textContent = t;
    return d.innerHTML;
}

/**
 * Create a safe <a> tag, blocking javascript:, data:, and vbscript: URIs.
 * Only http:// and https:// links are rendered as clickable.
 */
function safeLink(text, url) {
    // Decode HTML entities that esc() may have introduced (e.g., &amp; -> &)
    const tmp = document.createElement('textarea');
    tmp.innerHTML = url;
    const decoded = tmp.value.trim();

    // Only allow http and https schemes
    if (/^https?:\/\//i.test(decoded)) {
        return `<a href="${url}" target="_blank" rel="noopener noreferrer">${text}</a>`;
    }
    // Block javascript:, data:, vbscript:, and anything else
    return `<span class="blocked-link" title="Link blocked for security">${text}</span>`;
}

function iconClass(n) {
    return { web_search:'search', web_fetch:'fetch', deep_read:'read',
             cite_source:'cite', academic_search:'academic', news_search:'news',
             research_plan:'plan', ask_user:'ask' }[n] || 'search';
}

function iconEmoji(n) {
    return { web_search:'&#128269;', web_fetch:'&#127760;', deep_read:'&#128214;',
             cite_source:'&#128220;', academic_search:'&#127891;', news_search:'&#128240;',
             research_plan:'&#128203;', ask_user:'&#10067;' }[n] || '&#9881;';
}

function fmtName(n) { return n.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()); }

function toolPreview(d) {
    const i = d.tool_input || {};
    if (i.query) return `"${i.query}"`;
    if (i.question) return `"${i.question}"`;
    if (i.url) { try { const u = new URL(i.url); return u.hostname + u.pathname.slice(0,30); } catch { return i.url.slice(0,50); } }
    if (i.cached_path) return 'reading cached page';
    if (i.title) return i.title;
    return JSON.stringify(i).slice(0,60);
}

function toggleTool(hdr) { hdr.parentElement.classList.toggle('expanded'); }

// ── Input ──
queryInput.addEventListener('input', () => {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 160) + 'px';
    refreshBtn();
});

queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); handleSubmit(e); }
});

// ── Session History ──
function loadSessions() {
    fetch('/api/sessions', { headers: authHeaders() })
        .then(r => r.json())
        .then(data => {
            const sessions = data.sessions || [];
            sidebarHistory.innerHTML = '';

            if (sessions.length === 0) {
                sidebarHistory.innerHTML = '<div class="sidebar-history-empty">No past sessions</div>';
                return;
            }

            for (const s of sessions) {
                const el = document.createElement('div');
                el.className = 'history-item';
                el.dataset.sessionId = s.session_id;

                const queryText = s.query || 'Untitled';
                const meta = [];
                if (s.timestamp) meta.push(timeAgo(s.timestamp));
                if (s.num_citations > 0) meta.push(`${s.num_citations} sources`);

                el.innerHTML = `
                    <div class="history-item-query">${esc(queryText)}</div>
                    ${meta.length ? `<div class="history-item-meta">${esc(meta.join(' · '))}</div>` : ''}
                    <button class="history-delete-btn" title="Delete session">&#10005;</button>
                `;
                el.addEventListener('click', () => loadSession(s.session_id));
                el.querySelector('.history-delete-btn').addEventListener('click', (e) => {
                    e.stopPropagation();
                    deleteSession(s.session_id, el);
                });
                sidebarHistory.appendChild(el);
            }
        })
        .catch(e => console.warn('Failed to load sessions:', e));
}

function deleteSession(sessionId, el) {
    fetch(`/api/sessions/${sessionId}`, {
        method: 'DELETE',
        headers: authHeaders(),
    })
        .then(r => {
            if (r.ok) {
                el.remove();
                // If we just deleted the currently viewed session, go to welcome
                if (currentSessionId === sessionId) {
                    currentSessionId = null;
                    startNewChat();
                }
                // Show empty state if no sessions left
                if (!sidebarHistory.querySelector('.history-item')) {
                    sidebarHistory.innerHTML = '<div class="sidebar-history-empty">No past sessions</div>';
                }
            }
        })
        .catch(e => console.warn('Failed to delete session:', e));
}

function loadSession(sessionId) {
    fetch(`/api/sessions/${sessionId}`, { headers: authHeaders() })
        .then(r => {
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            return r.json();
        })
        .then(data => {
            // Highlight active item
            for (const el of sidebarHistory.querySelectorAll('.history-item')) {
                el.classList.toggle('active', el.dataset.sessionId === sessionId);
            }

            // Clear current view
            welcomeScreen.style.display = 'none';
            messagesEl.style.display = 'block';
            messagesEl.innerHTML = '';
            currentAssistantEl = null;
            currentProseEl = null;
            citationCount = 0;
            citations = [];

            // Render all turns (backward-compatible with old single-turn sessions)
            const turns = data.turns || [{ query: data.query, final_answer: data.final_answer }];
            let lastAssistEl = null;

            for (const turn of turns) {
                // Render user query
                if (turn.query) {
                    const userEl = document.createElement('div');
                    userEl.className = 'msg-user';
                    userEl.innerHTML = `<div class="msg-user-bubble">${esc(turn.query)}</div>`;
                    messagesEl.appendChild(userEl);
                }

                // Render assistant answer
                if (turn.final_answer) {
                    const assistEl = document.createElement('div');
                    assistEl.className = 'msg-assistant';
                    assistEl.innerHTML = `
                        <div class="assistant-header">
                            <div class="assistant-avatar">S</div>
                            <span class="assistant-name">SearchClaw</span>
                        </div>
                    `;

                    const proseEl = document.createElement('div');
                    proseEl.className = 'assistant-prose';
                    proseEl._raw = turn.final_answer;
                    proseEl.innerHTML = mdFinal(turn.final_answer);
                    assistEl.appendChild(proseEl);

                    messagesEl.appendChild(assistEl);
                    lastAssistEl = assistEl;
                }
            }

            // Stats bar on the last assistant block
            if (lastAssistEl) {
                const numCitations = data.num_citations || 0;
                const turnCount = data.turn_count || 0;
                const statsEl = document.createElement('div');
                statsEl.className = 'stats-bar';
                statsEl.innerHTML = `
                    <span>${turnCount} turns</span>
                    <span>${numCitations} sources</span>
                `;
                lastAssistEl.appendChild(statsEl);
            }

            // Continue this session if user sends a follow-up query
            currentSessionId = sessionId;
            pendingNewChat = false;
            conversation.scrollTop = 0;
        })
        .catch(e => console.warn('Failed to load session:', e));
}

function timeAgo(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now - date;
    const diffMin = Math.floor(diffMs / 60000);
    const diffHr = Math.floor(diffMs / 3600000);
    const diffDay = Math.floor(diffMs / 86400000);

    if (diffMin < 1) return 'just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    if (diffHr < 24) return `${diffHr}h ago`;
    if (diffDay < 7) return `${diffDay}d ago`;
    return date.toLocaleDateString();
}

// ── Auth ──
function checkAuth() {
    const appContainer = document.getElementById('appContainer');
    const loginOverlay = document.getElementById('loginOverlay');

    // Probe whether auth is required by hitting a protected endpoint
    fetch('/api/sessions', { headers: authHeaders() })
        .then(r => {
            if (r.ok) {
                // Authenticated (or no auth required) — show app
                loginOverlay.style.display = 'none';
                appContainer.style.display = 'flex';
                connect();
            } else {
                // Auth required — clear stale key and show login
                sessionStorage.removeItem('authKey');
                authKey = '';
                loginOverlay.style.display = 'flex';
                appContainer.style.display = 'none';
                const pwInput = document.getElementById('loginPassword');
                if (pwInput) pwInput.focus();
            }
        })
        .catch(() => {
            // Network error — show app anyway, WebSocket will handle reconnect
            loginOverlay.style.display = 'none';
            appContainer.style.display = 'flex';
            connect();
        });
}

function doLogin(e) {
    if (e) e.preventDefault();
    const pw = document.getElementById('loginPassword').value;
    const errorEl = document.getElementById('loginError');
    errorEl.textContent = '';

    fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: pw }),
    })
        .then(r => {
            if (r.ok) {
                authKey = pw;
                sessionStorage.setItem('authKey', pw);
                document.getElementById('loginOverlay').style.display = 'none';
                document.getElementById('appContainer').style.display = 'flex';
                connect();
            } else {
                errorEl.textContent = 'Wrong password';
                document.getElementById('loginPassword').select();
            }
        })
        .catch(() => {
            errorEl.textContent = 'Connection error';
        });
}

// ── Init ──
checkAuth();
refreshBtn();
