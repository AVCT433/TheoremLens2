// TheoremLens 프런트엔드. 서버가 준 텍스트만 렌더링하며, 모델 출력은 정제(DOMPurify) 후에만 DOM에 넣는다.
(function () {
  "use strict";

  var L = window.TLLib;
  var POLL_MS = 2000;
  var BUSY = { queued: 1, extracting: 1, indexing: 1 };
  var STATUS_KO = { queued: "대기 중", extracting: "텍스트 추출 중", indexing: "색인 중", ready: "준비됨", error: "오류" };

  var state = { books: [], selected: new Set(), conversationId: null, sending: false, pollTimer: null };
  var $ = function (id) {
    return document.getElementById(id);
  };

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  async function api(path, opts) {
    var res = await fetch(path, opts);
    if (!res.ok) {
      var detail = "";
      try {
        var j = await res.json();
        detail = typeof j.detail === "string" ? j.detail : "";
      } catch (e) {
        detail = "";
      }
      throw new Error(detail || "요청 실패 (" + res.status + ")");
    }
    return res.json();
  }

  // ---------- 마크다운 + 수식 + 인용 ----------

  var PH_RE = new RegExp(L.PH_OPEN + "M(\\d+)" + L.PH_CLOSE, "g");

  function renderMath(parent, item) {
    var span = el("span", item.display ? "math-display" : "math-inline");
    try {
      katex.render(item.tex, span, { displayMode: item.display, throwOnError: false, trust: false });
    } catch (e) {
      span.textContent = item.tex;
    }
    parent.appendChild(span);
  }

  function citeButton(sid, citations) {
    var btn = el("button", "cite", sid);
    btn.type = "button";
    var known = citations.some(function (c) {
      return c.sid === sid;
    });
    if (!known) btn.disabled = true;
    btn.addEventListener("click", function () {
      openSource(sid, citations);
    });
    return btn;
  }

  // 정제된 DOM의 텍스트 노드에서 수식 자리표시자와 [S#]를 실제 요소로 바꾼다.
  function hydrate(root, items, citations) {
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    var nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(function (node) {
      var text = node.nodeValue;
      var inCode = node.parentElement && node.parentElement.closest("code, pre");
      var frag = document.createDocumentFragment();
      var changed = false;
      text.split(PH_RE).forEach(function (piece, i) {
        if (i % 2 === 1) {
          var item = items[Number(piece)];
          if (item) renderMath(frag, item);
          changed = true;
          return;
        }
        if (inCode || !citations) {
          if (piece) frag.appendChild(document.createTextNode(piece));
          return;
        }
        L.splitCitations(piece).forEach(function (p) {
          if (p.type === "cite") {
            frag.appendChild(citeButton(p.sid, citations));
            changed = true;
          } else {
            frag.appendChild(document.createTextNode(p.value));
          }
        });
      });
      if (changed) node.parentNode.replaceChild(frag, node);
    });
  }

  function renderMarkdown(target, md, citations) {
    var m = L.extractMath(md);
    target.innerHTML = DOMPurify.sanitize(marked.parse(m.text), {
      FORBID_TAGS: ["style", "img", "a", "form", "input", "iframe"],
      FORBID_ATTR: ["style"],
    });
    hydrate(target, m.items, citations);
    target.querySelectorAll("h1, h2, h3, h4").forEach(function (h) {
      if (h.textContent.indexOf(L.OUTSIDE_HEADING) !== -1) {
        h.classList.add("outside-heading");
        var next = h.nextElementSibling;
        while (next && !/^H[1-4]$/.test(next.tagName)) {
          next.classList.add("outside-body");
          next = next.nextElementSibling;
        }
      }
    });
  }

  // ---------- 출처 패널 ----------

  function openSource(sid, citations) {
    var c = citations.find(function (x) {
      return x.sid === sid;
    });
    var body = $("source-body");
    body.textContent = "";
    if (!c) return;
    var pages = c.page_start === c.page_end ? "p." + c.page_start : "p." + c.page_start + "-" + c.page_end;
    body.appendChild(el("h2", "", "[" + c.sid + "] " + c.book_title));
    var meta = [c.display_label || c.label, [c.chapter, c.section].filter(Boolean).join(" > "), pages + " (PDF 쪽)"];
    body.appendChild(el("p", "muted", meta.filter(Boolean).join(" · ")));
    var text = el("div", "source-text");
    renderMarkdown(text, c.text, null);
    body.appendChild(text);
    L.pageRange(c, 4).forEach(function (n) {
      var img = el("img", "page-img");
      img.loading = "lazy";
      img.alt = c.book_title + " " + n + "쪽 원본";
      img.src = "/books/" + encodeURIComponent(c.book_id) + "/page/" + n;
      body.appendChild(img);
    });
    if (c.page_end - c.page_start + 1 > 4) body.appendChild(el("p", "muted", "긴 범위라 앞 4쪽만 표시합니다."));
    $("source-panel").hidden = false;
  }

  // ---------- 채팅 ----------

  function addMessage(role) {
    var box = el("div", "msg " + role);
    var empty = document.querySelector("#messages .empty");
    if (empty) empty.remove();
    $("messages").appendChild(box);
    $("messages").scrollTop = $("messages").scrollHeight;
    return box;
  }

  function renderClaims(box, resp) {
    if (!resp.claims.length) return;
    var details = el("details", "claims");
    var v = resp.verification || {};
    var okCount = resp.claims.filter(function (c) {
      return c.status === "verified";
    }).length;
    var summary = el(
      "summary",
      "",
      "검증 " + okCount + "/" + resp.claims.length + "개 주장" +
        (v.dropped ? " · 제외 " + v.dropped : "") +
        (v.flagged ? " · 경고 " + v.flagged : "") +
        (v.llm_checked ? "" : " · 의미 검증 생략")
    );
    details.appendChild(summary);
    var ul = el("ul");
    resp.claims.forEach(function (c) {
      var b = L.claimBadge(c.status);
      var li = el("li", b.ok ? "claim ok" : "claim warn");
      li.appendChild(el("span", "badge", b.icon));
      li.appendChild(el("span", "", " " + (c.display_text || c.text) + " "));
      (c.source_ids || []).forEach(function (sid) {
        li.appendChild(citeButton(String(sid).toUpperCase(), resp.citations));
      });
      li.title = b.label + (c.reason ? " - " + c.reason : "");
      if (!b.ok) li.appendChild(el("div", "muted reason", b.label + (c.reason ? ": " + c.reason : "")));
      ul.appendChild(li);
    });
    details.appendChild(ul);
    box.appendChild(details);
  }

  function renderAnswer(box, resp) {
    box.textContent = "";
    if (resp.not_found) {
      box.classList.add("not-found");
      box.appendChild(el("div", "banner", "🔍 책에서 찾지 못했습니다"));
    }
    if (resp.outside_book && !resp.not_found) {
      box.classList.add("outside");
      box.appendChild(el("div", "banner warn", "⚠️ 책에서 검증된 인용이 없는 답변입니다 (책 외부 지식)"));
    }
    var content = el("div", "md");
    renderMarkdown(content, resp.answer_markdown, resp.citations);
    box.appendChild(content);
    if (!resp.not_found && !resp.outside_book && resp.claims.length) {
      var allOk = resp.claims.every(function (c) {
        return c.status === "verified";
      });
      box.appendChild(el("div", allOk ? "status-line ok" : "status-line warn", allOk ? "✅ 모든 주장이 책 본문으로 검증됨" : "⚠️ 일부 주장은 검증이 완전하지 않습니다"));
    }
    renderClaims(box, resp);
    $("messages").scrollTop = $("messages").scrollHeight;
  }

  async function send() {
    var q = $("question").value.trim();
    if (!q || state.sending) return;
    var ids = Array.from(state.selected);
    if (!ids.length) {
      $("library-msg").textContent = "질문할 책을 먼저 선택하세요.";
      return;
    }
    state.sending = true;
    $("send").disabled = true;
    addMessage("user").textContent = q;
    $("question").value = "";
    var box = addMessage("assistant");
    box.textContent = "답변을 찾는 중...";
    try {
      var body = { book_ids: ids, question: q };
      if (state.conversationId) body.conversation_id = state.conversationId;
      var resp = await api("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      state.conversationId = resp.conversation_id;
      renderAnswer(box, resp);
    } catch (e) {
      box.textContent = "";
      box.classList.add("error");
      box.textContent = "오류: " + e.message;
    } finally {
      state.sending = false;
      $("send").disabled = false;
    }
  }

  // ---------- 서재 ----------

  function renderBooks() {
    var ul = $("book-list");
    ul.textContent = "";
    state.books.forEach(function (b) {
      var li = el("li", "book " + b.status);
      var row = el("div", "book-row");
      var cb = el("input");
      cb.type = "checkbox";
      cb.checked = state.selected.has(b.id);
      cb.disabled = b.status !== "ready";
      cb.setAttribute("aria-label", b.title + " 선택");
      cb.addEventListener("change", function () {
        if (cb.checked) state.selected.add(b.id);
        else state.selected.delete(b.id);
      });
      row.appendChild(cb);
      row.appendChild(el("span", "title", b.title));
      var del = el("button", "secondary del", "삭제");
      del.type = "button";
      del.addEventListener("click", function () {
        removeBook(b);
      });
      row.appendChild(del);
      li.appendChild(row);
      var info = (STATUS_KO[b.status] || b.status) + (b.num_pages ? " · " + b.num_pages + "쪽" : "");
      if (b.status === "ready" && b.num_chunks != null) info += " · 조각 " + b.num_chunks;
      li.appendChild(el("div", "muted small", info));
      if (BUSY[b.status]) {
        var bar = el("progress");
        bar.max = 1;
        bar.value = b.progress || 0;
        li.appendChild(bar);
      }
      if (b.message && (b.status === "error" || BUSY[b.status])) li.appendChild(el("div", "muted small", b.message));
      ul.appendChild(li);
    });
  }

  async function removeBook(b) {
    if (!window.confirm("'" + b.title + "' 책을 삭제할까요? 색인도 함께 지워집니다.")) return;
    try {
      await api("/books/" + encodeURIComponent(b.id), { method: "DELETE" });
      state.selected.delete(b.id);
      await loadBooks();
    } catch (e) {
      $("library-msg").textContent = "삭제 실패: " + e.message;
    }
  }

  async function loadBooks() {
    state.books = await api("/books");
    var ids = new Set(
      state.books.map(function (b) {
        return b.id;
      })
    );
    state.selected.forEach(function (id) {
      if (!ids.has(id)) state.selected.delete(id);
    });
    renderBooks();
    schedulePoll();
  }

  // 처리 중인 책이 있는 동안에만 상태를 폴링한다.
  function schedulePoll() {
    clearTimeout(state.pollTimer);
    var busy = state.books.filter(function (b) {
      return BUSY[b.status];
    });
    if (!busy.length) return;
    state.pollTimer = setTimeout(async function () {
      try {
        var done = false;
        await Promise.all(
          busy.map(async function (b) {
            var s = await api("/books/" + encodeURIComponent(b.id) + "/status");
            if (!BUSY[s.status]) done = true;
            Object.assign(b, s);
          })
        );
        renderBooks();
        if (done) await loadBooks();
        else schedulePoll();
      } catch (e) {
        $("library-msg").textContent = "상태 확인 실패: " + e.message;
        schedulePoll();
      }
    }, POLL_MS);
  }

  async function upload(file) {
    var fd = new FormData();
    fd.append("file", file);
    $("library-msg").textContent = "업로드 중...";
    try {
      await api("/books", { method: "POST", body: fd });
      $("library-msg").textContent = "";
      await loadBooks();
    } catch (e) {
      $("library-msg").textContent = "업로드 실패: " + e.message;
    }
  }

  function init() {
    $("file-input").addEventListener("change", function (ev) {
      var f = ev.target.files[0];
      ev.target.value = "";
      if (f) upload(f);
    });
    $("chat-form").addEventListener("submit", function (ev) {
      ev.preventDefault();
      send();
    });
    $("question").addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
        ev.preventDefault();
        send();
      }
    });
    $("new-chat").addEventListener("click", function () {
      state.conversationId = null;
      $("messages").textContent = "";
      $("source-panel").hidden = true;
    });
    $("close-panel").addEventListener("click", function () {
      $("source-panel").hidden = true;
    });
    loadBooks().catch(function (e) {
      $("library-msg").textContent = "서재를 불러오지 못했습니다: " + e.message;
    });
  }

  init();
})();
