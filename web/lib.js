// 순수 함수 모음(DOM 비의존). 브라우저에서는 전역 TLLib, node에서는 module.exports로 쓴다.
(function (root) {
  "use strict";

  var OUTSIDE_HEADING = "보충(책 외부)";
  // 마크다운 변환기가 수식 속 _ * \ 를 망가뜨리지 않도록, 수식은 자리표시자로 빼 둔다.
  var MATH_RE = /\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$(?![\s$])[^$\n]+?(?<![\s\\])\$(?!\d)/g;
  var PH_OPEN = "";
  var PH_CLOSE = "";
  var CITE_RE = /\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]/g;

  function extractMath(md) {
    var items = [];
    var text = String(md).replace(MATH_RE, function (m) {
      var display = false;
      var tex;
      if (m.startsWith("$$")) {
        display = true;
        tex = m.slice(2, -2);
      } else if (m.startsWith("\\[")) {
        display = true;
        tex = m.slice(2, -2);
      } else if (m.startsWith("\\(")) {
        tex = m.slice(2, -2);
      } else {
        tex = m.slice(1, -1);
      }
      items.push({ tex: tex.trim(), display: display });
      return PH_OPEN + "M" + (items.length - 1) + PH_CLOSE;
    });
    return { text: text, items: items };
  }

  // 텍스트를 일반 조각과 [S#] 인용 조각으로 나눈다. "[S1, S2]"는 인용 둘로 펼친다.
  function splitCitations(text) {
    var parts = [];
    var last = 0;
    var m;
    CITE_RE.lastIndex = 0;
    while ((m = CITE_RE.exec(text)) !== null) {
      if (m.index > last) parts.push({ type: "text", value: text.slice(last, m.index) });
      m[1].split(",").forEach(function (sid) {
        parts.push({ type: "cite", sid: sid.trim() });
      });
      last = m.index + m[0].length;
    }
    if (last < text.length) parts.push({ type: "text", value: text.slice(last) });
    return parts;
  }

  function claimBadge(status) {
    switch (status) {
      case "verified":
        return { ok: true, icon: "✅", label: "검증됨" };
      case "unchecked":
        return { ok: false, icon: "⚠️", label: "인용문만 일치(의미 검증 안 함)" };
      case "rejected":
        return { ok: false, icon: "⚠️", label: "인용문이 원문과 다름" };
      case "unsupported":
        return { ok: false, icon: "⚠️", label: "근거가 주장을 뒷받침하지 않음" };
      default:
        return { ok: false, icon: "⚠️", label: "확인되지 않음" };
    }
  }

  function pageRange(c, max) {
    var out = [];
    for (var p = c.page_start; p <= c.page_end && out.length < (max || 4); p++) out.push(p);
    return out;
  }

  var api = {
    OUTSIDE_HEADING: OUTSIDE_HEADING,
    PH_OPEN: PH_OPEN,
    PH_CLOSE: PH_CLOSE,
    extractMath: extractMath,
    splitCitations: splitCitations,
    claimBadge: claimBadge,
    pageRange: pageRange,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TLLib = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
