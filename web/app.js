"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const EDGE_RENDER_ORDER = {
  callout: 0,
  side: 1,
  residual: 2,
  detail: 3,
  flow: 4,
};
const DEFAULT_DIAGRAM_PALETTE = {
  accent: "#12a5ed",
  accentFill: "#52b9ee",
};
const HEX_COLOR = /^#[0-9a-f]{6}$/i;

function $(selector, root = document) {
  return root.querySelector(selector);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${path}: ${response.status}`);
  }
  return response.json();
}

function templateLabel(template) {
  return {
    dense: "Dense",
    moe: "MoE",
    mla: "MLA",
    hybrid: "Hybrid",
    parallel: "Parallel",
  }[template] || template;
}

function roleLabel(role) {
  return {
    config: "Config",
    presets: "Presets",
    norm: "Norm",
    position: "Position",
    attention: "Attention",
    mixer: "Mixer",
    attention_helper: "Helper",
    mlp: "MLP",
    expert: "Expert",
    moe: "MoE",
    block: "Block",
    model: "Model",
    helper: "Helper",
  }[role] || role;
}

const PY_KEYWORDS = new Set([
  "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
  "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
  "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return",
  "try", "while", "with", "yield",
]);

const PY_BUILTINS = new Set([
  "bool", "dict", "enumerate", "float", "int", "len", "list", "max", "min", "next", "object",
  "print", "range", "set", "str", "sum", "super", "tuple", "zip", "self", "cls",
]);

function appendToken(fragment, text, className) {
  if (!text) return;
  if (!className) {
    fragment.append(document.createTextNode(text));
    return;
  }
  const span = document.createElement("span");
  span.className = className;
  span.textContent = text;
  fragment.append(span);
}

function highlightedPythonLine(line, state) {
  const fragment = document.createDocumentFragment();
  let i = 0;
  let expectName = "";
  let importMode = false;
  const firstCode = line.search(/\S/);

  while (i < line.length) {
    if (state.tripleQuote) {
      const end = line.indexOf(state.tripleQuote, i);
      if (end === -1) {
        appendToken(fragment, line.slice(i), "py-string");
        i = line.length;
      } else {
        appendToken(fragment, line.slice(i, end + 3), "py-string");
        i = end + 3;
        state.tripleQuote = "";
      }
      continue;
    }

    const char = line[i];
    if (/\s/.test(char)) {
      let j = i + 1;
      while (j < line.length && /\s/.test(line[j])) j += 1;
      appendToken(fragment, line.slice(i, j));
      i = j;
      continue;
    }

    if (char === "#") {
      appendToken(fragment, line.slice(i), "py-comment");
      break;
    }

    if (char === "@" && i === firstCode) {
      let j = i + 1;
      while (j < line.length && /[\w.]/.test(line[j])) j += 1;
      appendToken(fragment, line.slice(i, j), "py-decorator");
      i = j;
      continue;
    }

    if (char === "\"" || char === "'") {
      const triple = line.slice(i, i + 3);
      if (triple === char.repeat(3)) {
        const end = line.indexOf(triple, i + 3);
        if (end === -1) {
          appendToken(fragment, line.slice(i), "py-string");
          state.tripleQuote = triple;
          i = line.length;
        } else {
          appendToken(fragment, line.slice(i, end + 3), "py-string");
          i = end + 3;
        }
      } else {
        let j = i + 1;
        while (j < line.length) {
          if (line[j] === "\\") {
            j += 2;
          } else if (line[j] === char) {
            j += 1;
            break;
          } else {
            j += 1;
          }
        }
        appendToken(fragment, line.slice(i, j), "py-string");
        i = j;
      }
      continue;
    }

    if (/\d/.test(char)) {
      let j = i + 1;
      while (j < line.length && /[\w._]/.test(line[j])) j += 1;
      appendToken(fragment, line.slice(i, j), "py-number");
      i = j;
      continue;
    }

    if (/[A-Za-z_]/.test(char)) {
      let j = i + 1;
      while (j < line.length && /[A-Za-z0-9_]/.test(line[j])) j += 1;
      const word = line.slice(i, j);
      if (expectName) {
        appendToken(fragment, word, expectName);
        expectName = "";
      } else if (PY_KEYWORDS.has(word)) {
        appendToken(fragment, word, "py-keyword");
        if (word === "def") expectName = "py-function";
        if (word === "class") expectName = "py-class";
        importMode = word === "import" || word === "from";
      } else if (PY_BUILTINS.has(word)) {
        appendToken(fragment, word, word === "self" || word === "cls" ? "py-self" : "py-builtin");
      } else if (importMode && word !== "as") {
        appendToken(fragment, word, "py-module");
      } else {
        appendToken(fragment, word);
      }
      i = j;
      continue;
    }

    appendToken(fragment, char, /[=+\-*/%<>!&|^~:.,()[\]{}]/.test(char) ? "py-operator" : "");
    i += 1;
  }

  if (!line) appendToken(fragment, " ");
  return fragment;
}

function initIndex() {
  const grid = $("#modelGrid");
  const searchInput = $("#searchInput");
  const tierFilter = $("#tierFilter");
  const templateFilter = $("#templateFilter");
  const resultMeta = $("#resultMeta");

  fetchJson("data/index.json")
    .then((data) => {
      const models = data.models;
      for (const template of data.templates) {
        const option = document.createElement("option");
        option.value = template;
        option.textContent = templateLabel(template);
        templateFilter.appendChild(option);
      }

      function render() {
        const query = searchInput.value.trim().toLowerCase();
        const tier = tierFilter.value;
        const template = templateFilter.value;
        const filtered = models.filter((model) => {
          const haystack = `${model.slug} ${model.name} ${model.archetype} ${model.summary}`.toLowerCase();
          return (
            (!query || haystack.includes(query)) &&
            (tier === "all" || String(model.tier) === tier) &&
            (template === "all" || model.template === template)
          );
        });

        resultMeta.textContent = `${filtered.length} of ${models.length} models`;
        grid.replaceChildren(...filtered.map(modelCard));
      }

      searchInput.addEventListener("input", render);
      tierFilter.addEventListener("change", render);
      templateFilter.addEventListener("change", render);
      render();
    })
    .catch((error) => {
      grid.replaceChildren(errorBox(error));
    });
}

function modelCard(model) {
  const card = el("a", "model-card");
  card.href = `viewer.html?model=${encodeURIComponent(model.slug)}`;

  const top = el("div", "card-top");
  top.append(el("span", "badge", `Tier ${model.tier}`));
  top.append(el("span", `badge template-${model.template}`, templateLabel(model.template)));

  const title = el("h2", "", model.name);
  const slug = el("p", "slug", model.slug);
  const arch = el("p", "arch", model.archetype);
  const stats = el("p", "stats", `${model.line_count} lines · ${Object.keys(model.section_role_counts).length} section types`);

  card.append(top, title, slug, arch, stats);
  return card;
}

function errorBox(error) {
  const box = el("div", "error-box");
  box.append(el("strong", "", "Could not load visualizer data."));
  box.append(el("p", "", error.message));
  box.append(el("p", "", "Run the static server from the repo root: python -m http.server 8000 --directory web"));
  return box;
}

function initViewer() {
  const params = new URLSearchParams(window.location.search);
  const requestedSlug = params.get("model");
  const requestedSection = params.get("section");
  const requestedTarget = params.get("target");

  Promise.all([fetchJson("data/index.json"), requestedSlug ? Promise.resolve(null) : fetchJson("data/index.json")])
    .then(([index]) => {
      const slug = requestedSlug || index.models[0].slug;
      populateModelSelect(index.models, slug);
      return fetchJson(`data/${encodeURIComponent(slug)}.json`).then((model) => ({ index, model }));
    })
    .then(({ model }) => renderViewer(model, requestedSection, requestedTarget))
    .catch((error) => {
      $(".viewer-shell").replaceChildren(errorBox(error));
    });
}

function populateModelSelect(models, activeSlug) {
  const select = $("#modelSelect");
  select.replaceChildren(
    ...models.map((model) => {
      const option = document.createElement("option");
      option.value = model.slug;
      option.textContent = model.name;
      option.selected = model.slug === activeSlug;
      return option;
    })
  );
  select.addEventListener("change", () => {
    window.location.href = `viewer.html?model=${encodeURIComponent(select.value)}`;
  });
}

function renderViewer(model, requestedSection, requestedTarget) {
  document.title = `${model.name} · visualizer`;
  $("#viewerSlug").textContent = model.slug;
  $("#viewerName").textContent = model.name;
  $("#sourcePath").textContent = model.source_path;
  $("#modelFacts").replaceChildren(...modelFactNodes(model));
  window.currentModel = model;
  window.currentTargetId = "";
  renderDiagram(model);
  renderSectionNav(model.sections);
  renderCode(model);

  const requested = resolveTarget(model, requestedTarget) || resolveTarget(model, requestedSection);
  const initialRoles = model.template === "hybrid" ? ["mixer", "attention", "model"] : ["attention", "mixer", "model"];
  const initial =
    requested || initialRoles.map((role) => model.sections.find((section) => section.role === role)).find(Boolean);
  if (initial) activateTarget(model, initial.id, { scroll: Boolean(requested), updateUrl: false });
}

function modelFactNodes(model) {
  const facts = [];
  facts.push(pill(`Tier ${model.tier}`));
  facts.push(pill(templateLabel(model.template)));
  if (model.release) facts.push(pill(model.release));
  facts.push(el("p", "fact-line", model.archetype));

  const links = el("div", "fact-links");
  if (model.links.gallery) links.append(link("Gallery", model.links.gallery));
  if (model.links.tech_report) links.append(link("Tech report", model.links.tech_report));
  facts.push(links);
  return facts;
}

function pill(text) {
  return el("span", "pill", text);
}

function link(text, href) {
  const anchor = el("a", "", text);
  anchor.href = href;
  anchor.target = "_blank";
  anchor.rel = "noreferrer";
  return anchor;
}

function renderSectionNav(sections) {
  const nav = $("#sectionNav");
  nav.replaceChildren(
    ...sections.map((section) => {
      const button = el("button", `section-chip role-${section.role}`);
      button.type = "button";
      button.dataset.sectionId = section.id;
      button.dataset.targetId = section.id;
      button.title = `${section.line_start}-${section.line_end}`;
      button.append(el("span", "chip-role", roleLabel(section.role)));
      button.append(el("span", "chip-label", section.label));
      button.addEventListener("click", () => activateTarget(window.currentModel, section.id));
      return button;
    })
  );
}

function renderCode(model) {
  const code = $("#codeLines");
  const fragment = document.createDocumentFragment();
  const highlightState = { tripleQuote: "" };
  model.source_lines.forEach((line, index) => {
    const number = index + 1;
    const row = el("span", "code-line");
    row.dataset.line = String(number);
    row.addEventListener("click", () => {
      const target = targetForLine(model, number);
      if (target) activateTarget(model, target.id, { scroll: false });
    });

    const gutter = el("span", "line-number", String(number));
    const text = el("span", "line-text");
    text.append(highlightedPythonLine(line, highlightState));
    row.append(gutter, text);
    fragment.append(row);
  });
  code.replaceChildren(fragment);
}

function renderDiagram(model) {
  if (model.diagram.artwork) {
    renderArtworkDiagram(model);
  } else {
    renderGeneratedDiagram(model);
  }
  restoreActiveDiagramTarget(model);
}

function restoreActiveDiagramTarget(model) {
  if (window.currentTargetId && resolveTarget(model, window.currentTargetId)) {
    activateTarget(model, window.currentTargetId, { scroll: false, updateUrl: false });
  }
}

function diagramSvg(model, viewBox) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", viewBox);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `${model.name} architecture diagram`);
  svg.classList.add("diagram");
  applyDiagramPalette(svg, model.diagram.palette);
  return svg;
}

function renderArtworkDiagram(model) {
  const wrap = $("#diagramWrap");
  const diagram = model.diagram;
  const artwork = diagram.artwork;
  const svg = diagramSvg(model, `0 0 ${artwork.width} ${artwork.height}`);

  const image = document.createElementNS(SVG_NS, "image");
  image.setAttribute("href", artwork.path);
  image.setAttribute("x", "0");
  image.setAttribute("y", "0");
  image.setAttribute("width", String(artwork.width));
  image.setAttribute("height", String(artwork.height));
  image.classList.add("diagram-artwork-image");
  svg.append(image);

  if (diagram.hotspotsCoordinateSpace === "artwork") {
    for (const hotspot of diagram.hotspots || []) {
      svg.append(renderDiagramHotspot(hotspot));
    }
  }

  wrap.replaceChildren(svg);
}

function renderGeneratedDiagram(model) {
  const wrap = $("#diagramWrap");
  const diagram = model.diagram;
  const svg = diagramSvg(model, diagram.viewBox);

  const defs = document.createElementNS(SVG_NS, "defs");
  const marker = document.createElementNS(SVG_NS, "marker");
  marker.setAttribute("id", "arrowhead");
  marker.setAttribute("markerWidth", "8");
  marker.setAttribute("markerHeight", "8");
  marker.setAttribute("refX", "6");
  marker.setAttribute("refY", "3");
  marker.setAttribute("orient", "auto");
  const markerPath = document.createElementNS(SVG_NS, "path");
  markerPath.setAttribute("d", "M0,0 L0,6 L6,3 z");
  marker.append(markerPath);
  defs.append(marker);
  svg.append(defs);

  for (const item of diagram.groups || []) {
    svg.append(renderDiagramGroup(item));
  }
  for (const item of sortedDiagramEdges(diagram)) {
    svg.append(renderDiagramEdge(item));
  }
  for (const item of diagram.nodes || []) {
    svg.append(renderDiagramNode(item));
  }
  for (const item of diagram.decorations || []) {
    svg.append(renderDiagramDecoration(item));
  }
  const annotationLeaderLayouts = [];
  for (const item of diagram.annotations || []) {
    const { group, layoutLeader } = renderDiagramAnnotation(item);
    svg.append(group);
    if (layoutLeader) annotationLeaderLayouts.push(layoutLeader);
  }

  wrap.replaceChildren(svg);
  for (const layoutLeader of annotationLeaderLayouts) {
    layoutLeader();
  }
}

function applyDiagramPalette(svg, palette = {}) {
  const accent = HEX_COLOR.test(palette.accent || "") ? palette.accent : DEFAULT_DIAGRAM_PALETTE.accent;
  const accentFill = HEX_COLOR.test(palette.accentFill || "")
    ? palette.accentFill
    : DEFAULT_DIAGRAM_PALETTE.accentFill;
  svg.style.setProperty("--diagram-accent", accent);
  svg.style.setProperty("--diagram-accent-fill", accentFill);
}

function sortedDiagramEdges(diagram) {
  return [...(diagram.edges || diagram.arrows || [])]
    .map((item, index) => ({ item, index }))
    .sort((a, b) => {
      const aOrder = EDGE_RENDER_ORDER[a.item.role || "flow"] ?? EDGE_RENDER_ORDER.flow;
      const bOrder = EDGE_RENDER_ORDER[b.item.role || "flow"] ?? EDGE_RENDER_ORDER.flow;
      return aOrder - bOrder || a.index - b.index;
    })
    .map(({ item }) => item);
}

function interactiveSvgGroup(item, baseClass) {
  const group = document.createElementNS(SVG_NS, "g");
  group.classList.add(baseClass, `${baseClass}-${item.role || "default"}`);
  if (item.tone) group.classList.add(`${baseClass}-tone-${item.tone}`);
  attachInteractiveTarget(group, item);
  return group;
}

function attachInteractiveTarget(node, item) {
  node.dataset.targetId = item.target_id || item.section_id || "";
  node.dataset.sectionId = item.section_id || "";
  if (node.dataset.targetId) {
    node.setAttribute("tabindex", "0");
    node.setAttribute("role", "button");
    node.addEventListener("click", () => activateTarget(window.currentModel, node.dataset.targetId));
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activateTarget(window.currentModel, node.dataset.targetId);
      }
    });
  }
}

function renderDiagramHotspot(item) {
  const group = document.createElementNS(SVG_NS, "g");
  group.classList.add("diagram-hotspot", `diagram-hotspot-${item.role || "default"}`);
  attachInteractiveTarget(group, item);

  const title = document.createElementNS(SVG_NS, "title");
  title.textContent = item.label || item.id || "Code section";
  group.append(title);

  const rect = document.createElementNS(SVG_NS, "rect");
  rect.setAttribute("x", String(item.x));
  rect.setAttribute("y", String(item.y));
  rect.setAttribute("width", String(item.w));
  rect.setAttribute("height", String(item.h));
  if (item.shape === "roundrect") {
    rect.setAttribute("rx", String(item.rx || 0));
    rect.setAttribute("ry", String(item.ry || item.rx || 0));
  } else {
    rect.setAttribute("rx", "0");
    rect.setAttribute("ry", "0");
  }
  group.append(rect);
  return group;
}

function renderDiagramGroup(item) {
  const group = interactiveSvgGroup(item, "diagram-group");
  if (item.outline) group.classList.add(`diagram-group-outline-${item.outline}`);
  const rect = document.createElementNS(SVG_NS, "rect");
  rect.setAttribute("x", String(item.x));
  rect.setAttribute("y", String(item.y));
  rect.setAttribute("width", String(item.w));
  rect.setAttribute("height", String(item.h));
  rect.setAttribute("rx", String(item.rx || 18));
  group.append(rect);
  if (item.showLabel !== false && item.label) {
    appendSvgText(group, item.label, item.x + 22, item.y + 28, {
      anchor: "start",
      className: "group-label",
      maxChars: item.role === "block" ? 22 : 28,
      yOffset: item.role === "block" ? -10 : 0,
    });
  }
  if (item.badge) {
    const badge = document.createElementNS(SVG_NS, "text");
    badge.setAttribute("x", String(item.x - 8));
    badge.setAttribute("y", String(item.y + item.h - 18));
    badge.setAttribute("text-anchor", "end");
    badge.setAttribute("class", "group-badge");
    badge.textContent = item.badge;
    group.append(badge);
  }
  return group;
}

function renderDiagramEdge(item) {
  const path = document.createElementNS(SVG_NS, "path");
  const classes = ["diagram-edge", `edge-${item.role || "flow"}`];
  if (item.dashed) classes.push("dashed");
  if (item.tone) classes.push(`edge-tone-${item.tone}`);
  path.setAttribute("class", classes.join(" "));
  path.dataset.edgeId = item.id || "";
  path.dataset.targetId = item.target_id || item.section_id || "";
  path.dataset.sectionId = item.section_id || "";
  if (item.path) {
    path.setAttribute("d", item.path);
  } else if (item.from && item.to) {
    path.setAttribute("d", "");
  }
  if (item.arrow !== false) path.setAttribute("marker-end", "url(#arrowhead)");
  return path;
}

function renderDiagramNode(item) {
  const group = interactiveSvgGroup(item, "diagram-node");
  group.classList.add(`node-${item.role || "default"}`);
  if (item.shape === "circle") {
    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", String(item.x + item.w / 2));
    circle.setAttribute("cy", String(item.y + item.h / 2));
    circle.setAttribute("r", String(Math.min(item.w, item.h) / 2));
    group.append(circle);
  } else {
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", String(item.x));
    rect.setAttribute("y", String(item.y));
    rect.setAttribute("width", String(item.w));
    rect.setAttribute("height", String(item.h));
    rect.setAttribute("rx", String(item.rx || 8));
    group.append(rect);
  }
  appendSvgText(group, item.label, item.x + item.w / 2, item.y + item.h / 2, {
    maxChars: item.w >= 220 ? 28 : item.w < 130 ? 13 : 18,
    className: "node-label",
  });
  if (item.subtitle) {
    appendSvgText(group, item.subtitle, item.x + item.w / 2, item.y + item.h + 34, {
      className: "node-subtitle",
      maxChars: 36,
      maxLines: 1,
    });
  }
  return group;
}

function renderDiagramAnnotation(item) {
  const group = interactiveSvgGroup(item, "diagram-annotation");
  let leader = null;
  if (item.to) {
    leader = document.createElementNS(SVG_NS, "path");
    leader.setAttribute("class", "annotation-leader");
    group.append(leader);
  }

  const textGroup = document.createElementNS(SVG_NS, "g");
  if (item.lines) {
    appendSvgRichLines(textGroup, item.lines, item.x, item.y, {
      anchor: "start",
      className: "annotation-label",
      lineHeight: item.lineHeight || 20,
    });
  } else {
    appendSvgText(textGroup, item.label, item.x, item.y, {
      anchor: "start",
      className: "annotation-label",
      maxChars: 24,
      maxLines: 6,
    });
    if (item.value) {
      appendSvgText(textGroup, item.value, item.x, item.y + 28, {
        anchor: "start",
        className: "annotation-value",
        maxChars: 24,
        maxLines: 3,
      });
    }
  }
  group.append(textGroup);

  const layoutLeader =
    leader && item.to
      ? () => {
          const start = annotationLeaderStart(item, textGroup);
          leader.setAttribute("d", `M${start.x},${start.y} L${item.to.x},${item.to.y}`);
        }
      : null;
  return { group, layoutLeader };
}

function renderDiagramDecoration(item) {
  const group = document.createElementNS(SVG_NS, "g");
  group.classList.add("diagram-decoration", `diagram-decoration-${item.role || "default"}`);
  if (item.path) {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", item.path);
    group.append(path);
  }
  if (item.lines) {
    appendSvgRichLines(group, item.lines, item.x || 0, item.y || 0, {
      anchor: "start",
      className: "decoration-label",
      lineHeight: item.lineHeight || 20,
    });
  } else if (item.text) {
    appendSvgText(group, item.text, item.x || 0, item.y || 0, {
      anchor: "start",
      className: "decoration-label",
      maxChars: 40,
      maxLines: 1,
    });
  }
  return group;
}

function annotationLeaderStart(item, textGroup) {
  let bbox;
  try {
    bbox = textGroup.getBBox();
  } catch {
    return { x: item.x, y: item.y + 18 };
  }

  if (!Number.isFinite(bbox.x) || !Number.isFinite(bbox.y)) {
    return { x: item.x, y: item.y + 18 };
  }

  const side = annotationLeaderSide(item);
  const centerX = bbox.x + bbox.width / 2;
  const centerY = bbox.y + bbox.height / 2;
  if (side === "top") return { x: centerX, y: bbox.y };
  if (side === "bottom") return { x: centerX, y: bbox.y + bbox.height };
  if (side === "right") return { x: bbox.x + bbox.width, y: centerY };
  return { x: bbox.x, y: centerY };
}

function annotationLeaderSide(item) {
  if (["left", "right", "top", "bottom"].includes(item.side)) {
    return item.side;
  }
  return item.to && item.to.x >= item.x ? "right" : "left";
}

function appendSvgText(group, label, x, y, options = {}) {
  const { anchor = "middle", className = "", maxChars = 18, maxLines = 2, yOffset = 0 } = options;
  const lines = splitLabel(String(label || ""), maxChars, maxLines);
  lines.forEach((lineText, index) => {
    const text = document.createElementNS(SVG_NS, "text");
    text.setAttribute("x", String(x));
    text.setAttribute("y", String(y + yOffset + (index - (lines.length - 1) / 2) * 15 + 5));
    text.setAttribute("text-anchor", anchor);
    if (className) text.setAttribute("class", className);
    text.textContent = lineText;
    group.append(text);
  });
}

function appendSvgRichLines(group, lines, x, y, options = {}) {
  const { anchor = "start", className = "", lineHeight = 20 } = options;
  const normalized = normalizeRichLines(lines);
  normalized.forEach((line, index) => {
    const text = document.createElementNS(SVG_NS, "text");
    text.setAttribute("x", String(x));
    text.setAttribute("y", String(y + index * lineHeight));
    text.setAttribute("text-anchor", anchor);
    if (className) text.setAttribute("class", className);
    for (const run of line) {
      const tspan = document.createElementNS(SVG_NS, "tspan");
      if (run.tone) tspan.setAttribute("class", `text-tone-${run.tone}`);
      tspan.textContent = run.text;
      text.append(tspan);
    }
    group.append(text);
  });
}

function normalizeRichLines(lines) {
  return lines.map((line) => {
    if (typeof line === "string") return [{ text: line }];
    if (Array.isArray(line)) return line.map((run) => (typeof run === "string" ? { text: run } : run));
    return [{ text: String(line?.text || "") }];
  });
}

function splitLabel(label, maxChars = 18, maxLines = 2) {
  if (label.length <= maxChars) return [label];
  const words = label.replace(/([_/])/g, "$1 ").split(/\s+/);
  const lines = [];
  let current = "";
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length > maxChars && current) {
      lines.push(current);
      current = word;
    } else {
      current = candidate;
    }
  }
  if (current) lines.push(current);
  return Number.isFinite(maxLines) ? lines.slice(0, maxLines) : lines;
}

function sectionForLine(sections, line) {
  return sections.find((section) => line >= section.line_start && line <= section.line_end);
}

function targetForLine(model, line) {
  const anchors = (model.anchors || [])
    .filter((anchor) => line >= anchor.line_start && line <= anchor.line_end)
    .sort((a, b) => a.line_count - b.line_count || a.line_start - b.line_start);
  return anchors[0] || sectionForLine(model.sections, line);
}

function resolveTarget(model, targetId) {
  if (!targetId) return null;
  return (
    (model.anchors || []).find((anchor) => anchor.id === targetId) ||
    model.sections.find((section) => section.id === targetId) ||
    null
  );
}

function parentSectionId(target) {
  return target.section_id || target.id;
}

function escapeAttr(value) {
  return String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function activateTarget(model, targetId, options = {}) {
  const { scroll = true, updateUrl = true } = options;
  const target = resolveTarget(model, targetId);
  if (!target) return;
  window.currentTargetId = target.id;
  const sectionId = parentSectionId(target);
  const section = model.sections.find((item) => item.id === sectionId);
  const exactTargetId = target.id;

  document.querySelectorAll(".code-line.active").forEach((line) => line.classList.remove("active"));
  document.querySelectorAll(".code-line.section-active").forEach((line) => line.classList.remove("section-active"));
  if (section) {
    for (let line = section.line_start; line <= section.line_end; line += 1) {
      const row = document.querySelector(`.code-line[data-line="${line}"]`);
      if (row) row.classList.add("section-active");
    }
  }
  for (let line = target.line_start; line <= target.line_end; line += 1) {
    const row = document.querySelector(`.code-line[data-line="${line}"]`);
    if (row) row.classList.add("active");
  }

  document
    .querySelectorAll(
      ".diagram-node.active, .diagram-group.active, .diagram-annotation.active, .diagram-edge.active, .diagram-hotspot.active"
    )
    .forEach((node) => node.classList.remove("active"));
  document
    .querySelectorAll(`[data-target-id="${escapeAttr(exactTargetId)}"]`)
    .forEach((node) => node.classList.add("active"));

  document.querySelectorAll(".section-chip.active").forEach((chip) => chip.classList.remove("active"));
  document.querySelectorAll(`.section-chip[data-section-id="${sectionId}"]`).forEach((chip) => chip.classList.add("active"));

  $("#activeRange").textContent = `${target.label} · lines ${target.line_start}-${target.line_end}`;
  if (scroll) {
    const start = document.querySelector(`.code-line[data-line="${target.line_start}"]`);
    if (start) start.scrollIntoView({ block: "center", behavior: "smooth" });
  }
  if (updateUrl) {
    const params = new URLSearchParams(window.location.search);
    params.set("model", model.slug);
    params.set("section", sectionId);
    if (exactTargetId !== sectionId) {
      params.set("target", exactTargetId);
    } else {
      params.delete("target");
    }
    window.history.replaceState(null, "", `${window.location.pathname}?${params.toString()}`);
  }
}

const page = document.body.dataset.page;
if (page === "index") initIndex();
if (page === "viewer") initViewer();
