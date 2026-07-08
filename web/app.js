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
const VIEWER_TAB_STORAGE_PREFIX = "llm-gallery-viewer-tab:";

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

let DATA_VERSION = "";

function versionedDataPath(path, version = DATA_VERSION) {
  if (!version) return path;
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}v=${encodeURIComponent(version)}`;
}

function modelDataPath(slug, version = DATA_VERSION) {
  return versionedDataPath(`data/${encodeURIComponent(slug)}.json`, version);
}

function modelUrl(slug, extra = {}) {
  const params = new URLSearchParams({ model: slug, ...extra });
  return `viewer.html?${params.toString()}`;
}

function galleryHref(model) {
  if (!model.links?.gallery) return "";
  if (!model.gallery_card_id || model.links.gallery.includes("#")) return model.links.gallery;
  const anchor = model.gallery_card_id.startsWith("card-") ? model.gallery_card_id : `card-${model.gallery_card_id}`;
  const base = model.links.gallery.endsWith("/") ? model.links.gallery : `${model.links.gallery}/`;
  return `${base}#${anchor}`;
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

function initIndex() {
  const grid = $("#modelGrid");
  const searchInput = $("#searchInput");
  const tierFilter = $("#tierFilter");
  const templateFilter = $("#templateFilter");
  const resultMeta = $("#resultMeta");

  fetchJson("data/index.json")
    .then((data) => {
      DATA_VERSION = data.data_version || "";
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
  installModelPrefetch(card, model.slug);

  const top = el("div", "card-top");
  top.append(el("span", "badge", `Tier ${model.tier}`));
  top.append(el("span", `badge template-${model.template}`, templateLabel(model.template)));

  const title = el("h2", "", model.name);
  const slug = el("p", "slug", model.slug);
  const arch = el("p", "arch", model.archetype);
  const stats = el(
    "p",
    "stats",
    [model.parameter_scale, model.release_year, `${Object.keys(model.section_role_counts).length} section types`]
      .filter(Boolean)
      .join(" · ")
  );

  card.append(top, title, slug, arch, stats);
  return card;
}

function installModelPrefetch(node, slug) {
  node.addEventListener(
    "pointerenter",
    () => {
      fetch(modelDataPath(slug), { priority: "low" }).catch(() => {});
    },
    { once: true }
  );
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

  fetchJson("data/index.json")
    .then((index) => {
      DATA_VERSION = index.data_version || "";
      const known = new Set(index.models.map((model) => model.slug));
      const fallback = index.models[0]?.slug;
      const slug = requestedSlug && known.has(requestedSlug) ? requestedSlug : fallback;
      if (!slug) throw new Error("No models are available in data/index.json");
      populateModelSelect(index.models, slug);
      setupModelNav(index.models, slug);
      if (requestedSlug && requestedSlug !== slug) {
        showNotice(`Unknown model "${requestedSlug}" - showing ${slug} instead.`);
        const clean = new URLSearchParams(window.location.search);
        clean.set("model", slug);
        clean.delete("section");
        clean.delete("target");
        window.history.replaceState(null, "", `${window.location.pathname}?${clean.toString()}`);
      }
      return fetchJson(modelDataPath(slug, index.data_version)).then((model) => ({ index, model }));
    })
    .then(({ index, model }) => renderViewer(model, index, requestedSection, requestedTarget))
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
    window.location.href = modelUrl(select.value);
  });
}

function setupModelNav(models, activeSlug) {
  const index = models.findIndex((model) => model.slug === activeSlug);
  const prev = $("#prevModel");
  const next = $("#nextModel");
  if (index < 0 || !prev || !next) return;
  const prevModel = models[(index - 1 + models.length) % models.length];
  const nextModel = models[(index + 1) % models.length];
  prev.addEventListener("click", () => {
    window.location.href = modelUrl(prevModel.slug);
  });
  next.addEventListener("click", () => {
    window.location.href = modelUrl(nextModel.slug);
  });
}

function showNotice(message) {
  const notice = $("#viewerNotice");
  if (!notice) return;
  notice.textContent = message;
  notice.hidden = false;
}

function renderViewer(model, index, requestedSection, requestedTarget) {
  document.title = `${model.name} · visualizer`;
  $("#viewerSlug").textContent = model.slug;
  $("#viewerName").textContent = model.name;
  $("#sourcePath").textContent = model.source_path;
  $("#modelFacts").replaceChildren(...modelFactNodes(model));
  renderRelatedModels(index.models, model);
  window.currentModel = model;
  window.currentIndex = index;
  window.currentTargetId = "";
  renderDiagram(model);
  renderSectionNav(model.sections);
  renderCode(model);
  setupViewerTabs(model);
  setupViewerActions(model, index);

  const requested = resolveTarget(model, requestedTarget) || resolveTarget(model, requestedSection);
  const initialRoles = model.template === "hybrid" ? ["mixer", "attention", "model"] : ["attention", "mixer", "model"];
  const initial =
    requested || initialRoles.map((role) => model.sections.find((section) => section.role === role)).find(Boolean);
  if (initial) activateTarget(model, initial.id, { scroll: Boolean(requested), updateUrl: false });
}

function modelFactNodes(model) {
  const facts = [];
  facts.push(el("p", "fact-line", model.archetype));
  if (model.release) facts.push(el("p", "fact-meta", `Released ${model.release}`));
  for (const note of model.notes || []) {
    const node = el("p", `fact-note note-${note.kind}`, `${note.kind}: ${note.text}`);
    facts.push(node);
  }

  const links = el("div", "fact-links");
  const gallery = galleryHref(model);
  if (gallery) links.append(link("Gallery", gallery));
  if (links.childElementCount) facts.push(links);
  return facts;
}

function formatValue(value) {
  if (value === undefined) return "";
  if (value === null) return "null";
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString() : String(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}

function normalizedReportUrl(rawUrl) {
  if (!rawUrl) return "";
  let url;
  try {
    url = new URL(rawUrl);
  } catch {
    return "";
  }

  if (url.hostname === "arxiv.org" && url.pathname.startsWith("/abs/")) {
    url.pathname = url.pathname.replace(/^\/abs\//, "/pdf/");
    return url.toString();
  }
  if (url.hostname === "huggingface.co" && url.pathname.includes("/blob/") && /\.pdf$/i.test(url.pathname)) {
    url.pathname = url.pathname.replace("/blob/", "/resolve/");
    return url.toString();
  }
  return url.toString();
}

function embeddableReportUrl(rawUrl) {
  const url = normalizedReportUrl(rawUrl);
  if (!url) return "";
  try {
    const parsed = new URL(url);
    if (parsed.hostname === "arxiv.org" && parsed.pathname.startsWith("/pdf/")) return url;
    if (/\.pdf$/i.test(parsed.pathname)) return url;
  } catch {
    return "";
  }
  return "";
}

function setupViewerTabs(model) {
  const codePanel = $("#codeTabPanel");
  const reportPanel = $("#reportTabPanel");
  const reportFrame = $("#reportFrame");
  const fallback = $("#reportFallback");
  const fallbackText = $("#reportFallbackText");
  const openLink = $("#reportOpenLink");
  if (!codePanel || !reportPanel) return;

  const originalReportUrl = model.links?.tech_report || "";
  const embedUrl = embeddableReportUrl(originalReportUrl);
  if (embedUrl) {
    reportFrame.src = embedUrl;
    reportFrame.hidden = false;
    fallback.hidden = true;
  } else {
    reportFrame.removeAttribute("src");
    reportFrame.hidden = true;
    fallback.hidden = false;
    fallbackText.textContent = originalReportUrl
      ? "This report is available outside the embedded viewer."
      : "No tech report link is available for this model.";
    openLink.hidden = !originalReportUrl;
    if (originalReportUrl) openLink.href = originalReportUrl;
  }

  const storageKey = `${VIEWER_TAB_STORAGE_PREFIX}${model.slug}`;
  const storedTab = localStorage.getItem(storageKey) === "report" ? "report" : "code";

  function setActiveTab(tabName, persist = true) {
    const active = tabName === "report" ? "report" : "code";
    document.querySelectorAll("[data-viewer-tab]").forEach((tab) => {
      const selected = tab.dataset.viewerTab === active;
      tab.classList.toggle("active", selected);
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    codePanel.hidden = active !== "code";
    reportPanel.hidden = active !== "report";
    codePanel.classList.toggle("active", active === "code");
    reportPanel.classList.toggle("active", active === "report");
    if (persist) localStorage.setItem(storageKey, active);
  }

  document.querySelectorAll("[data-viewer-tab]").forEach((tab) => {
    tab.addEventListener("click", () => setActiveTab(tab.dataset.viewerTab));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const next = tab.dataset.viewerTab === "code" ? "report" : "code";
      setActiveTab(next);
      document.querySelector(`[data-viewer-tab="${next}"]`)?.focus();
    });
  });

  setActiveTab(storedTab, false);
}

function renderRelatedModels(models, active) {
  const wrap = $("#relatedModels");
  if (!wrap) return;
  const archWords = new Set(active.archetype.toLowerCase().split(/[^a-z0-9.]+/).filter((word) => word.length > 3));
  const scored = models
    .filter((model) => model.slug !== active.slug)
    .map((model) => {
      const modelWords = new Set(model.archetype.toLowerCase().split(/[^a-z0-9.]+/).filter((word) => word.length > 3));
      let score = 0;
      if (model.template === active.template) score += 4;
      if (model.tier === active.tier) score += 1;
      for (const word of archWords) {
        if (modelWords.has(word)) score += 1;
      }
      return { model, score };
    })
    .filter((item) => item.score > 2)
    .sort((a, b) => b.score - a.score || a.model.slug.localeCompare(b.model.slug))
    .slice(0, 6);
  const label = el("span", "related-label", "Related");
  const links = scored.map(({ model }) => {
    const anchor = el("a", "", model.name);
    anchor.href = modelUrl(model.slug);
    anchor.title = `${templateLabel(model.template)} · ${model.archetype}`;
    return anchor;
  });
  wrap.replaceChildren(label, ...links);
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
      button.dataset.searchText = `${section.role} ${section.label} ${section.summary || ""}`.toLowerCase();
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
    text.append(renderHighlightedLine(model.source_tokens?.[index], line));
    row.append(gutter, text);
    fragment.append(row);
  });
  code.replaceChildren(fragment);
}

function renderHighlightedLine(tokens, fallbackLine) {
  const fragment = document.createDocumentFragment();
  if (!Array.isArray(tokens) || !tokens.length) {
    appendToken(fragment, fallbackLine || " ");
    return fragment;
  }
  for (const token of tokens) {
    appendToken(fragment, token.t, token.c || "");
  }
  return fragment;
}

function setupViewerActions(model, index) {
  const compareLink = $("#compareLink");
  if (compareLink) {
    const next = nextModel(index.models, model.slug);
    compareLink.href = `compare.html?left=${encodeURIComponent(model.slug)}&right=${encodeURIComponent(next.slug)}`;
  }

  $("#copyActiveLink")?.addEventListener("click", () => copyText(window.location.href));
  $("#copySection")?.addEventListener("click", () => {
    const target = resolveTarget(model, window.currentTargetId);
    if (target) copyText(linesForRange(model, target.line_start, target.line_end));
  });
  $("#copyCode")?.addEventListener("click", () => copyText(model.source_lines.join("\n")));

  document.addEventListener("keydown", function shortcutListener(event) {
    handleViewerShortcut(event, model, index);
  });
}

function nextModel(models, slug, direction = 1) {
  const index = models.findIndex((model) => model.slug === slug);
  if (index < 0) return models[0];
  return models[(index + direction + models.length) % models.length];
}

function isTypingTarget(target) {
  return ["INPUT", "SELECT", "TEXTAREA"].includes(target?.tagName) || target?.isContentEditable;
}

function handleViewerShortcut(event, model, index) {
  if (isTypingTarget(event.target)) return;
  if (event.key === "[") {
    window.location.href = modelUrl(nextModel(index.models, model.slug, -1).slug);
  } else if (event.key === "]") {
    window.location.href = modelUrl(nextModel(index.models, model.slug, 1).slug);
  } else if (event.key === "j" || event.key === "k") {
    event.preventDefault();
    activateAdjacentSection(model, event.key === "j" ? 1 : -1);
  } else if (event.key.toLowerCase() === "c") {
    copyText(window.location.href);
  }
}

function activateAdjacentSection(model, direction) {
  const current = resolveTarget(model, window.currentTargetId);
  const currentSectionId = current ? parentSectionId(current) : model.sections[0]?.id;
  const index = model.sections.findIndex((section) => section.id === currentSectionId);
  if (index < 0) return;
  const next = model.sections[(index + direction + model.sections.length) % model.sections.length];
  activateTarget(model, next.id);
}

function linesForRange(model, start, end) {
  return model.source_lines.slice(start - 1, end).join("\n");
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = el("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
}

function renderDiagram(model, wrap = $("#diagramWrap")) {
  if (model.diagram.artwork) {
    renderArtworkDiagram(model, wrap);
  } else {
    renderGeneratedDiagram(model, wrap);
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

function renderArtworkDiagram(model, wrap = $("#diagramWrap")) {
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

  mountDiagram(wrap, svg);
}

function renderGeneratedDiagram(model, wrap = $("#diagramWrap")) {
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

  mountDiagram(wrap, svg);
  for (const layoutLeader of annotationLeaderLayouts) {
    layoutLeader();
  }
}

function mountDiagram(wrap, svg) {
  if (!wrap) return;
  const viewport = el("div", "diagram-viewport");
  viewport.setAttribute("tabindex", "0");
  viewport.setAttribute("aria-label", "Pan and zoom diagram");
  viewport.append(svg);

  const controls = el("div", "zoom-controls");
  const zoomOut = el("button", "zoom-button", "-");
  const reset = el("button", "zoom-button", "Reset");
  const zoomIn = el("button", "zoom-button", "+");
  for (const button of [zoomOut, reset, zoomIn]) {
    button.type = "button";
  }
  zoomOut.setAttribute("aria-label", "Zoom out");
  reset.setAttribute("aria-label", "Reset zoom");
  zoomIn.setAttribute("aria-label", "Zoom in");
  controls.append(zoomOut, reset, zoomIn);
  wrap.replaceChildren(viewport, controls);
  installDiagramViewport(svg, viewport, {
    zoomIn,
    zoomOut,
    reset,
  });
}

function installDiagramViewport(svg, viewport, controls) {
  const base = parseViewBox(svg.getAttribute("viewBox"));
  if (!base) return;
  const state = { ...base };
  const minScale = 0.35;
  const maxScale = 4;

  function apply() {
    svg.setAttribute("viewBox", `${state.x} ${state.y} ${state.w} ${state.h}`);
  }

  function zoom(factor, center = { x: state.x + state.w / 2, y: state.y + state.h / 2 }) {
    const scale = base.w / state.w;
    const nextScale = Math.min(maxScale, Math.max(minScale, scale * factor));
    const nextW = base.w / nextScale;
    const nextH = base.h / nextScale;
    const rx = (center.x - state.x) / state.w;
    const ry = (center.y - state.y) / state.h;
    state.x = center.x - nextW * rx;
    state.y = center.y - nextH * ry;
    state.w = nextW;
    state.h = nextH;
    apply();
  }

  controls.zoomIn.addEventListener("click", () => zoom(1.25));
  controls.zoomOut.addEventListener("click", () => zoom(0.8));
  controls.reset.addEventListener("click", () => {
    Object.assign(state, base);
    apply();
  });

  viewport.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const point = svgPointForEvent(svg, event, state);
      zoom(event.deltaY < 0 ? 1.12 : 0.88, point);
    },
    { passive: false }
  );

  let drag = null;
  viewport.addEventListener("pointerdown", (event) => {
    if (isDiagramControlTarget(event.target)) return;
    drag = { x: event.clientX, y: event.clientY, view: { ...state } };
    viewport.classList.add("dragging");
    viewport.setPointerCapture(event.pointerId);
  });
  viewport.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const rect = svg.getBoundingClientRect();
    const dx = ((event.clientX - drag.x) / rect.width) * drag.view.w;
    const dy = ((event.clientY - drag.y) / rect.height) * drag.view.h;
    state.x = drag.view.x - dx;
    state.y = drag.view.y - dy;
    apply();
  });
  viewport.addEventListener("pointerup", () => {
    drag = null;
    viewport.classList.remove("dragging");
  });
  viewport.addEventListener("pointercancel", () => {
    drag = null;
    viewport.classList.remove("dragging");
  });
}

function isDiagramControlTarget(target) {
  return Boolean(
    target.closest?.(
      "button, .diagram-node, .diagram-group, .diagram-annotation, .diagram-hotspot"
    )
  );
}

function parseViewBox(value) {
  const parts = String(value || "").split(/\s+/).map(Number);
  if (parts.length !== 4 || parts.some((part) => !Number.isFinite(part))) return null;
  return { x: parts[0], y: parts[1], w: parts[2], h: parts[3] };
}

function svgPointForEvent(svg, event, viewBox) {
  const rect = svg.getBoundingClientRect();
  return {
    x: viewBox.x + ((event.clientX - rect.left) / rect.width) * viewBox.w,
    y: viewBox.y + ((event.clientY - rect.top) / rect.height) * viewBox.h,
  };
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
    node.addEventListener("click", () => {
      if (window.currentModel) activateTarget(window.currentModel, node.dataset.targetId);
    });
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        if (window.currentModel) activateTarget(window.currentModel, node.dataset.targetId);
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
    if (start) centerCodeLine(start);
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

function centerCodeLine(line) {
  const codeBlock = line.closest(".code-block");
  if (!codeBlock) {
    line.scrollIntoView({ block: "center", behavior: "smooth" });
    return;
  }

  const codeRect = codeBlock.getBoundingClientRect();
  const lineRect = line.getBoundingClientRect();
  const lineCenter = lineRect.top - codeRect.top + codeBlock.scrollTop + lineRect.height / 2;
  const targetTop = lineCenter - codeBlock.clientHeight / 2;
  codeBlock.scrollTo({
    top: Math.max(0, targetTop),
    behavior: "auto",
  });

  const updatedCodeRect = codeBlock.getBoundingClientRect();
  if (updatedCodeRect.top < 0 || updatedCodeRect.bottom > window.innerHeight) {
    codeBlock.scrollIntoView({ block: "nearest", behavior: "auto" });
  }
}

function initCompare() {
  const params = new URLSearchParams(window.location.search);
  fetchJson("data/index.json")
    .then((index) => {
      DATA_VERSION = index.data_version || "";
      const first = index.models[0]?.slug;
      const second = index.models[1]?.slug || first;
      const known = new Set(index.models.map((model) => model.slug));
      const leftSlug = known.has(params.get("left")) ? params.get("left") : first;
      const rightSlug = known.has(params.get("right")) ? params.get("right") : second;
      populateCompareSelect($("#leftModelSelect"), index.models, leftSlug, "left", rightSlug);
      populateCompareSelect($("#rightModelSelect"), index.models, rightSlug, "right", leftSlug);
      return Promise.all([
        fetchJson(modelDataPath(leftSlug, index.data_version)),
        fetchJson(modelDataPath(rightSlug, index.data_version)),
      ]).then(([left, right]) => ({ left, right }));
    })
    .then(({ left, right }) => renderCompare(left, right))
    .catch((error) => {
      $(".compare-shell").replaceChildren(errorBox(error));
    });
}

function populateCompareSelect(select, models, activeSlug, side, otherSlug) {
  if (!select) return;
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
    const params = new URLSearchParams(window.location.search);
    params.set(side, select.value);
    params.set(side === "left" ? "right" : "left", otherSlug);
    window.location.href = `compare.html?${params.toString()}`;
  });
}

function renderCompare(left, right) {
  window.currentModel = null;
  document.title = `${left.name} vs ${right.name} · compare`;
  $("#leftModelName").textContent = left.name;
  $("#rightModelName").textContent = right.name;
  $("#leftViewerLink").href = modelUrl(left.slug);
  $("#rightViewerLink").href = modelUrl(right.slug);
  $("#leftModelFacts").replaceChildren(...modelFactNodes(left));
  $("#rightModelFacts").replaceChildren(...modelFactNodes(right));
  renderDiagram(left, $("#leftDiagramWrap"));
  renderDiagram(right, $("#rightDiagramWrap"));
  renderConfigDiff(left, right);
}

function renderConfigDiff(left, right) {
  const table = $("#configDiffTable");
  const keys = Array.from(new Set([...Object.keys(left.config || {}), ...Object.keys(right.config || {})])).sort();
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const text of ["Field", left.slug, right.slug]) {
    headRow.append(el("th", "", text));
  }
  head.append(headRow);

  const body = document.createElement("tbody");
  for (const key of keys) {
    const leftValue = left.config?.[key];
    const rightValue = right.config?.[key];
    const same = JSON.stringify(leftValue) === JSON.stringify(rightValue);
    const row = document.createElement("tr");
    row.className = same ? "same" : "diff";
    row.append(el("td", "", key), el("td", "", formatValue(leftValue)), el("td", "", formatValue(rightValue)));
    body.append(row);
  }
  table.replaceChildren(head, body);
}

const page = document.body.dataset.page;
if (page === "index") initIndex();
if (page === "viewer") initViewer();
if (page === "compare") initCompare();
