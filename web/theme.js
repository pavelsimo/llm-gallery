"use strict";

(function initializeTheme() {
  const STORAGE_KEY = "llm-gallery-theme";
  const media = window.matchMedia("(prefers-color-scheme: dark)");

  function storedTheme() {
    try {
      const value = window.localStorage.getItem(STORAGE_KEY);
      return value === "light" || value === "dark" ? value : null;
    } catch {
      return null;
    }
  }

  function systemTheme() {
    return media.matches ? "dark" : "light";
  }

  function apply(theme) {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    window.dispatchEvent(new CustomEvent("llm-gallery-theme-change", { detail: { theme } }));
    return theme;
  }

  function save(theme) {
    try {
      window.localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      // Theme switching still works when storage is unavailable.
    }
  }

  function set(theme, { persist = false } = {}) {
    if (theme !== "light" && theme !== "dark") return current();
    if (persist) save(theme);
    return apply(theme);
  }

  function current() {
    return document.documentElement.dataset.theme || systemTheme();
  }

  function toggle() {
    return set(current() === "dark" ? "light" : "dark", { persist: true });
  }

  const initialPreference = storedTheme();
  apply(initialPreference || systemTheme());

  media.addEventListener?.("change", () => {
    if (!storedTheme()) apply(systemTheme());
  });

  window.LlmGalleryTheme = { current, set, storedTheme, toggle };
})();
