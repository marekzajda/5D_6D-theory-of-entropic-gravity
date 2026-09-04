(() => {
  const supported = [
    ["en", "English"],
    ["cs", "Čeština"],
    ["de", "Deutsch"],
    ["fr", "Français"],
    ["es", "Español"]
  ];
  const projectPath = "/5D_6D-theory-of-entropic-gravity/";
  const raw = window.location.pathname.startsWith(projectPath)
    ? window.location.pathname.slice(projectPath.length)
    : window.location.pathname.replace(/^\//, "");
  const parts = raw.split("/").filter(Boolean);
  let currentLang = "en";
  let page = "index.html";
  if (parts.length && ["cs", "de", "fr", "es"].includes(parts[0])) {
    currentLang = parts[0];
    page = parts[1] || "index.html";
  } else if (parts.length) {
    page = parts[0];
  }
  const targetFor = (lang) => {
    if (lang === "en") return projectPath + (page === "index.html" ? "" : page);
    return projectPath + lang + "/" + page;
  };
  const nav = document.querySelector(".site-header .nav");
  if (!nav) return;
  const wrap = document.createElement("div");
  wrap.className = "language-switcher";
  const label = document.createElement("label");
  label.setAttribute("for", "closure-lang");
  label.textContent = "Language";
  const select = document.createElement("select");
  select.id = "closure-lang";
  select.setAttribute("aria-label", "Language");
  for (const [code, name] of supported) {
    const option = document.createElement("option");
    option.value = code;
    option.textContent = name;
    option.selected = code === currentLang;
    select.appendChild(option);
  }
  select.addEventListener("change", () => {
    window.location.href = targetFor(select.value);
  });
  wrap.append(label, select);
  nav.appendChild(wrap);
})();
