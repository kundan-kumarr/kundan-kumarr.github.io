<script>
(function () {
  const root = document.documentElement;
  const STORAGE_KEY = "quarto-color-scheme";

  // 1) Find the navbar placeholder you added in _quarto.yml
  const slot = document.getElementById("theme-toggle");
  if (!slot) return; // if not on a page with navbar, do nothing

  // 2) Get the hidden template and clone it
  const tpl = document.getElementById("theme-toggle-template");
  if (!tpl) return;

  const btn = tpl.content.firstElementChild.cloneNode(true);

  // 3) Replace the placeholder with the real button
  slot.replaceWith(btn);

  // 4) Initial state: localStorage > system preference
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "dark") {
    root.classList.add("dark");
  } else if (stored === "light") {
    root.classList.remove("dark");
  } else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
    root.classList.add("dark");
  }

  // 5) Toggle click handler
  btn.addEventListener("click", function (e) {
    e.preventDefault();
    const isDark = root.classList.toggle("dark");
    localStorage.setItem(STORAGE_KEY, isDark ? "dark" : "light");
  });
})();
</script>
