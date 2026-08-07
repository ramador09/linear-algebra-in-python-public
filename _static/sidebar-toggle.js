// Make the "Toggle primary sidebar" button work. As shipped it does nothing at
// any width, and on a phone that leaves the site with no navigation at all.
//
// Cause: sphinx-book-theme 1.2.0 and pydata-sphinx-theme 0.16.1 disagree about
// the mechanism. The page renders TWO toggles — one in the navbar (which
// sphinx-book-theme hides with `.bd-header button.sidebar-toggle{display:none}`)
// and one in the article toolbar (the visible one). The theme's own handler does
// not drive the visible button, so no class is ever applied. Separately,
// sphinx-book-theme only gives pydata's `pst-sidebar-hidden` class an effect
// inside `@media (min-width: 992px)`; its narrow-screen rules still target
// `input#pst-primary-sidebar-checkbox`, an element pydata 0.16 no longer emits.
//
// So we drive the toggle ourselves:
// So we drive the toggle ourselves, from a body class in both regimes:
//   >= 992px  `body.ecp-nav-collapsed` folds the sidebar out of the flow.
//   <  992px  `body.ecp-nav-open` slides an off-canvas drawer in, with a
//             dismiss backdrop. Both are styled in _static/ecp.css.
(function () {
  var BREAKPOINT = 992;
  var OPEN = "ecp-nav-open";
  var COLLAPSED = "ecp-nav-collapsed";
  var backdrop = null;

  function isNarrow() {
    return window.innerWidth < BREAKPOINT;
  }

  function sidebar() {
    return document.querySelector(".bd-sidebar-primary");
  }

  function toggles() {
    return Array.prototype.slice.call(
      document.querySelectorAll(".primary-toggle")
    );
  }

  function setExpanded(state) {
    toggles().forEach(function (b) {
      b.setAttribute("aria-expanded", state ? "true" : "false");
    });
  }

  function removeBackdrop() {
    if (backdrop) {
      backdrop.remove();
      backdrop = null;
    }
  }

  function closeDrawer() {
    document.body.classList.remove(OPEN);
    removeBackdrop();
    setExpanded(false);
  }

  function openDrawer() {
    document.body.classList.add(OPEN);
    backdrop = document.createElement("button");
    backdrop.className = "ecp-nav-backdrop";
    backdrop.setAttribute("aria-label", "Close navigation");
    backdrop.addEventListener("click", closeDrawer);
    document.body.appendChild(backdrop);
    setExpanded(true);
  }

  function onToggle(ev) {
    var sb = sidebar();
    if (!sb) return;
    // The theme's own listener would otherwise also fire and fight us.
    ev.preventDefault();
    ev.stopImmediatePropagation();

    if (isNarrow()) {
      if (document.body.classList.contains(OPEN)) closeDrawer();
      else openDrawer();
      return;
    }
    // Wide: collapse it out of the flow so the article reflows to full width
    // (see the matching rule in ecp.css).
    var nowCollapsed = document.body.classList.toggle(COLLAPSED);
    setExpanded(!nowCollapsed);
  }

  function init() {
    var sb = sidebar();
    if (!sb) return;

    toggles().forEach(function (btn) {
      btn.addEventListener("click", onToggle, true);
    });

    // Following a link should not leave the drawer covering the new page.
    sb.addEventListener("click", function (ev) {
      if (isNarrow() && ev.target.closest("a")) closeDrawer();
    });

    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && document.body.classList.contains(OPEN)) {
        closeDrawer();
      }
    });

    // Crossing the breakpoint hands the sidebar back to its other regime;
    // leftover state would strand a backdrop over a desktop layout, or leave
    // the drawer collapsed with no way to reopen it.
    window.addEventListener("resize", function () {
      if (!isNarrow() && document.body.classList.contains(OPEN)) {
        closeDrawer();
      }
      if (isNarrow()) document.body.classList.remove(COLLAPSED);
    });

    setExpanded(!isNarrow() && !document.body.classList.contains(COLLAPSED));
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
