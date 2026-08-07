// Repoint the download menu's ".pdf" entry at the official typeset PDF
// edition instead of the theme's default window.print(), which just prints
// the current page through the browser.
//
// On a NOTEBOOK page the link serves that notebook's own excerpt, cut from
// the typeset edition by tools/split_pdf.py and published as a release asset
// named "<part>--<name>.pdf" (e.g. "03-eigenvalues--spectral-theorem.pdf").
// Everywhere else (chapter landing pages, front matter) it serves the whole
// book — which the Preface and the site footer also link.
(function () {
  var RELEASE =
    "https://github.com/ramador09/linear-algebra-in-python-public/releases/latest/download/";
  var FULL = RELEASE + "linear-algebra-in-python.pdf";

  function excerptUrl() {
    var m = window.location.pathname.match(/\/notebooks\/([^/]+)\/([^/]+)\.html$/);
    if (!m) return null;
    var part = m[1], name = m[2];
    if (name === "index" || part === "00-front-matter") return null;
    return RELEASE + part + "--" + name + ".pdf";
  }

  function repointPdfButton() {
    var excerpt = excerptUrl();
    document.querySelectorAll(".btn-download-pdf-button").forEach(function (btn) {
      if (btn.tagName === "A") return; // already repointed
      var a = document.createElement("a");
      a.href = excerpt || FULL;
      a.className = btn.className;
      a.title = excerpt
        ? "This notebook as PDF (from the typeset edition)"
        : "PDF edition (whole course, typeset)";
      a.setAttribute("data-bs-placement", "left");
      a.setAttribute("data-bs-toggle", "tooltip");
      a.innerHTML = btn.innerHTML;
      btn.replaceWith(a);
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", repointPdfButton);
  } else {
    repointPdfButton();
  }
})();
