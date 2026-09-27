(function () {
  "use strict";

  var STORAGE_KEY = "ga-consent";

  document.addEventListener("DOMContentLoaded", function () {
    var meta = document.querySelector('meta[name="ga-measurement-id"]');
    if (!meta) {
      return;
    }

    var stored = null;
    try {
      stored = window.localStorage.getItem(STORAGE_KEY);
    } catch (err) {
      stored = null;
    }

    if (stored === "granted") {
      window.gtag("consent", "update", { analytics_storage: "granted" });
      return;
    }
    if (stored === "denied") {
      window.gtag("consent", "update", { analytics_storage: "denied" });
      return;
    }

    showBanner();
  });

  function showBanner() {
    var banner = document.createElement("div");
    banner.className = "cc-banner";
    banner.setAttribute("role", "dialog");
    banner.setAttribute("aria-label", "Cookie consent");

    var text = document.createElement("p");
    text.className = "cc-banner__text";
    text.textContent =
      "We use Google Analytics to understand how this documentation is used. " +
      "Analytics cookies are only set if you accept. ";
    var link = document.createElement("a");
    link.className = "cc-banner__link";
    link.href = "privacy.html";
    link.textContent = "Privacy note";
    text.appendChild(link);

    var actions = document.createElement("div");
    actions.className = "cc-banner__actions";

    var accept = document.createElement("button");
    accept.type = "button";
    accept.className = "cc-banner__btn cc-banner__btn--accept";
    accept.textContent = "Accept";
    accept.addEventListener("click", function () {
      persist("granted");
      window.gtag("consent", "update", { analytics_storage: "granted" });
      hide(banner);
    });

    var decline = document.createElement("button");
    decline.type = "button";
    decline.className = "cc-banner__btn cc-banner__btn--decline";
    decline.textContent = "Decline";
    decline.addEventListener("click", function () {
      persist("denied");
      window.gtag("consent", "update", { analytics_storage: "denied" });
      hide(banner);
    });

    actions.appendChild(decline);
    actions.appendChild(accept);
    banner.appendChild(text);
    banner.appendChild(actions);
    document.body.appendChild(banner);
  }

  function persist(value) {
    try {
      window.localStorage.setItem(STORAGE_KEY, value);
    } catch (err) {
      /* storage unavailable; consent lasts for this page view only */
    }
  }

  function hide(banner) {
    banner.classList.add("cc-banner--hidden");
    banner.addEventListener("transitionend", function () {
      if (banner.parentNode) {
        banner.parentNode.removeChild(banner);
      }
    });
    /* Fallback in case transitionend never fires */
    window.setTimeout(function () {
      if (banner.parentNode) {
        banner.parentNode.removeChild(banner);
      }
    }, 500);
  }
})();
