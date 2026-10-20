
document.addEventListener("DOMContentLoaded", async () => {
    const footerHost = document.createElement("div");
    footerHost.id = "footer-host";
    document.body.appendChild(footerHost);

    const currentScript = document.querySelector(
        'script[src$="footer.js"]'
    );

    const basePath = currentScript
        ? new URL(".", currentScript.src)
        : new URL("../components/", window.location.href);

    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = new URL("footer.css", basePath).href;
    document.head.appendChild(css);

    try {
        const response = await fetch(
            new URL("footer.html", basePath)
        );

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        footerHost.innerHTML = await response.text();
    } catch (error) {
        console.error("Could not load GRID-X footer:", error);
    }
});
