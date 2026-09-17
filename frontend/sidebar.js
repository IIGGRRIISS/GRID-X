(function () {

    // Detect if we're inside /pages/ subfolder
    const inPages = window.location.pathname.includes('/pages/');
    const base = inPages ? '../' : '';

    const PAGES = [
    { label: "Home",               icon: "⬡", href: `${base}index.html`,                       match: "index" },
    { label: "Pitwall",        icon: "◈", href: `${base}pitwall.html`,                  match: "pitwall" },
    { label: "Chassis",        icon: "⬢", href: `${base}chassis.html`,                  match: "chassis" },
    { label: "Predict Race",       icon: "🏁", href: `${base}pages/predict_race.html`,          match: "predict_race" },
    { label: "Stint Simulator",    icon: "⏱", href: `${base}pages/stint_simulate.html`,        match: "stint_simulate" },
    { label: "Next Lap",           icon: "↻", href: `${base}pages/predict_next_lap.html`,       match: "predict_next_lap" },
    { label: "Strategy Optimizer", icon: "◎", href: `${base}pages/strategy_optimize.html`,     match: "strategy_optimize" },
    { label: "Circuit Analyzer",   icon: "◉", href: `${base}pages/analyze_circuit.html`,       match: "analyze_circuit" },
    { label: "Explain Lap",        icon: "◍", href: `${base}pages/explain_lap.html`,            match: "explain_lap" },
    { label: "Tire Safety",        icon: "◍", href: `${base}pages/tire_safety.html`,            match: "tire_safety" },
    { label: "Crash Risk",         icon: "⚠",  href: `${base}pages/crash_risk.html`,            match: "crash_risk" },
];

    const currentPath = window.location.pathname;

    const navLinks = PAGES.map(page => {
        const isActive = currentPath.includes(page.match);
        return `
            <a href="${page.href}" class="${isActive ? 'active' : ''}">
                <span class="gx-nav-icon">${page.icon}</span>
                ${page.label}
            </a>
        `;
    }).join('');

    const sidebarHTML = `
        <div id="gx-sidebar">
            <div class="gx-sidebar-logo">
                <span><em>GRID</em>-X</span>
            </div>
            <div class="gx-sidebar-label">Navigation</div>
            <nav class="gx-nav">
                ${navLinks}
            </nav>
            <div class="gx-sidebar-footer">
                <a href="${base}index.html">
                    <span>⬡</span> Back to Home
                </a>
            </div>
        </div>
        <div id="gx-sidebar-toggle" title="Toggle sidebar">❯</div>
    `;

    document.body.insertAdjacentHTML('afterbegin', sidebarHTML);

    const sidebar = document.getElementById('gx-sidebar');
    const toggle  = document.getElementById('gx-sidebar-toggle');
    let isOpen = localStorage.getItem('gx-sidebar') !== 'closed';

    function applyState() {
        if (isOpen) {
            sidebar.classList.remove('collapsed');
            toggle.style.left = '240px';
            toggle.textContent = '❮';
            document.body.classList.add('sidebar-open');
            document.body.classList.remove('sidebar-collapsed');
        } else {
            sidebar.classList.add('collapsed');
            toggle.style.left = '0px';
            toggle.textContent = '❯';
            document.body.classList.remove('sidebar-open');
            document.body.classList.add('sidebar-collapsed');
        }
    }

    toggle.addEventListener('click', () => {
        isOpen = !isOpen;
        localStorage.setItem('gx-sidebar', isOpen ? 'open' : 'closed');
        applyState();
    });

    applyState();

})();