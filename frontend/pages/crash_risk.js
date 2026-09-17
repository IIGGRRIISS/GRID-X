console.log("Crash risk JS loaded");

const DRIVERS = ["VER", "NOR", "LEC", "HAM", "RUS", "PIA", "SAI", "ALO"];

const DEFAULT_PTS = {
    VER: 250,
    NOR: 220,
    LEC: 210,
    HAM: 190,
    RUS: 170,
    PIA: 150,
    SAI: 140,
    ALO: 130
};

let crashGaugeChart = null;
let scGaugeChart = null;

const NAV_FIELDS = ["circuit", "track_temp"];

// ============================================================
// ENTER KEY NAVIGATION
// ============================================================

document.addEventListener("keydown", function (e) {

    if (e.key !== "Enter") return;

    const active = document.activeElement;
    const id = active.id;

    const staticIdx = NAV_FIELDS.indexOf(id);

    if (staticIdx !== -1) {

        e.preventDefault();

        if (staticIdx < NAV_FIELDS.length - 1) {

            document
                .getElementById(NAV_FIELDS[staticIdx + 1])
                .focus();

        } else {

            document
                .getElementById("grid_0")
                .focus();
        }

        return;
    }

    const gridSelectMatch = id.match(/^grid_(\d+)$/);

    if (gridSelectMatch) {

        e.preventDefault();

        const i = parseInt(gridSelectMatch[1]);

        document
            .getElementById(`pts_${i}`)
            .focus();

        return;
    }

    const gridPtsMatch = id.match(/^pts_(\d+)$/);

    if (gridPtsMatch) {

        e.preventDefault();

        const i = parseInt(gridPtsMatch[1]);

        if (i < DRIVERS.length - 1) {

            document
                .getElementById(`grid_${i + 1}`)
                .focus();

        } else {

            predictCrashRisk();
        }
    }
});


// ============================================================
// BUILD GRID LIST
// ============================================================

function buildGridList() {

    const list = document.getElementById("grid-list");

    list.innerHTML = "";

    DRIVERS.forEach((drv, i) => {

        const row = document.createElement("div");

        row.className = "grid-row";

        row.innerHTML = `
            <span class="grid-pos">P${i + 1}</span>

            <select
                id="grid_${i}"
                onchange="onDriverChange(${i})"
            >
                ${DRIVERS.map(
                    d =>
                        `<option value="${d}" ${d === drv ? "selected" : ""}>
                            ${d}
                        </option>`
                ).join("")}
            </select>

            <span class="grid-pts-label">PTS</span>

            <input
                type="number"
                id="pts_${i}"
                class="air-input"
                min="0"
                max="500"
                value="${DEFAULT_PTS[drv]}"
            >
        `;

        list.appendChild(row);
    });

    refreshAllDropdowns();
}


// ============================================================
// DRIVER CHANGE
// ============================================================

function onDriverChange(i) {

    const drv =
        document.getElementById(`grid_${i}`).value;

    document.getElementById(`pts_${i}`).value =
        DEFAULT_PTS[drv] || 0;

    refreshAllDropdowns();
}


// ============================================================
// REFRESH DRIVER DROPDOWNS
// ============================================================

function refreshAllDropdowns() {

    const selected = DRIVERS.map(
        (_, i) =>
            document.getElementById(`grid_${i}`)?.value
    );

    DRIVERS.forEach((_, i) => {

        const sel =
            document.getElementById(`grid_${i}`);

        if (!sel) return;

        Array.from(sel.options).forEach(opt => {

            const takenElsewhere =
                selected.some(
                    (s, j) =>
                        j !== i &&
                        s === opt.value
                );

            opt.disabled = takenElsewhere;

            opt.style.color =
                takenElsewhere ? "#444" : "#fff";
        });
    });
}

buildGridList();


// ============================================================
// PERCENTAGE COLOR
// ============================================================

function getPctColor(pct) {

    if (pct < 15) return "#00ff87";

    if (pct < 30) return "#ffcc00";

    return "#ff2a2a";
}


// ============================================================
// GAUGE
// ============================================================

function makeGauge(
    canvasId,
    value,
    color,
    existingChart
) {

    if (existingChart) {
        existingChart.destroy();
    }

    const ctx =
        document
            .getElementById(canvasId)
            .getContext("2d");

    return new Chart(ctx, {

        type: "doughnut",

        data: {

            datasets: [{

                data: [
                    value,
                    100 - value
                ],

                backgroundColor: [
                    color,
                    "rgba(255,255,255,0.05)"
                ],

                borderWidth: 0,

                circumference: 180,

                rotation: 270
            }]
        },

        options: {

            responsive: true,

            cutout: "75%",

            plugins: {

                legend: {
                    display: false
                },

                tooltip: {
                    enabled: false
                }
            }
        }
    });
}


// ============================================================
// PREDICT CRASH RISK
// ============================================================

function predictCrashRisk() {

    console.log("Predict function triggered");

    // --------------------------------------------------------
    // LOADING ELEMENTS
    // --------------------------------------------------------

    const loadingMsg =
        document.getElementById("loading-msg");

    const predictBtn =
        document.querySelector(".check-btn");

    const resultsSection =
        document.getElementById("results-section");

    const errorMsg =
        document.getElementById("error-msg");


    // --------------------------------------------------------
    // SHOW LOADING SPINNER
    // --------------------------------------------------------

    loadingMsg.innerHTML = `
        <div class="loading-state">
            <span class="loading-spinner"></span>
            <span>Predicting Crash Risk</span>
        </div>
    `;

    loadingMsg.style.display = "flex";

    predictBtn.disabled = true;


    // --------------------------------------------------------
    // HIDE PREVIOUS RESULTS
    // --------------------------------------------------------

    resultsSection.style.display = "none";

    errorMsg.style.display = "none";


    // --------------------------------------------------------
    // BUILD GRID DATA
    // --------------------------------------------------------

    const grid_positions = [];

    const championship_standings = {};

    for (let i = 0; i < DRIVERS.length; i++) {

        const drv =
            document.getElementById(`grid_${i}`).value;

        const pts =
            parseFloat(
                document
                    .getElementById(`pts_${i}`)
                    .value || "0"
            );

        grid_positions.push(drv);

        championship_standings[drv] = pts;
    }


    // --------------------------------------------------------
    // BUILD API PAYLOAD
    // --------------------------------------------------------

    const payload = {

        circuit:
            document.getElementById("circuit").value,

        weather_wet:
            document.getElementById("weather_wet").checked,

        track_temp:
            parseFloat(
                document.getElementById("track_temp").value || "25"
            ),

        grid_positions,

        championship_standings
    };


    console.log("Payload:", payload);


    // --------------------------------------------------------
    // MINIMUM LOADING TIME
    // --------------------------------------------------------

    const minDelay =
        new Promise(resolve =>
            setTimeout(resolve, 1500)
        );


    // --------------------------------------------------------
    // API REQUEST
    // --------------------------------------------------------

    Promise.all([

        fetch(
            "http://localhost:8000/crash-risk-predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(payload)
            }
        ).then(response => {

            if (!response.ok) {
                throw new Error(
                    `HTTP error: ${response.status}`
                );
            }

            return response.json();
        }),

        minDelay

    ])


    // --------------------------------------------------------
    // SUCCESS
    // --------------------------------------------------------

    .then(([data]) => {

        console.log("API response:", data);


        // Hide loader
        loadingMsg.style.display = "none";
        loadingMsg.innerHTML = "";

        predictBtn.disabled = false;


        if (!data) {

            errorMsg.style.display = "block";

            return;
        }


        // ----------------------------------------------------
        // CRASH + SAFETY CAR %
        // ----------------------------------------------------

        const crashPct =
            Math.round(
                (data.crash_probability || 0) * 100
            );

        const scPct =
            Math.round(
                (data.safety_car_probability || 0) * 100
            );


        const crashColor =
            getPctColor(crashPct);

        const scColor =
            getPctColor(scPct);


        // ----------------------------------------------------
        // UPDATE PERCENTAGES
        // ----------------------------------------------------

        document.getElementById(
            "crash-pct"
        ).textContent =
            crashPct + "%";

        document.getElementById(
            "crash-pct"
        ).style.color =
            crashColor;


        document.getElementById(
            "sc-pct"
        ).textContent =
            scPct + "%";

        document.getElementById(
            "sc-pct"
        ).style.color =
            scColor;


        // ----------------------------------------------------
        // UPDATE GAUGES
        // ----------------------------------------------------

        crashGaugeChart =
            makeGauge(
                "crashGaugeChart",
                crashPct,
                crashColor,
                crashGaugeChart
            );


        scGaugeChart =
            makeGauge(
                "scGaugeChart",
                scPct,
                scColor,
                scGaugeChart
            );


        // ----------------------------------------------------
        // RISK FACTORS
        // ----------------------------------------------------

        const factorsList =
            document.getElementById("factors-list");

        factorsList.innerHTML = "";


        (data.risk_factors || []).forEach(item => {

            const isPositive =
                item.contribution &&
                item.contribution.startsWith("+");


            const el =
                document.createElement("div");

            el.className = "factor-item";


            el.innerHTML = `

                <div class="factor-left">

                    <span class="factor-name">
                        ${item.factor.replace(/\_/g, " ")}
                    </span>

                    <span class="factor-explanation">
                        ${item.explanation || ""}
                    </span>

                </div>

                <span
                    class="factor-contribution"
                    style="color:${isPositive
                        ? "#ff2a2a"
                        : "#0096ff"}"
                >
                    ${item.contribution}
                </span>
            `;


            factorsList.appendChild(el);
        });


        // ----------------------------------------------------
        // RECOMMENDATIONS
        // ----------------------------------------------------

        const recsList =
            document.getElementById("recs-list");

        recsList.innerHTML = "";


        (data.recommendations || []).forEach(rec => {

            const el =
                document.createElement("div");

            el.className = "rec-item";

            el.textContent = rec;

            recsList.appendChild(el);
        });


        // ----------------------------------------------------
        // SHOW RESULTS
        // ----------------------------------------------------

        resultsSection.style.display = "block";

        resultsSection.scrollIntoView({
            behavior: "smooth"
        });
    })


    // --------------------------------------------------------
    // ERROR
    // --------------------------------------------------------

    .catch(err => {

        console.log("Fetch error:", err);


        // Hide loader
        loadingMsg.style.display = "none";
        loadingMsg.innerHTML = "";

        predictBtn.disabled = false;


        // Show error
        errorMsg.style.display = "block";
    });
}