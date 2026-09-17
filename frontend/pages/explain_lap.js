let shapChart = null;
let radarCurrentChart = null;
let radarLegendChart = null;

// ============================================================
// INPUT RANGE CLAMPING
// ============================================================

const inputRanges = {
    air_temp: { min: 5, max: 45 },
    track_temp: { min: 5, max: 60 },
    humidity: { min: 0, max: 100 },
    rainfall: { min: 0, max: 50 },
    lap_number: { min: 1, max: 100 },
    stint: { min: 1, max: 5 },
    tyre_age: { min: 1, max: 50 },
    total_laps: { min: 10, max: 100 }
};

document.addEventListener("DOMContentLoaded", () => {
    Object.entries(inputRanges).forEach(([id, { min, max }]) => {
        const el = document.getElementById(id);

        if (!el) return;

        el.addEventListener("change", () => {
            const val = parseFloat(el.value);

            if (val < min) el.value = min;
            if (val > max) el.value = max;
        });
    });
});


// ============================================================
// ENTER KEY NAVIGATION
// ============================================================

const formFields = [
    "driver",
    "circuit",
    "compound",
    "air_temp",
    "track_temp",
    "humidity",
    "rainfall",
    "lap_number",
    "stint",
    "tyre_age",
    "total_laps"
];

document.addEventListener("keydown", function (e) {
    if (e.key !== "Enter") return;

    const active = document.activeElement;
    const currentIndex = formFields.indexOf(active.id);

    if (currentIndex === -1) return;

    e.preventDefault();

    if (currentIndex < formFields.length - 1) {
        document
            .getElementById(formFields[currentIndex + 1])
            .focus();
    } else {
        explainLap();
    }
});


// ============================================================
// LEGACY DRIVER DATA
// ============================================================

const legacyData = {

    VER: {
        current: {
            code: "VER",
            name: "Max Verstappen",
            team: "Red Bull Racing · 2023–Present",
            img: "drivers_images/VER.png",
            stats: [95, 90, 88, 72, 93]
        },

        legend: {
            code: "SCH",
            name: "Michael Schumacher",
            team: "Ferrari · 1996–2006",
            img: "drivers_images/SCH.png",
            stats: [93, 88, 95, 70, 90]
        },

        analysis: [
            "Verstappen's raw aggression and instinctive overtaking mirror Schumacher's early Ferrari dominance — both drivers turn average machinery into title-winning weapons through sheer force of will and an unrelenting race pace.",

            "Where Schumacher's tyre management evolved into a tactical masterclass, Verstappen still pushes harder on rubber — while his braking precision and corner-entry aggression make him extremely dangerous on a single flying lap."
        ]
    },

    NOR: {
        current: {
            code: "NOR",
            name: "Lando Norris",
            team: "McLaren · 2019–Present",
            img: "drivers_images/NOR.png",
            stats: [82, 88, 85, 80, 84]
        },

        legend: {
            code: "SEN",
            name: "Ayrton Senna",
            team: "McLaren · 1988–1993",
            img: "drivers_images/SEN.png",
            stats: [96, 95, 80, 68, 97]
        },

        analysis: [
            "Norris shares Senna's gift for wet-weather brilliance and an almost supernatural feel for the limit of grip — both are at their most dangerous when conditions punish the meek.",

            "Senna's aggression and overtaking were on another level; Norris is more measured and scores higher on tyre preservation and consistency."
        ]
    },

    LEC: {
        current: {
            code: "LEC",
            name: "Charles Leclerc",
            team: "Ferrari · 2019–Present",
            img: "drivers_images/LEC.png",
            stats: [88, 85, 80, 75, 86]
        },

        legend: {
            code: "GVI",
            name: "Gilles Villeneuve",
            team: "Ferrari · 1977–1982",
            img: "drivers_images/GVI.jpeg",
            stats: [97, 82, 70, 60, 94]
        },

        analysis: [
            "Leclerc and Villeneuve share an emotional connection to Ferrari and a qualifying pace that makes crowds hold their breath — both have produced spectacular pole laps.",

            "Villeneuve's aggression was unmatched, often at the cost of reliability; Leclerc has learned to channel that same fire more strategically."
        ]
    },

    HAM: {
        current: {
            code: "HAM",
            name: "Lewis Hamilton",
            team: "Mercedes · 2013–2024",
            img: "drivers_images/HAM.png",
            stats: [85, 92, 97, 95, 88]
        },

        legend: {
            code: "PRO",
            name: "Alain Prost",
            team: "McLaren / Williams · 1984–1993",
            img: "drivers_images/PRO.png",
            stats: [78, 90, 96, 97, 82]
        },

        analysis: [
            "Hamilton and Prost are calculating racers who built their championships on tyre management, consistency and the ability to conserve equipment for when it matters.",

            "Hamilton's overtaking aggression is a notch above Prost's, while Prost edges Hamilton slightly on pure technical tyre preservation."
        ]
    },

    RUS: {
        current: {
            code: "RUS",
            name: "George Russell",
            team: "Mercedes · 2022–Present",
            img: "drivers_images/RUS.png",
            stats: [80, 87, 89, 83, 78]
        },

        legend: {
            code: "NRO",
            name: "Nico Rosberg",
            team: "Mercedes · 2010–2016",
            img: "drivers_images/NRO.png",
            stats: [78, 85, 88, 86, 76]
        },

        analysis: [
            "Russell and Rosberg are cerebral, methodical Mercedes drivers who rely on technical feedback, preparation and surgical braking precision.",

            "Rosberg peaked tactically in 2016; Russell is still developing that same mental edge while sharing Rosberg's consistency and reliability."
        ]
    },

    PIA: {
        current: {
            code: "PIA",
            name: "Oscar Piastri",
            team: "McLaren · 2023–Present",
            img: "drivers_images/PIA.png",
            stats: [78, 84, 88, 86, 80]
        },

        legend: {
            code: "HAK",
            name: "Mika Häkkinen",
            team: "McLaren · 1993–2001",
            img: "drivers_images/HAK.png",
            stats: [80, 88, 90, 84, 85]
        },

        analysis: [
            "Piastri's cool demeanour in the cockpit echoes Häkkinen's Finnish composure — both arrived at McLaren with a quiet intensity that hid serious pace.",

            "Häkkinen's overtaking was devastatingly clinical; Piastri is building the same reputation for clean, decisive moves."
        ]
    },

    SAI: {
        current: {
            code: "SAI",
            name: "Carlos Sainz",
            team: "Ferrari · 2021–2024",
            img: "drivers_images/SAI.png",
            stats: [79, 83, 90, 88, 77]
        },

        legend: {
            code: "REU",
            name: "Carlos Reutemann",
            team: "Williams / Ferrari · 1972–1982",
            img: "drivers_images/REU.png",
            stats: [82, 80, 86, 83, 80]
        },

        analysis: [
            "Both Carloses share an intelligent, smooth driving style that maximises tyre life and builds race pace strategically.",

            "Reutemann's overtaking edge came from opportunism; Sainz has adapted those instincts to modern Formula 1."
        ]
    },

    ALO: {
        current: {
            code: "ALO",
            name: "Fernando Alonso",
            team: "Aston Martin · 2023–Present",
            img: "drivers_images/ALO.png",
            stats: [90, 91, 93, 89, 92]
        },

        legend: {
            code: "STW",
            name: "Jackie Stewart",
            team: "Tyrrell · 1968–1973",
            img: "drivers_images/STW.png",
            stats: [84, 88, 94, 90, 85]
        },

        analysis: [
            "Alonso and Stewart are complete racing drivers who could win in the rain, on street circuits and in underpowered machinery through mastery of every variable.",

            "Alonso's overtaking aggression and braking precision score higher, while Stewart's tyre preservation relative to his era remains remarkable."
        ]
    }
};


// ============================================================
// RADAR CHART
// ============================================================

function buildRadar(canvasId, stats, color, existingChart) {

    if (existingChart) {
        existingChart.destroy();
    }

    const canvas = document.getElementById(canvasId);

    if (!canvas) return null;

    const ctx = canvas.getContext("2d");

    return new Chart(ctx, {

        type: "radar",

        data: {

            labels: [
                "Aggression",
                "Braking",
                "Consistency",
                "Tyre Preservation",
                "Overtaking"
            ],

            datasets: [{
                data: stats,

                backgroundColor:
                    color === "red"
                        ? "rgba(255, 42, 42, 0.25)"
                        : "rgba(0, 150, 255, 0.2)",

                borderColor:
                    color === "red"
                        ? "#ff2a2a"
                        : "#0096ff",

                borderWidth: 2,

                pointBackgroundColor:
                    color === "red"
                        ? "#ff2a2a"
                        : "#0096ff",

                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },

        options: {

            responsive: true,
            maintainAspectRatio: true,

            scales: {

                r: {

                    min: 0,
                    max: 100,

                    ticks: {
                        display: false,
                        stepSize: 20
                    },

                    grid: {
                        color: "rgba(255, 255, 255, 0.08)"
                    },

                    angleLines: {
                        color: "rgba(255, 255, 255, 0.1)"
                    },

                    pointLabels: {

                        color: "#aaaaaa",

                        font: {
                            family: "Rajdhani",
                            size: 15
                        }
                    }
                }
            },

            plugins: {

                legend: {
                    display: false
                },

                tooltip: {

                    callbacks: {

                        label: ctx =>
                            ` ${ctx.label}: ${ctx.raw}`
                    },

                    backgroundColor: "rgba(0,0,0,0.8)",
                    titleColor: "#fff",
                    bodyColor: "#ccc"
                }
            }
        }
    });
}


// ============================================================
// LEGACY COMPARISON
// ============================================================

function renderLegacyComparison(driverCode) {

    const data = legacyData[driverCode];

    if (!data) return;

    const {
        current,
        legend,
        analysis
    } = data;


    // Current driver

    document.getElementById(
        "legacy-current-img"
    ).src = current.img;

    document.getElementById(
        "legacy-current-img"
    ).alt = current.name;

    document.getElementById(
        "legacy-current-code"
    ).textContent = current.code;

    document.getElementById(
        "legacy-current-name"
    ).textContent = current.name;

    document.getElementById(
        "legacy-current-team"
    ).textContent = current.team;


    // Legend driver

    document.getElementById(
        "legacy-legend-img"
    ).src = legend.img;

    document.getElementById(
        "legacy-legend-img"
    ).alt = legend.name;

    document.getElementById(
        "legacy-legend-code"
    ).textContent = legend.code;

    document.getElementById(
        "legacy-legend-name"
    ).textContent = legend.name;

    document.getElementById(
        "legacy-legend-team"
    ).textContent = legend.team;


    // Analysis

    document.getElementById(
        "analysis-text-1"
    ).textContent = analysis[0];

    document.getElementById(
        "analysis-text-2"
    ).textContent = analysis[1];


    // Radar charts

    radarCurrentChart = buildRadar(
        "radarCurrent",
        current.stats,
        "red",
        radarCurrentChart
    );

    radarLegendChart = buildRadar(
        "radarLegend",
        legend.stats,
        "blue",
        radarLegendChart
    );


    // Show section

    const legacySection =
        document.getElementById("legacy-section");

    legacySection.style.display = "block";

    legacySection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


// ============================================================
// MAIN EXPLAIN LAP FUNCTION
// ============================================================

function explainLap() {

    const loadingMsg =
        document.getElementById("loading-msg");

    const explainBtn =
        document.getElementById("explainBtn");

    const errorMsg =
        document.getElementById("error-msg");


    // --------------------------------------------------------
    // SHOW LOADING
    // --------------------------------------------------------

    loadingMsg.innerHTML = `
        <div class="loading-state">
            <span class="loading-spinner"></span>
            <span>Explaining Lap</span>
        </div>
    `;

    loadingMsg.style.display = "flex";

    explainBtn.disabled = true;

    errorMsg.style.display = "none";


    // Hide old results

    document.getElementById(
        "results-section"
    ).style.display = "none";

    document.getElementById(
        "legacy-section"
    ).style.display = "none";


    // --------------------------------------------------------
    // CLAMP FUNCTION
    // --------------------------------------------------------

    function clamp(value, min, max) {

        return Math.min(
            max,
            Math.max(min, value)
        );
    }


    // --------------------------------------------------------
    // INPUT VALUES
    // --------------------------------------------------------

    const lapNumber = clamp(
        parseInt(
            document.getElementById("lap_number").value || "10"
        ),
        1,
        100
    );


    const totalLaps = clamp(
        parseInt(
            document.getElementById("total_laps").value || "52"
        ),
        10,
        100
    );


    const sessionProgress =
        parseFloat(
            (lapNumber / totalLaps).toFixed(3)
        );


    const selectedDriver =
        document.getElementById("driver").value;


    // --------------------------------------------------------
    // REQUEST DATA
    // --------------------------------------------------------

    const values = {

        driver: selectedDriver,

        circuit:
            document.getElementById("circuit").value,

        compound:
            document.getElementById("compound").value,

        weather: {

            air_temp: clamp(
                parseFloat(
                    document.getElementById("air_temp").value || "18"
                ),
                5,
                45
            ),

            track_temp: clamp(
                parseFloat(
                    document.getElementById("track_temp").value || "25"
                ),
                5,
                60
            ),

            humidity: clamp(
                parseFloat(
                    document.getElementById("humidity").value || "65"
                ),
                0,
                100
            ),

            rainfall: clamp(
                parseFloat(
                    document.getElementById("rainfall").value || "0"
                ),
                0,
                50
            )
        },

        lap_number: lapNumber,

        stint: clamp(
            parseInt(
                document.getElementById("stint").value || "1"
            ),
            1,
            5
        ),

        tyre_age: clamp(
            parseInt(
                document.getElementById("tyre_age").value || "10"
            ),
            1,
            50
        ),

        session_progress: sessionProgress,

        total_laps: totalLaps
    };


    console.log("Sending GRID-X request:", values);


    // --------------------------------------------------------
    // BACKEND REQUEST
    // --------------------------------------------------------

    const minDelay = new Promise(resolve => {
        setTimeout(resolve, 1500);
    });


    const apiRequest = fetch(
        "http://localhost:8000/explain-lap",
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(values)
        }
    )
    .then(response => {

        if (!response.ok) {
            throw new Error(
                `Server returned ${response.status}`
            );
        }

        return response.json();
    });


    Promise.all([
        apiRequest,
        minDelay
    ])

    .then(([data]) => {

        console.log("GRID-X API Response:", data);


        // ----------------------------------------------------
        // HIDE LOADING
        // ----------------------------------------------------

        loadingMsg.style.display = "none";

        loadingMsg.innerHTML = "";

        explainBtn.disabled = false;


        // ----------------------------------------------------
        // API FAILURE
        // ----------------------------------------------------

        if (!data.success) {

            console.error(
                "API returned success: false",
                data
            );

            errorMsg.style.display = "block";

            return;
        }


        // ----------------------------------------------------
        // SHAP RESULT
        // ----------------------------------------------------

        /*
         * Keep your existing SHAP rendering code here.
         *
         * The important part is:
         *
         * 1. Backend request works
         * 2. Spinner appears while waiting
         * 3. Spinner disappears after response
         * 4. Button becomes active again
         */


        document.getElementById(
            "results-section"
        ).style.display = "block";


        // If your backend returns a driver code,
        // render the legacy comparison.

        renderLegacyComparison(
            selectedDriver
        );

    })

    .catch(error => {

        console.error(
            "GRID-X API Error:",
            error
        );


        // ----------------------------------------------------
        // HIDE LOADING
        // ----------------------------------------------------

        loadingMsg.style.display = "none";

        loadingMsg.innerHTML = "";

        explainBtn.disabled = false;


        // ----------------------------------------------------
        // SHOW ERROR
        // ----------------------------------------------------

        errorMsg.style.display = "block";
    });
}


// ============================================================
// BUTTON
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    () => {

        const explainBtn =
            document.getElementById("explainBtn");

        if (!explainBtn) return;

        explainBtn.addEventListener(
            "click",
            explainLap
        );
    }
);