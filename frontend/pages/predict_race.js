const driverNames = {
    VER: "Max Verstappen",
    NOR: "Lando Norris",
    LEC: "Charles Leclerc",
    HAM: "Lewis Hamilton",
    RUS: "George Russell",
    PIA: "Oscar Piastri",
    SAI: "Carlos Sainz",
    ALO: "Fernando Alonso"
};

// ===== INPUT RANGE CLAMPING =====
const inputRanges = {
    air_temp:   { min: 5,    max: 45 },
    track_temp: { min: 5,    max: 60 },
    humidity:   { min: 0,    max: 100 },
    rainfall:   { min: 0,    max: 50 },
    year:       { min: 2000, max: 2026 },
    round:      { min: 1,    max: 24 }
};

Object.entries(inputRanges).forEach(([id, { min, max }]) => {
    const el = document.getElementById(id);
    if (!el) return;

    el.addEventListener("change", () => {
        const val = parseFloat(el.value);

        if (val < min) el.value = min;
        if (val > max) el.value = max;
    });
});

// ===== ENTER KEY NAVIGATION =====
const NAV_ORDER = [
    "circuit", "year", "round", "air_temp", "track_temp", "humidity", "rainfall",
    "pos_ver", "tyre_ver",
    "pos_nor", "tyre_nor",
    "pos_lec", "tyre_lec",
    "pos_ham", "tyre_ham",
    "pos_rus", "tyre_rus",
    "pos_pia", "tyre_pia",
    "pos_sai", "tyre_sai",
    "pos_alo", "tyre_alo"
];

document.addEventListener("keydown", function (e) {
    if (e.key !== "Enter") return;

    const active = document.activeElement;
    const idx = NAV_ORDER.indexOf(active.id);

    if (idx === -1) return;

    e.preventDefault();

    if (idx < NAV_ORDER.length - 1) {
        document.getElementById(NAV_ORDER[idx + 1]).focus();
    } else {
        clickFunction();
    }
});

// ===== POSITION DROPDOWN LOCK =====
const positionSelects = [
    "pos_ver", "pos_nor", "pos_lec", "pos_ham",
    "pos_rus", "pos_pia", "pos_sai", "pos_alo"
];

function updatePositionDropdowns() {

    const selected = positionSelects.map(
        id => document.getElementById(id).value
    );

    positionSelects.forEach(id => {

        const select = document.getElementById(id);
        const currentVal = select.value;

        Array.from(select.options).forEach(option => {

            option.disabled =
                selected.includes(option.value) &&
                option.value !== currentVal;

        });

    });
}

positionSelects.forEach(id => {
    document.getElementById(id).addEventListener(
        "change",
        updatePositionDropdowns
    );
});

// ===== MAIN PREDICT FUNCTION =====
function clickFunction() {

    // ===== LOADING ELEMENTS =====
    const loadingMsg = document.getElementById("loading-msg");
    const button = document.querySelector(".predict-btn");

    // ===== SHOW LOADING STATE =====
    loadingMsg.innerHTML = `
        <div class="loading-state">
            <span class="loading-spinner"></span>
            <span>Predicting Race</span>
        </div>
    `;

    loadingMsg.style.display = "flex";
    button.disabled = true;

    // Hide previous results/errors
    document.getElementById("results-section").style.display = "none";
    document.getElementById("error-msg").style.display = "none";


    function clamp(val, min, max) {
        return Math.min(max, Math.max(min, val));
    }


    // ===== COLLECT INPUT VALUES =====
    const values = {

        year: clamp(
            parseInt(
                document.getElementById("year").value || "2024"
            ),
            2000,
            2026
        ),

        round: clamp(
            parseInt(
                document.getElementById("round").value || "1"
            ),
            1,
            24
        ),

        circuit:
            document.getElementById("circuit").value,

        conditions: {

            air_temp: clamp(
                parseFloat(
                    document.getElementById("air_temp").value || "20"
                ),
                5,
                45
            ),

            track_temp: clamp(
                parseFloat(
                    document.getElementById("track_temp").value || "30"
                ),
                5,
                60
            ),

            humidity: clamp(
                parseFloat(
                    document.getElementById("humidity").value || "60"
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

        qualifying_results: {

            VER: parseInt(
                document.getElementById("pos_ver").value
            ),

            NOR: parseInt(
                document.getElementById("pos_nor").value
            ),

            LEC: parseInt(
                document.getElementById("pos_lec").value
            ),

            HAM: parseInt(
                document.getElementById("pos_ham").value
            ),

            RUS: parseInt(
                document.getElementById("pos_rus").value
            ),

            PIA: parseInt(
                document.getElementById("pos_pia").value
            ),

            SAI: parseInt(
                document.getElementById("pos_sai").value
            ),

            ALO: parseInt(
                document.getElementById("pos_alo").value
            )
        },

        tyre_compounds: {

            VER: document.getElementById("tyre_ver").value,
            NOR: document.getElementById("tyre_nor").value,
            LEC: document.getElementById("tyre_lec").value,
            HAM: document.getElementById("tyre_ham").value,
            RUS: document.getElementById("tyre_rus").value,
            PIA: document.getElementById("tyre_pia").value,
            SAI: document.getElementById("tyre_sai").value,
            ALO: document.getElementById("tyre_alo").value
        }
    };


    // Keep loader visible for at least 1.5 seconds
    const minDelay = new Promise(resolve =>
        setTimeout(resolve, 1500)
    );


    // ===== API REQUEST =====
    Promise.all([

        fetch("http://localhost:8000/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(values)

        }).then(response => {

            if (!response.ok) {
                throw new Error(
                    `HTTP error: ${response.status}`
                );
            }

            return response.json();

        }),

        minDelay

    ])

    // ===== API SUCCESS =====
    .then(([data]) => {

        console.log("API Response:", data);

        // ===== HIDE LOADER =====
        loadingMsg.style.display = "none";
        loadingMsg.innerHTML = "";
        button.disabled = false;


        // ===== DISPLAY RESULTS =====

        document.getElementById(
            "summary-favorite"
        ).textContent =
            data.summary.favorite;


        document.getElementById(
            "summary-favorite-name"
        ).textContent =
            driverNames[data.summary.favorite];


        document.getElementById(
            "summary-bestlap"
        ).textContent =
            data.summary.predicted_best_lap;


        document.getElementById(
            "summary-bestlap-name"
        ).textContent =
            driverNames[data.summary.predicted_best_lap];


        const container =
            document.getElementById(
                "driver-cards-container"
            );

        container.innerHTML = "";


        // Sort drivers by average of
        // win + podium + points probability
        const drivers =
            Object.keys(data.lap_time_predictions);


        const sorted = drivers.sort((a, b) => {

            const scoreA = (
                data.race_outcome_predictions[a].win_probability +
                data.race_outcome_predictions[a].podium_probability +
                data.race_outcome_predictions[a].points_finish_probability
            ) / 3;


            const scoreB = (
                data.race_outcome_predictions[b].win_probability +
                data.race_outcome_predictions[b].podium_probability +
                data.race_outcome_predictions[b].points_finish_probability
            ) / 3;


            return scoreB - scoreA;

        });


        sorted.forEach((driver, index) => {

            const lap =
                data.lap_time_predictions[driver];

            const outcome =
                data.race_outcome_predictions[driver];

            const analysis =
                data.driver_analysis[driver];

            const isFavourite =
                index === 0;


            const card =
                document.createElement("div");

            card.className =
                "driver-card";


            card.innerHTML = `

                ${
                    isFavourite
                        ? '<div class="featured-badge">🏆 RACE FAVOURITE</div>'
                        : ''
                }

                <div class="driver-card-header">

                    <img
                        src="drivers_images/${driver}.png"
                        alt="${driver}"
                    >

                    <div>

                        <h2>${driver}</h2>

                        <p class="driver-fullname">
                            ${driverNames[driver]}
                        </p>

                        <span class="driving-style">
                            ${analysis.driving_style}
                        </span>

                    </div>

                </div>

                <p>
                    Lap Time:
                    <strong style="color:#fff">
                        ${lap.predicted_lap_time}s
                    </strong>
                </p>

                <div class="prob-bar-label">
                    Win
                    <span>
                        ${(outcome.win_probability * 100).toFixed(1)}%
                    </span>
                </div>

                <div class="prob-bar">
                    <div
                        class="prob-fill"
                        style="
                            width:${(
                                outcome.win_probability * 100
                            ).toFixed(1)}%
                        "
                    ></div>
                </div>

                <div class="prob-bar-label">
                    Podium
                    <span>
                        ${(outcome.podium_probability * 100).toFixed(1)}%
                    </span>
                </div>

                <div class="prob-bar">
                    <div
                        class="prob-fill"
                        style="
                            width:${(
                                outcome.podium_probability * 100
                            ).toFixed(1)}%
                        "
                    ></div>
                </div>

                <div class="prob-bar-label">
                    Points
                    <span>
                        ${(outcome.points_finish_probability * 100).toFixed(1)}%
                    </span>
                </div>

                <div class="prob-bar">
                    <div
                        class="prob-fill"
                        style="
                            width:${(
                                outcome.points_finish_probability * 100
                            ).toFixed(1)}%
                        "
                    ></div>
                </div>

            `;


            container.appendChild(card);


            const canvas =
                document.createElement("canvas");

            canvas.id =
                `radar-${driver}`;

            canvas.style.marginTop =
                "16px";

            card.appendChild(canvas);


            new Chart(canvas, {

                type: "radar",

                data: {

                    labels: [
                        "Aggression",
                        "Consistency",
                        "Tyre Preservation",
                        "Overtaking",
                        "Braking"
                    ],

                    datasets: [{

                        data: [
                            analysis.aggression,
                            analysis.consistency,
                            analysis.tyre_preservation,
                            analysis.overtaking_ability,
                            analysis.braking_intensity
                        ],

                        backgroundColor:
                            "rgba(255, 42, 42, 0.15)",

                        borderColor:
                            "#ff2a2a",

                        pointBackgroundColor:
                            "#ff2a2a",

                        pointRadius: 3,

                        borderWidth: 2
                    }]
                },

                options: {

                    layout: {
                        padding: 20
                    },

                    scales: {

                        r: {

                            min: 0,
                            max: 1,

                            ticks: {
                                display: false
                            },

                            grid: {
                                color:
                                    "rgba(255,255,255,0.1)"
                            },

                            pointLabels: {

                                color: "#ccc",

                                font: {
                                    size: 11
                                }
                            }
                        }
                    },

                    plugins: {

                        legend: {
                            display: false
                        }

                    }

                }

            });

        });


        document.getElementById(
            "results-section"
        ).style.display = "block";


        document.getElementById(
            "results-section"
        ).scrollIntoView({
            behavior: "smooth"
        });

    })

    // ===== API ERROR =====
    .catch(error => {

        console.log("Error:", error);

        // ===== HIDE LOADER ON ERROR =====
        loadingMsg.style.display = "none";
        loadingMsg.innerHTML = "";
        button.disabled = false;

        // Show error
        document.getElementById(
            "error-msg"
        ).style.display = "block";
    });
}