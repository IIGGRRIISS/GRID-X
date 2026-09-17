function formatRaceTime(seconds) {

    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(2);

    return `${mins}:${secs < 10 ? "0" : ""}${secs}`;
}


function getCompoundColor(compound) {

    switch (compound) {

        case "SOFT":
            return "#ff2a2a";

        case "MEDIUM":
            return "#ffcc00";

        case "HARD":
            return "#ffffff";

        default:
            return "#aaaaaa";
    }
}


// ===== INPUT RANGE CLAMPING =====

const inputRanges = {

    air_temp:   { min: 5,  max: 45 },
    track_temp: { min: 5,  max: 60 },
    humidity:   { min: 0,  max: 100 },
    rainfall:   { min: 0,  max: 50 },
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


// ===== ENTER KEY NAVIGATION =====

const NAV_ORDER = [

    "driver",
    "circuit",
    "start_compound",
    "air_temp",
    "track_temp",
    "humidity",
    "rainfall",
    "total_laps"

];


document.addEventListener("keydown", function (e) {

    if (e.key !== "Enter") return;

    const active = document.activeElement;
    const idx = NAV_ORDER.indexOf(active.id);

    if (idx === -1) return;

    e.preventDefault();

    if (idx < NAV_ORDER.length - 1) {

        document
            .getElementById(NAV_ORDER[idx + 1])
            .focus();

    } else {

        optimizeStrategy();

    }

});


// ===== MAIN OPTIMIZE FUNCTION =====

function optimizeStrategy() {

    // Elements
    const loadingMsg =
        document.getElementById("loading-msg");

    const optimizeBtn =
        document.querySelector(".predict-btn");


    // Hide previous results/errors
    document.getElementById(
        "results-section"
    ).style.display = "none";

    document.getElementById(
        "error-msg"
    ).style.display = "none";


    // ===== SHOW LOADING STATE =====

    loadingMsg.innerHTML = `
        <div class="loading-state">
            <span class="loading-spinner"></span>
            <span>Optimizing Strategy</span>
        </div>
    `;

    loadingMsg.style.display = "flex";


    // Disable button while API is running
    if (optimizeBtn) {
        optimizeBtn.disabled = true;
    }


    // ===== CLAMP FUNCTION =====

    function clamp(val, min, max) {

        return Math.min(
            max,
            Math.max(min, val)
        );

    }


    // ===== COLLECT INPUT VALUES =====

    const values = {

        driver:
            document.getElementById("driver").value,

        circuit:
            document.getElementById("circuit").value,

        start_compound:
            document.getElementById("start_compound").value,

        weather: {

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

        total_laps: clamp(
            parseInt(
                document.getElementById("total_laps").value || "52"
            ),
            10,
            100
        )

    };


    // ===== MINIMUM LOADING TIME =====

    const minDelay = new Promise(resolve =>
        setTimeout(resolve, 1500)
    );


    // ===== API REQUEST =====

    Promise.all([

        fetch(
            "http://localhost:8000/strategy-optimize",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(values)
            }
        ).then(response => response.json()),

        minDelay

    ])


    // ===== SUCCESS =====

    .then(([data]) => {

        console.log("API Response:", data);


        // Hide loading
        loadingMsg.style.display = "none";
        loadingMsg.innerHTML = "";


        // Re-enable button
        if (optimizeBtn) {
            optimizeBtn.disabled = false;
        }


        // ===== RACE TIME =====

        document.getElementById(
            "res-race-time"
        ).textContent =
            formatRaceTime(data.total_race_time);


        // ===== PIT STOP COUNT =====

        document.getElementById(
            "res-pit-count"
        ).textContent =
            data.pit_stops.length;


        const totalLaps =
            values.total_laps;


        // ===== TIMELINE =====

        const container =
            document.getElementById(
                "timeline-container"
            );

        container.innerHTML = "";


        const allStops = [

            {
                lap: 1,
                compound: values.start_compound
            },

            ...data.pit_stops.map(
                (lap, i) => ({
                    lap,
                    compound:
                        data.pit_compounds[i]
                })
            )

        ];


        allStops.forEach((stop, i) => {

            const width =
                100 / allStops.length;

            const color =
                getCompoundColor(
                    stop.compound
                );


            const segment =
                document.createElement("div");


            segment.className =
                "timeline-segment";


            segment.style.width =
                `${width}%`;


            segment.style.borderColor =
                color;


            segment.innerHTML = `

                <span
                    class="segment-label"
                    style="color:${color}"
                >
                    LAP ${stop.lap}
                </span>

                <span
                    class="segment-compound"
                    style="color:${color}"
                >
                    ${stop.compound}
                </span>

            `;


            container.appendChild(segment);

        });


        // ===== PIT STOP TABLE =====

        const tbody =
            document.getElementById(
                "pit-table-body"
            );

        tbody.innerHTML = "";


        data.pit_stops.forEach((lap, i) => {

            const row =
                document.createElement("tr");


            const color =
                getCompoundColor(
                    data.pit_compounds[i]
                );


            row.innerHTML = `

                <td>
                    Stop ${i + 1}
                </td>

                <td>
                    Lap ${lap}
                </td>

                <td
                    style="
                        color:${color};
                        font-weight:bold;
                    "
                >
                    ${data.pit_compounds[i]}
                </td>

            `;


            tbody.appendChild(row);

        });


        // ===== SHOW RESULTS =====

        document.getElementById(
            "results-section"
        ).style.display = "block";


        document.getElementById(
            "results-section"
        ).scrollIntoView({
            behavior: "smooth"
        });

    })


    // ===== ERROR =====

    .catch(error => {

        console.log("Error:", error);


        // Hide loading
        loadingMsg.style.display = "none";
        loadingMsg.innerHTML = "";


        // Re-enable button
        if (optimizeBtn) {
            optimizeBtn.disabled = false;
        }


        // Show error
        document.getElementById(
            "error-msg"
        ).style.display = "block";

    });

}