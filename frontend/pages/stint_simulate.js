let stintChart = null;

// ===== INPUT RANGE CLAMPING =====

const inputRanges = {
    air_temp:   { min: 5,  max: 45 },
    track_temp: { min: 5,  max: 60 },
    humidity:   { min: 0,  max: 100 },
    rainfall:   { min: 0,  max: 50 },
    n_laps:     { min: 1,  max: 50 }
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
    "compound",
    "air_temp",
    "track_temp",
    "humidity",
    "rainfall",
    "n_laps"
];

document.addEventListener("keydown", function (e) {

    if (e.key !== "Enter") return;

    const active = document.activeElement;
    const idx = NAV_ORDER.indexOf(active.id);

    if (idx === -1) return;

    e.preventDefault();

    if (idx < NAV_ORDER.length - 1) {

        document.getElementById(
            NAV_ORDER[idx + 1]
        ).focus();

    } else {

        simulateStint();

    }
});


// ===== MAIN SIMULATE FUNCTION =====

function simulateStint() {

    // Elements
    const loadingMsg = document.getElementById("loading-msg");
    const simulateBtn = document.querySelector(".predict-btn");

    // Hide previous results/errors
    document.getElementById("results-section").style.display = "none";
    document.getElementById("error-msg").style.display = "none";

    // ===== SHOW LOADING STATE =====

    loadingMsg.innerHTML = `
        <div class="loading-state">
            <span class="loading-spinner"></span>
            <span>Simulating Stint</span>
        </div>
    `;

    loadingMsg.style.display = "flex";

    // Disable button while prediction is running
    if (simulateBtn) {
        simulateBtn.disabled = true;
    }


    // ===== CLAMP FUNCTION =====

    function clamp(val, min, max) {
        return Math.min(max, Math.max(min, val));
    }


    // ===== COLLECT INPUT VALUES =====

    const values = {

        driver_code:
            document.getElementById("driver").value,

        circuit:
            document.getElementById("circuit").value,

        compound:
            document.getElementById("compound").value,

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

        n_laps: clamp(
            parseInt(
                document.getElementById("n_laps").value || "15"
            ),
            1,
            50
        )
    };


    // ===== MINIMUM LOADING TIME =====

    const minDelay = new Promise(resolve =>
        setTimeout(resolve, 1500)
    );


    // ===== API REQUEST =====

    Promise.all([

        fetch("http://localhost:8000/stint-simulate", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(values)

        }).then(response => response.json()),

        minDelay

    ])

    // ===== SUCCESS =====

    .then(([data]) => {

        console.log("API Response:", data);

        // Hide loading
        loadingMsg.style.display = "none";
        loadingMsg.innerHTML = "";

        // Re-enable button
        if (simulateBtn) {
            simulateBtn.disabled = false;
        }


        // ===== DISPLAY BASIC RESULTS =====

        document.getElementById("res-driver").textContent =
            data.driver;

        document.getElementById("res-circuit").textContent =
            data.circuit
                .replace(/\_/g, " ")
                .toUpperCase();

        document.getElementById("res-compound").textContent =
            data.compound;

        document.getElementById("res-laps").textContent =
            data.n_laps;


        // ===== BUILD CHART =====

        const labels = data.laps.map(
            (_, i) => `Lap ${i + 1}`
        );

        const lapTimes = data.laps.map(
            t => t.toFixed(3)
        );


        if (stintChart) {
            stintChart.destroy();
        }


        const ctx =
            document
                .getElementById("stintChart")
                .getContext("2d");


        stintChart = new Chart(ctx, {

            type: "line",

            data: {

                labels,

                datasets: [{

                    label: "Lap Time (s)",

                    data: lapTimes,

                    borderColor: "#00d4ff",

                    backgroundColor:
                        "rgba(0, 212, 255, 0.08)",

                    pointBackgroundColor:
                        "#00d4ff",

                    pointRadius: 4,

                    borderWidth: 2,

                    tension: 0.3,

                    fill: true

                }]
            },


            options: {

                responsive: true,

                plugins: {

                    legend: {
                        display: false
                    }

                },

                scales: {

                    x: {

                        ticks: {
                            color: "#aaa",
                            font: {
                                family: "Rajdhani"
                            }
                        },

                        grid: {
                            color:
                                "rgba(255,255,255,0.05)"
                        }

                    },

                    y: {

                        ticks: {
                            color: "#aaa",
                            font: {
                                family: "Rajdhani"
                            }
                        },

                        grid: {
                            color:
                                "rgba(255,255,255,0.05)"
                        }

                    }

                }

            }

        });


        // ===== BUILD LAP TABLE =====

        const tbody =
            document.getElementById(
                "lap-table-body"
            );

        tbody.innerHTML = "";


        data.laps.forEach((time, i) => {

            const delta =
                i === 0
                    ? "—"
                    : "+" +
                      (
                          time -
                          data.laps[i - 1]
                      ).toFixed(3) +
                      "s";


            const row =
                document.createElement("tr");


            row.innerHTML = `

                <td>Lap ${i + 1}</td>

                <td>${time.toFixed(3)}s</td>

                <td class="${
                    delta === "—"
                        ? ""
                        : "delta"
                }">
                    ${delta}
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
        if (simulateBtn) {
            simulateBtn.disabled = false;
        }

        // Show error
        document.getElementById(
            "error-msg"
        ).style.display = "block";

    });
}