let gaugeChart = null;


// ===== ENTER KEY NAVIGATION =====

const formFields = [
    "driver",
    "circuit",
    "compound",
    "tyre_age",
    "current_lap_time",
    "air_temp",
    "track_temp",
    "humidity",
    "rainfall",
    "position",
    "total_laps",
    "lap_number"
];


document.addEventListener("keydown", function (e) {

    if (e.key !== "Enter") return;

    const active = document.activeElement;
    const currentIndex = formFields.indexOf(active.id);

    if (currentIndex !== -1 && currentIndex < formFields.length - 1) {

        e.preventDefault();

        document
            .getElementById(formFields[currentIndex + 1])
            .focus();

    } else if (currentIndex === formFields.length - 1) {

        e.preventDefault();

        checkTireSafety();

    }

});


// ===== RISK COLOR =====

function getRiskColor(category) {

    switch (category) {

        case "SAFE":
            return "#00ff87";

        case "CAUTION":
            return "#ffcc00";

        case "CRITICAL":
            return "#ff2a2a";

        default:
            return "#aaaaaa";

    }

}


// ===== MAIN TYRE SAFETY FUNCTION =====

function checkTireSafety() {

    // Elements
    const loadingMsg =
        document.getElementById("loading-msg");

    const checkBtn =
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
            <span>Checking Tyre Safety</span>
        </div>
    `;

    loadingMsg.style.display = "flex";


    // Disable button while API is running

    if (checkBtn) {
        checkBtn.disabled = true;
    }


    // ===== COLLECT INPUTS =====

    const lapNumber =
        parseInt(
            document.getElementById("lap_number").value || "26"
        );

    const totalLaps =
        parseInt(
            document.getElementById("total_laps").value || "52"
        );

    const sessionProgress =
        parseFloat(
            (lapNumber / totalLaps).toFixed(3)
        );

    const tyreAge =
        parseInt(
            document.getElementById("tyre_age").value || "14"
        );


    const values = {

        driver:
            document.getElementById("driver").value,

        circuit:
            document.getElementById("circuit").value,

        compound:
            document.getElementById("compound").value,

        tyre_age:
            tyreAge,

        stint_lap_number:
            tyreAge,

        current_lap_time:
            parseFloat(
                document.getElementById("current_lap_time").value || "89.2"
            ),

        track_temp:
            parseFloat(
                document.getElementById("track_temp").value || "25"
            ),

        air_temp:
            parseFloat(
                document.getElementById("air_temp").value || "18"
            ),

        humidity:
            parseFloat(
                document.getElementById("humidity").value || "65"
            ),

        rainfall:
            parseFloat(
                document.getElementById("rainfall").value || "0"
            ),

        position:
            parseInt(
                document.getElementById("position").value || "1"
            ),

        session_progress:
            sessionProgress,

        total_laps:
            totalLaps

    };


    // ===== MINIMUM LOADING TIME =====

    const minDelay = new Promise(resolve =>
        setTimeout(resolve, 1500)
    );


    // ===== API REQUEST =====

    Promise.all([

        fetch(
            "http://localhost:8000/tire-safety-check",
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

        if (checkBtn) {
            checkBtn.disabled = false;
        }


        // ===== CHECK API SUCCESS =====

        if (!data || !data.success) {

            document.getElementById(
                "error-msg"
            ).style.display = "block";

            return;

        }


        // ===== RISK COLOR =====

        const riskColor =
            getRiskColor(
                data.risk_category
            );


        // ===== RISK SCORE =====

        document.getElementById(
            "risk-score"
        ).textContent =
            (data.risk_score || 0).toFixed(1);


        document.getElementById(
            "risk-score"
        ).style.color =
            riskColor;


        document.getElementById(
            "risk-category"
        ).textContent =
            data.risk_category;


        document.getElementById(
            "risk-category"
        ).style.color =
            riskColor;


        // ===== ACTION CARDS =====

        document.getElementById(
            "recommended-action"
        ).textContent =
            data.recommended_action;


        document.getElementById(
            "safe-laps"
        ).textContent =
            data.safe_remaining_laps;


        // ===== GAUGE CHART =====

        if (gaugeChart) {
            gaugeChart.destroy();
        }


        const ctx =
            document
                .getElementById("gaugeChart")
                .getContext("2d");


        gaugeChart = new Chart(ctx, {

            type: "doughnut",

            data: {

                datasets: [{

                    data: [
                        data.risk_score,
                        100 - data.risk_score
                    ],

                    backgroundColor: [
                        riskColor,
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


        // ===== SHAP EXPLANATION FACTORS =====

        const factorsList =
            document.getElementById(
                "factors-list"
            );

        factorsList.innerHTML = "";


        if (data.explanation) {

            data.explanation.forEach(item => {

                const isPositive =
                    item.shap_value >= 0;


                const factor =
                    document.createElement("div");


                factor.className =
                    "factor-item";


                factor.innerHTML = `

                    <span class="factor-name">
                        ${item.feature.replace(/\_/g, " ")}
                    </span>

                    <span
                        class="factor-value"
                        style="color:${
                            isPositive
                                ? "#ff2a2a"
                                : "#0096ff"
                        }"
                    >
                        ${
                            isPositive ? "+" : ""
                        }${item.shap_value.toFixed(3)}
                    </span>

                `;


                factorsList.appendChild(factor);

            });

        }


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

        if (checkBtn) {
            checkBtn.disabled = false;
        }


        // Show error

        document.getElementById(
            "error-msg"
        ).style.display = "block";

    });

}