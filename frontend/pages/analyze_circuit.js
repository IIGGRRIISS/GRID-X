let selectedFile = null;


// =====================================================
// FILE UPLOAD
// =====================================================

function handleFile(event) {

    const file = event.target.files[0];

    if (!file) return;

    selectedFile = file;

    showPreview(file);
}


// =====================================================
// SHOW IMAGE PREVIEW
// =====================================================

function showPreview(file) {

    const reader = new FileReader();

    reader.onload = function (e) {

        document.getElementById("image-preview").src = e.target.result;

        document.getElementById("preview-container").style.display = "block";

        document.getElementById("analyze-btn").style.display = "inline-block";

        document.getElementById("loading-msg").style.display = "none";

        document.getElementById("results-section").style.display = "none";

        document.getElementById("error-msg").style.display = "none";
    };

    reader.readAsDataURL(file);
}


// =====================================================
// DRAG AND DROP
// =====================================================

const uploadArea = document.getElementById("upload-area");


uploadArea.addEventListener("dragover", function (e) {

    e.preventDefault();

    uploadArea.classList.add("drag-over");

});


uploadArea.addEventListener("dragleave", function () {

    uploadArea.classList.remove("drag-over");

});


uploadArea.addEventListener("drop", function (e) {

    e.preventDefault();

    uploadArea.classList.remove("drag-over");

    const file = e.dataTransfer.files[0];

    if (!file) return;

    selectedFile = file;

    showPreview(file);

});


// =====================================================
// FORMAT METADATA KEYS
// =====================================================

function formatKey(key) {

    return key
        .replace(/\_/g, " ")
        .replace(/\b\w/g, c => c.toUpperCase());
}


// =====================================================
// ANALYZE CIRCUIT
// =====================================================

function analyzeCircuit() {

    if (!selectedFile) return;


    // Hide previous results/errors
    document.getElementById("results-section").style.display = "none";

    document.getElementById("error-msg").style.display = "none";


    // Show loading spinner
    document.getElementById("loading-msg").style.display = "block";


    // Hide analyze button while request is running
    document.getElementById("analyze-btn").style.display = "none";


    // Build form data
    const formData = new FormData();

    formData.append("file", selectedFile);


    // Minimum loading time so spinner is actually visible
    const minDelay = new Promise(resolve =>
        setTimeout(resolve, 1500)
    );


    Promise.all([

        fetch("http://localhost:8000/analyze-circuit", {

            method: "POST",

            body: formData

        }).then(response => {

            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }

            return response.json();

        }),

        minDelay

    ])


    // =================================================
    // SUCCESS
    // =================================================

    .then(([data]) => {

        console.log(data);


        // Hide loading
        document.getElementById("loading-msg").style.display = "none";


        // Bring Analyze button back
        document.getElementById("analyze-btn").style.display = "inline-block";


        // Check API response
        if (!data || !data.success) {

            document.getElementById("error-msg").style.display = "block";

            return;
        }


        // =================================================
        // TOP PREDICTION
        // =================================================

        if (!data.predictions || data.predictions.length === 0) {

            document.getElementById("error-msg").style.display = "block";

            return;
        }


        const top = data.predictions[0];


        const confidence = (top.confidence * 100).toFixed(1);


        document.getElementById("top-circuit-name").textContent =
            top.circuit
                .replace(/\_/g, " ")
                .toUpperCase();


        document.getElementById("top-circuit-confidence").textContent =
            `Confidence: ${confidence}%`;


        document.getElementById("conf-fill").style.width =
            `${confidence}%`;


        // =================================================
        // METADATA GRID
        // =================================================

        const metaGrid =
            document.getElementById("metadata-grid");


        metaGrid.innerHTML = "";


        if (
            top.metadata &&
            Object.keys(top.metadata).length > 0
        ) {

            Object.entries(top.metadata).forEach(
                ([key, value]) => {

                    // Description handled separately
                    if (key === "description") return;


                    const item =
                        document.createElement("div");


                    item.className = "meta-item";


                    item.innerHTML = `
                        <p class="meta-key">
                            ${formatKey(key)}
                        </p>

                        <p class="meta-value">
                            ${value}
                        </p>
                    `;


                    metaGrid.appendChild(item);

                }
            );


            // Description
            if (top.metadata.description) {

                const desc =
                    document.createElement("div");


                desc.className = "meta-description";


                desc.textContent =
                    top.metadata.description;


                metaGrid.appendChild(desc);
            }

        }


        // =================================================
        // OTHER PREDICTIONS
        // =================================================

        const othersContainer =
            document.getElementById("other-predictions");


        othersContainer.innerHTML = "";


        data.predictions
            .slice(1)
            .forEach(pred => {

                const conf =
                    (pred.confidence * 100).toFixed(1);


                const card =
                    document.createElement("div");


                card.className = "other-card";


                card.innerHTML = `

                    <p class="other-name">
                        ${pred.circuit
                            .replace(/\_/g, " ")
                            .toUpperCase()}
                    </p>

                    <p class="other-conf">
                        ${conf}%
                    </p>

                    <div class="other-bar">

                        <div
                            class="other-fill"
                            style="width:${conf}%"
                        ></div>

                    </div>

                `;


                othersContainer.appendChild(card);

            });


        // =================================================
        // SHOW RESULTS
        // =================================================

        document.getElementById("results-section").style.display =
            "block";


        // IMPORTANT:
        // No scrollIntoView here.
        // This prevents the page from jumping automatically.

    })


    // =================================================
    // ERROR
    // =================================================

    .catch(error => {

        console.log("Error:", error);


        // Hide loading
        document.getElementById("loading-msg").style.display =
            "none";


        // Bring button back
        document.getElementById("analyze-btn").style.display =
            "inline-block";


        // Show error
        document.getElementById("error-msg").style.display =
            "block";

    });

}


// =====================================================
// ENTER KEY
// =====================================================

document.addEventListener("keydown", function (e) {

    if (e.key === "Enter" && selectedFile) {

        e.preventDefault();

        analyzeCircuit();

    }

});