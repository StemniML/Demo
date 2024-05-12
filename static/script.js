document.addEventListener("DOMContentLoaded", function() {
    console.log("The DOM is ready.");
});

function clearInputs() {
    for (var i = 1; i <= 20; i++) {
        var input = document.getElementById('feature_' + i);
        var toggle = document.getElementById('toggle_feature_' + i);
        input.value = "";
        toggle.value = "0";
    }
}

function loadData() {
    // Fill input fields with random data between 0 and 1
    for (var i = 1; i <= 20; i++) {
        var randomNumber = Math.random(); // Generate a random number between 0 and 1
        document.getElementById('feature_' + i).value = randomNumber.toFixed(2); // Round to 2 decimal places
        var toggleValue = Math.random() < 0.5 ? "0" : "1"; // Randomly set toggle to "No" or "Yes"
        document.getElementById('toggle_feature_' + i).value = toggleValue;

        // Simulate negative class for some features
        if (toggleValue === "0") {
            document.getElementById('feature_' + i).value = (Math.random() * 0.5).toFixed(2); // Set low values for negative class
        }
    }
    calculateAUC(); // Recalculate AUC score after loading new data
}

function checkInputs() {
    var inputs = document.querySelectorAll('input[type="number"]');
    var missingInput = false;
    inputs.forEach(function(input) {
        if (input.value === "") {
            missingInput = true;
            input.style.border = "1px solid red"; // Highlight missing inputs
        } else {
            input.style.border = ""; // Reset border for filled inputs
        }
    });
    if (missingInput) {
        alert("Please enter values for all features!");
        return false; // Prevent form submission
    }
    return true; // Allow form submission
}

function calculateAUC() {
    var form = document.getElementById("inputForm");
    var formData = new FormData(form);

    fetch('/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(html => {
        var temp = document.createElement('div');
        temp.innerHTML = html;
        var aucScoreElement = temp.querySelector('#auc_score');
        var pltPath = temp.querySelector('#plt_path').getAttribute('src');
        document.getElementById('auc_score').innerText = aucScoreElement.innerText;
        document.getElementById('roc_curve_img').setAttribute('src', pltPath);
    })
    .catch(error => console.error('Error:', error));
}
