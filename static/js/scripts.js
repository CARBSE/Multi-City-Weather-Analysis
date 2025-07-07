document.addEventListener("DOMContentLoaded", function () {
    const stateSelect = document.getElementById("state");
    const citySelect = document.getElementById("city");
    const comfortModel = document.getElementById("comfort_model");
    const previewMap = document.getElementById("preview-map");
    const comfortChart = document.getElementById("comfort-chart");
    const chart24x7 = document.getElementById("chart-24x7");
    const chart9x6 = document.getElementById("chart-9x6");

    function updateCityOptions() {
        const selectedState = stateSelect.value;
        fetch(`/?state=${encodeURIComponent(selectedState)}`)
            .then(() => {
                // Let the server handle rendering on form submission.
            });
    }

    if (stateSelect) {
        stateSelect.addEventListener("change", function () {
            this.form.submit(); // Triggers re-render with updated cities.
        });
    }

    function updateImages() {
        const city = citySelect.value;
        const model = comfortModel.value;

        previewMap.src = `/map_preview?city=${encodeURIComponent(city)}`;
        comfortChart.src = `/chart/comfort?city=${encodeURIComponent(city)}&model=${encodeURIComponent(model)}`;
        chart24x7.src = `/chart/24x7?city=${encodeURIComponent(city)}&model=${encodeURIComponent(model)}`;
        chart9x6.src = `/chart/9x6?city=${encodeURIComponent(city)}&model=${encodeURIComponent(model)}`;
    }

    if (citySelect && comfortModel) {
        citySelect.addEventListener("change", updateImages);
        comfortModel.addEventListener("change", updateImages);
    }

    updateImages(); // Load initial images if values already selected
});
