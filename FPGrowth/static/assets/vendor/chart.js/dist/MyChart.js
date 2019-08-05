$(document).ready(function () {


    /**
     * Options for Line chart
     */

    var globalOptions = {
        responsive: true,
        legend: {
            labels:{
                fontColor:"#90969D"
            }
        },
        scales: {
            xAxes: [{
                ticks: {
                    fontColor: "#90969D"
                },
                gridLines: {
                    color: "#37393F"
                }
            }],
            yAxes: [{
                ticks: {
                    fontColor: "#90969D"
                },
                gridLines: {
                    color: "#37393F"
                }
            }]
        }
    };

    var lineData = {
        labels: ["January", "February", "March", "April", "May", "June", "July"],
        datasets: [

            {
                label: "Data 1",
                backgroundColor: 'transparent',
                borderColor: "#f6a821",
                pointBorderWidth: 0,
                pointRadius: 2,
                pointBorderColor: '#f6a821',
                borderWidth: 1,
                data: [16, 32, 18, 26, 42, 33, 44]
            },
            {
                label: "Data 2",
                backgroundColor: 'transparent',
                borderColor: "#676B73",
                pointBorderWidth: 0,
                pointRadius: 2,
                pointBorderColor: '#676B73',
                borderWidth: 1,
                data: [22, 44, 67, 43, 76, 45, 12]
            }
        ]
    };

    var c1 = document.getElementById("lineOptions").getContext("2d");
    new Chart(c1, {type: 'line', data: lineData, options: globalOptions});

    /**
     * Options for Sharp Line chart
     */
    var sharpLineData = {
        labels: ["January", "February", "March", "April", "May", "June", "July"],
        datasets: [
            {
                label: "Example dataset",
                backgroundColor: 'rgba(246,168,33, 0.1)',
                borderColor: "#f6a821",
                pointBorderWidth: 0,
                pointRadius: 2,
                pointBorderColor: '#f6a821',
                borderWidth: 1,
                data: [33, 48, 40, 19, 54, 27, 54],
                lineTension: 0
            }
        ]
    };

    var c2 = document.getElementById("sharpLineOptions").getContext("2d");
    new Chart(c2, {type: 'line', data: sharpLineData, options: globalOptions});


    /**
     * Data for Bar chart
     */
    var barData = {
        labels: ["January", "February", "March", "April", "May", "June", "July"],
        datasets: [
            {
                label: "Data 1",
                backgroundColor: 'transparent',
                borderColor: "#f6a821",
                borderWidth: 1,
                data: [9, 8, 5, 6, 3, 2, 16]
            },
            {
                label: "Data 2",
                backgroundColor: 'transparent',
                borderColor: "#676B73",
                borderWidth: 1,
                data: [5, 5, 5, 4, 5, 2, 23]
            }
        ]
    };

    var c3 = document.getElementById("barOptions").getContext("2d");
    new Chart(c3, {type: 'bar', data: barData, options: globalOptions});

    /**
     * Options for Bar chart
     */

    /**
     * Data for Bar chart
     */
    var singleBarData = {
        labels: ["January", "February", "March", "April", "May", "June", "July"],
        datasets: [
            {
                label: "Data 0",
                backgroundColor: 'transparent',
                borderColor: "#f6a821",
                borderWidth: 1,
                data: [15, 20, 30, 40, 30, 50, 60]
            }
        ]
    };

    var c4 = document.getElementById("singleBarOptions").getContext("2d");
    new Chart(c4, {type: 'bar', data: singleBarData, options: globalOptions});


    var polarData = {
        datasets: [{
            data: [
                120,
                130,
                100
            ],
            borderWidth: 0,
            backgroundColor: [
                "rgba(246,168,33, 0.1)",
                "rgba(246,168,33, 0.4)",
                "rgba(246,168,33, 0.8)"
            ],
            label: 'My dataset' // for legend
        }],
        labels: [
            "Homer",
            "Inspinia",
            "Luna"
        ]
    };

    var c5 = document.getElementById("polarOptions").getContext("2d");
    new Chart(c5, {type: 'polarArea', data: polarData, options: globalOptions});


    var doughnutData = {
        labels: [
            "App",
            "Software",
            "Laptop"
        ],
        datasets: [
            {
                data: [20, 120, 100],
                borderWidth: 0,
                backgroundColor: [
                    "rgba(246,168,33, 0.1)",
                    "rgba(246,168,33, 0.4)",
                    "rgba(246,168,33, 0.8)"
                ],
                hoverBackgroundColor: [
                    "#F6A821",
                    "#F6A821",
                    "#F6A821"
                ]
            }]
    };


    var c6 = document.getElementById("doughnutChart").getContext("2d");
    new Chart(c6, {type: 'doughnut', data: doughnutData, options: globalOptions});

});