let waiting = false;

window.onload = function () {
    loadRound();
};


async function loadRound() {

    waiting = false;

    document.getElementById("message").innerHTML = "";
    document.getElementById("reveal").innerHTML = "";

    document.getElementById("leftButton").disabled = false;
    document.getElementById("rightButton").disabled = false;

    const response = await fetch("/new_round");

    const data = await response.json();

    // Prevent browser caching
    document.getElementById("leftImage").src =
        data.left + "?t=" + Date.now();

    document.getElementById("rightImage").src =
        data.right + "?t=" + Date.now();

    document.getElementById("streak").innerHTML =
        data.streak;

    document.getElementById("best").innerHTML =
        data.best;

    document.getElementById("accuracy").innerHTML =
        data.accuracy + "%";
}


async function guess(side) {

    if (waiting)
        return;

    waiting = true;

    document.getElementById("leftButton").disabled = true;
    document.getElementById("rightButton").disabled = true;

    const response = await fetch("/guess", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({

            choice: side

        })

    });

    const data = await response.json();

    //------------------------------------------------------
    // Update scoreboard
    //------------------------------------------------------

    document.getElementById("streak").innerHTML =
        data.streak;

    document.getElementById("best").innerHTML =
        data.best;

    document.getElementById("accuracy").innerHTML =
        data.accuracy + "%";


    //------------------------------------------------------
    // Correct / Wrong message
    //------------------------------------------------------

    const message = document.getElementById("message");

    if (data.correct) {

        message.className = "message correct";

        message.innerHTML = "✔ Correct!";

    }

    else {

        message.className = "message wrong";

        message.innerHTML = "✘ Wrong!";

    }


    //------------------------------------------------------
    // Reveal
    //------------------------------------------------------

    const reveal = document.getElementById("reveal");

    reveal.innerHTML = `

        <b>Left Image:</b>
        ${data.fake_side === "left"
            ? "Generated"
            : "Real"
        }
        <br>

        <b>Right Image:</b>
        ${data.fake_side === "right"
            ? "Generated"
            : "Real"
        }

        <br><br>

        Generated Digit:
        <b>${data.fake_label}</b>

        <br>

        Real Digit:
        <b>${data.real_label}</b>

    `;


    //------------------------------------------------------
    // Next round
    //------------------------------------------------------

    setTimeout(loadRound, 2000);

}