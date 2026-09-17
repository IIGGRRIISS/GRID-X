const START_TIME = 0;
const DURATION = 5.5;

const roarSound = new Audio("index_sound.mp3");

document.getElementById("logo").addEventListener("click", () => {
    console.log("Playing...");
    roarSound.play()
        .then(() => console.log("Started"))
        .catch(err => console.error(err));
});

document.getElementById("logo").addEventListener("click", () => {
    console.log("Logo clicked");
});