const songTitle = document.getElementsByClassName('audio-player-title');
const fillBar = document.getElementById('fill');
const progressBar = document.getElementById('progressBar');
const handle = document.getElementById('handle');
const currentMinutesTime = document.getElementById('current-time-minutes-song');
let songSrc = "../audios/"+"chapter1"+".mp3";


window.audioPlayer = new Audio();
window.audioPlayer.src = songSrc;

window.isSeeking = false; // Global variable to indicate if the user is dragging

function playOrPauseSong() {
    if (window.audioPlayer.paused) {
        window.audioPlayer.play();
        $("#play img").attr("src", "./img/pause-button.png");
    } else {
        window.audioPlayer.pause();
        $("#play img").attr("src", "./img/play-button.png");
    }
}

function updateTime() {
    window.audioPlayer.addEventListener('timeupdate', () => {
        if (!window.isSeeking) {
            let currentTimeFormatted = formatTime(window.audioPlayer.currentTime);
            let totalDurationFormatted = formatTime(window.audioPlayer.duration);

            currentMinutesTime.textContent = `${currentTimeFormatted} / ${totalDurationFormatted}`;

            // Update progress bar and handle position
            let position = (window.audioPlayer.currentTime / window.audioPlayer.duration) * 100;
            fillBar.style.width = position + '%';
            handle.style.left = `calc(${position}% - 8px)`;
        }
    });

    // Get total duration when metadata is loaded
    window.audioPlayer.addEventListener('loadedmetadata', () => {
        let totalDurationFormatted = formatTime(window.audioPlayer.duration);
        currentMinutesTime.textContent = `00:00 / ${totalDurationFormatted}`;
    });
}

function formatTime(timeInSeconds) {
    if (isNaN(timeInSeconds)) {
        return "00:00";
    }

    let hours = Math.floor(timeInSeconds / 3600);
    let minutes = Math.floor((timeInSeconds % 3600) / 60);
    let seconds = Math.floor(timeInSeconds % 60);

    if (hours > 0) {
        return `${hours < 10 ? '0' : ''}${hours}:${minutes < 10 ? '0' : ''}${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
    } else {
        return `${minutes < 10 ? '0' : ''}${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
    }
}

// Add drag functionality
let isDragging = false;

progressBar.addEventListener('mousedown', (e) => {
    isDragging = true;
    window.isSeeking = true; // Start dragging
    updateProgress(e);
});

document.addEventListener('mousemove', (e) => {
    if (isDragging) {
        updateProgress(e);
    }
});

document.addEventListener('mouseup', () => {
    if (isDragging) {
        isDragging = false;
        window.isSeeking = false; // End dragging

        // Dispatch seeked event to notify other listeners
        const seekedEvent = new Event('seeked');
        window.audioPlayer.dispatchEvent(seekedEvent);
    }
});

function updateProgress(e) {
    const rect = progressBar.getBoundingClientRect();
    let clickX = e.clientX - rect.left;

    // Limit click position within progress bar bounds
    clickX = Math.max(0, Math.min(clickX, progressBar.clientWidth));

    // Calculate new play time and update audioPlayer
    const newTime = (clickX / progressBar.clientWidth) * window.audioPlayer.duration;
    window.audioPlayer.currentTime = newTime;

    // Update progress bar and handle position
    const position = (window.audioPlayer.currentTime / window.audioPlayer.duration) * 100;
    fillBar.style.width = position + '%';
    handle.style.left = `calc(${position}% - 8px)`;
}

updateTime();