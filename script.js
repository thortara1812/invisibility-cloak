// Optional: A simple script to refresh the image for better user experience
// This script automatically reloads the image to get the next frame in real-time

window.onload = function() {
    const videoElement = document.getElementById('live-video');
    
    // Function to reload the image every 100 milliseconds
    setInterval(function() {
        videoElement.src = '/video_feed/' + filename + '?' + new Date().getTime(); // Adding a timestamp for cache busting
    }, 100);  // Adjust interval if needed
};
