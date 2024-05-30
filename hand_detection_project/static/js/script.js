document.addEventListener("DOMContentLoaded", function() {
    // Add event listener to the recognition form
    document.getElementById("recognition-form").addEventListener("submit", function(event) {
        // Prevent the default form submission behavior
        event.preventDefault();

        // Send an asynchronous POST request to the server to start gesture recognition
        fetch("/recognize_gesture", {
            method: "POST"
        })
        .then(response => {
            // Check if the response is successful
            if (response.ok) {
                // If successful, parse the response as JSON
                return response.json();
            } else {
                // If not successful, throw an error
                throw new Error("Failed to recognize hand gesture");
            }
        })
        .then(data => {
            // Display a message to the user
            document.getElementById("result").innerText = data.message;
            // Start polling for the result
            checkResult();
        })
        .catch(error => {
            // Handle any errors that occur during the fetch operation
            console.error("Error:", error);
            // Display an error message to the user (optional)
            alert("An error occurred while recognizing hand gesture. Please try again later.");
        });
    });

    function checkResult() {
        // Send an asynchronous GET request to check the result
        fetch("/get_result", {
            method: "GET"
        })
        .then(response => {
            // Check if the response is successful
            if (response.ok) {
                // If successful, parse the response as JSON
                return response.json();
            } else {
                // If not successful, throw an error
                throw new Error("Failed to fetch result");
            }
        })
        .then(data => {
            // Check if the result is ready
            if (data.result) {
                // Display the result to the user
                document.getElementById("result").innerText = data.result;
            } else {
                // If the result is not ready, poll again after a delay
                setTimeout(checkResult, 1000); // Check again after 1 second
            }
        })
        .catch(error => {
            // Handle any errors that occur during the fetch operation
            console.error("Error:", error);
            // Display an error message to the user (optional)
            alert("An error occurred while fetching the result. Please try again later.");
        });
    }
});
