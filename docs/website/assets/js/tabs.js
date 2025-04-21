function openTab(evt, tabName) {
    var i, tabcontent, tabbuttons;
    
    // Hide all tab content
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    
    // Remove "active" class from all tab buttons
    tabbuttons = document.getElementsByClassName("tab-button");
    for (i = 0; i < tabbuttons.length; i++) {
        tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
    }
    
    // Show the current tab and add "active" class to the button
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}

// Nested tab functionality
function openNestedTab(evt, tabName) {
    var i, tabcontent, tabbuttons;
    
    // Hide all nested tab content within the parent tab
    var parentTab = evt.currentTarget.closest('.tab-content');
    tabcontent = parentTab.getElementsByClassName("nested-tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    
    // Remove "active" class from all nested tab buttons
    tabbuttons = parentTab.getElementsByClassName("nested-tab-button");
    for (i = 0; i < tabbuttons.length; i++) {
        tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
    }
    
    // Show the current nested tab and add "active" class to the button
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}


function copyTextToClipboard(elementId, event) {
    console.log("Citation button clicked for: " + elementId);
    
    // Get the citation text
    var citationText = document.getElementById(elementId);
    if (!citationText) {
        console.error("Citation element not found: " + elementId);
        return;
    }
    
    // Force create popup if not exists
    var popup = document.getElementById('citation-popup');
    if (!popup) {
        console.log("Creating popup because it doesn't exist yet");
        var popupHTML = 
            '<div class="citation-popup" id="citation-popup">' +
                '<div class="popup-header">' +
                    '<h3>Citation</h3>' +
                    '<span class="close-popup">&times;</span>' +
                '</div>' +
                '<div class="citation-box">' +
                    '<pre id="citation-text"></pre>' +
                '</div>' +
                '<button id="copy-citation-button" class="copy-button">Copy to Clipboard</button>' +
            '</div>';
        
        document.body.insertAdjacentHTML('beforeend', popupHTML);
        popup = document.getElementById('citation-popup');
        
        // Set up event handlers for the newly created popup
        var closeButton = document.querySelector('.close-popup');
        var copyButton = document.getElementById('copy-citation-button');
        
        if (closeButton) {
            closeButton.onclick = function() {
                popup.style.display = 'none';
            };
        }
        
        if (copyButton) {
            copyButton.onclick = function() {
                var text = document.getElementById('citation-text').innerText;
                navigator.clipboard.writeText(text).then(function() {
                    copyButton.innerText = 'Copied!';
                    setTimeout(function() {
                        copyButton.innerText = 'Copy to Clipboard';
                    }, 1500);
                });
            };
        }
    }
    
    // Set the citation text in the popup
    var citationTextElement = document.getElementById('citation-text');
    if (citationTextElement) {
        citationTextElement.innerText = citationText.innerText;
    }
    
    // Position the popup near the mouse cursor instead of the button
    var x = event.clientX;
    var y = event.clientY;
    
    // Get dimensions
    var viewportWidth = window.innerWidth || document.documentElement.clientWidth;
    var viewportHeight = window.innerHeight || document.documentElement.clientHeight;
    var scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    var scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
    
    // Show the popup temporarily to get its dimensions
    popup.style.visibility = 'hidden';
    popup.style.display = 'block';
    var popupWidth = popup.offsetWidth;
    var popupHeight = popup.offsetHeight;
    
    // Calculate position to ensure popup stays in viewport
    // Add 10px padding from edges
    var padding = 10;
    
    // Position horizontally
    if (x + popupWidth + padding > viewportWidth) {
        // If too far right, position to the left of cursor
        x = Math.max(padding, x - popupWidth - padding);
    } else {
        // Otherwise position to the right of cursor with padding
        x = x + padding;
    }
    
    // Position vertically
    if (y + popupHeight + padding > viewportHeight) {
        // If too far down, position above cursor
        y = Math.max(padding, y - popupHeight - padding);
    } else {
        // Otherwise position below cursor with padding
        y = y + padding;
    }
    
    // Apply the position (convert from viewport coordinates to document coordinates)
    popup.style.left = (x + scrollLeft) + 'px';
    popup.style.top = (y + scrollTop) + 'px';
    
    // Make the popup visible
    popup.style.visibility = 'visible';
    
    // Prevent default action and event bubbling
    event.preventDefault();
    event.stopPropagation();
}

// Initialize tabs and set up the citation popup
document.addEventListener('DOMContentLoaded', function() {
    // Make sure the first tab and its first nested tab are active by default
    var firstTabButton = document.querySelector('.tab-button');
    if (firstTabButton) {
        firstTabButton.click();
    }
    
    // Create the citation popup HTML if it doesn't exist
    if (!document.getElementById('citation-popup')) {
        var popupHTML = 
            '<div class="citation-popup" id="citation-popup">' +
                '<div class="popup-header">' +
                    '<h3>Citation</h3>' +
                    '<span class="close-popup">&times;</span>' +
                '</div>' +
                '<div class="citation-box">' +
                    '<pre id="citation-text"></pre>' +
                '</div>' +
                '<button id="copy-citation-button" class="copy-button">Copy to Clipboard</button>' +
            '</div>';
        
        document.body.insertAdjacentHTML('beforeend', popupHTML);
        
        // Now set up the event handlers for the popup
        var popup = document.getElementById('citation-popup');
        var closeButton = document.querySelector('.close-popup');
        var copyButton = document.getElementById('copy-citation-button');
        
        // Close popup when clicking the close button
        if (closeButton) {
            closeButton.onclick = function() {
                popup.style.display = 'none';
            };
        }
        
        // When the user clicks the copy button
        if (copyButton) {
            copyButton.onclick = function() {
                var text = document.getElementById('citation-text').innerText;
                navigator.clipboard.writeText(text).then(function() {
                    // Change button text temporarily to provide feedback
                    var originalText = copyButton.innerText;
                    copyButton.innerText = 'Copied!';
                    setTimeout(function() {
                        copyButton.innerText = originalText;
                    }, 1500);
                }).catch(function(err) {
                    console.error('Could not copy text: ', err);
                });
            };
        }
        
        // Close popup when clicking outside
        document.addEventListener('click', function(event) {
            if (popup && 
                !popup.contains(event.target) && 
                !event.target.classList.contains('cite-button') && 
                popup.style.display === 'block') {
                popup.style.display = 'none';
            }
        });
    }
});


document.addEventListener('DOMContentLoaded', function() {
    // Find the simplified table by its container
    const simplifiedTableContainer = document.getElementById('tab6-subtab1');
    if (simplifiedTableContainer) {
      const table = simplifiedTableContainer.querySelector('table');
      if (table) {
        table.classList.add('simplified-scores');
      }
    }
  });