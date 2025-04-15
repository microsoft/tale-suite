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

// Initialize tabs
document.addEventListener('DOMContentLoaded', function() {
    // Make sure the first tab and its first nested tab are active by default
    document.querySelector('.tab-button').click();
});