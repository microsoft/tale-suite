---
layout: default
title: <strong><em>T A L E S</em>
description: "<em>the</em> <strong><em>T</em></strong><em>ext</em> <strong><em>A</em></strong><em>dventure</em> <strong><em>L</em></strong><em>earning</em> <strong><em>E</em></strong><em>nvironment</em> <strong><em>S</em></strong><em>uite</em>"
---
<div class="tab-container">
    <div class="tabs">
        <button class="tab-button active" onclick="openTab(event, 'tab1')">Overview</button>
        <button class="tab-button" onclick="openTab(event, 'tab4')">Environments</button>
        <button class="tab-button" onclick="openTab(event, 'tab3')">Scores By Framework</button>
        <button class="tab-button" onclick="openTab(event, 'tab2')">Scores By Game</button>
        <button class="tab-button" onclick="openTab(event, 'tab5')">Bloopers</button>
    </div>
    
    <div id="tab1" class="tab-content active">
        <!-- Nested tabs for tab1 -->
        <div class="nested-tabs">
            <button class="nested-tab-button active" onclick="openNestedTab(event, 'tab1-subtab1')">Overview</button>
            <button class="nested-tab-button" onclick="openNestedTab(event, 'tab1-subtab2')">Environments</button>
        </div>
        
        <div id="tab1-subtab1" class="nested-tab-content active">
            <h2 id="overview">Overview</h2>
            Insert overview description here.
        </div>
        
        <div id="tab1-subtab2" class="nested-tab-content">
            <h2 id="Environment Description">Environments</h2>
            
        </div>
    </div>
    
    <div id="tab2" class="tab-content">
        <!-- Nested tabs for tab2 -->
        <div class="nested-tabs">
            <button class="nested-tab-button active" onclick="openNestedTab(event, 'tab2-subtab1')">Textworld</button>
            <button class="nested-tab-button" onclick="openNestedTab(event, 'tab2-subtab2')">Textworld Express</button>
            <button class="nested-tab-button" onclick="openNestedTab(event, 'tab3-subtab3')">Alfworld</button>
            <button class="nested-tab-button" onclick="openNestedTab(event, 'tab4-subtab4')">Scienceworld</button>
            <button class="nested-tab-button" onclick="openNestedTab(event, 'tab5-subtab5')">Jericho</button>
        </div>
        
        <div id="tab1-subtab1" class="nested-tab-content active">
            <h2 id="tw_all_games">Scores for all Textworld games for Top 9 models</h2>
            <p><img src="assets/figs/textworld_all_games.png" alt="tw_allgames chart" /></p>
        </div>

        <div id="tab2-subtab2" class="nested-tab-content active">
            <h2 id="twx_all_games">Scores for all Textworld Express games for Top 9 models</h2>
            <p><img src="assets/figs/textworld_express_all_games.png" alt="twx_allgames chart" /></p>
        </div>

        <div id="tab3-subtab3" class="nested-tab-content active">
            <h2 id="alfworld_all_games">Scores for all Alfworld games for Top 9 models</h2>
            <p><img src="assets/figs/alfworld_all_games.png" alt="alfw_allgames chart" /></p>
        </div>

        <div id="tab4-subtab4" class="nested-tab-content active">
            <h2 id="scienceworld_all_games">Scores for all Scienceworld games for Top 9 models</h2>
            <p><img src="assets/figs/scienceworld_all_games.png" alt="sciencew_allgames chart" /></p>
        </div>

        <div id="tab5-subtab5" class="nested-tab-content active">
            <h2 id="jericho_all_games">Scores for all Jericho games for Top 9 models</h2>
            <p><img src="assets/figs/jericho_all_games.png" alt="jericho_allgames chart" /></p>
        </div>
        
        <div id="tab1-subtab1" class="nested-tab-content active">
            <h2 id="jericho_all_games">Scores for all Jericho games for Top 9 models</h2>
            <p><img src="assets/figs/jericho_all_games.png" alt="jerichoallgames chart" /></p>
        </div>
    </div>
    
    <div id="tab3" class="tab-content">
        <!-- Insert Tab 3 content here -->
        <h2>Breakdown of scores per framework</h2>
        <p><img src="assets/figs/all_framework_scores.png" alt="fws chart" /></p>
    </div>
    
    <div id="tab4" class="tab-content">
        <!-- Insert Tab 4 content here -->
        <h2>Tab 4 Content</h2>
        <p>This is where you'll put the content for Tab 4.</p>
    </div>
    
    <div id="tab5" class="tab-content">
        <!-- Insert Tab 5 content here -->
        <h2>Tab 5 Content</h2>
        <p>This is where you'll put the content for Tab 5.</p>
    </div>
</div>