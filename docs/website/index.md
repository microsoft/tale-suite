---
layout: default
title: <strong><em>T A L E S</em>
description: "<em>the</em> <strong><em>T</em></strong><em>ext</em> <strong><em>A</em></strong><em>dventure</em> <strong><em>L</em></strong><em>earning</em> <strong><em>E</em></strong><em>nvironment</em> <strong><em>S</em></strong><em>uite</em>"
---
<div class="tab-container">
    <div class="tabs">
        <button class="tab-button active" onclick="openTab(event, 'tab1')"><strong><em>TALES</em></strong></button>
        <button class="tab-button" onclick="openTab(event, 'tab4')">Environments</button>
        <button class="tab-button" onclick="openTab(event, 'tab3')">Scores By Framework</button>
        <button class="tab-button" onclick="openTab(event, 'tab2')">Scores By Game</button>
        <button class="tab-button" onclick="openTab(event, 'tab6')">All LLM Scores</button>
        <!-- <button class="tab-button" onclick="openTab(event, 'tab5')">Bloopers</button> -->
    </div>
    
<div id="tab1" class="tab-content active">
    <!-- Content for main tab without nested tabs -->
    <p class="author-tagline"><em>Anonymized for review</em></p>
    <p><img src="assets/figs/figure1_eric.png" alt="figure1" /></p>
    <p id="main-description">An example of an agent playing through a text-adventure game. Thought traces are fabricated for illustration. For text-adventure games, due to their length and the variety of puzzles required for progression, players must make use of a range of different reasoning skills to solve roadblocks and continue through the game. Because of the long-range, causal dependencies often found in these games, a single mistake at any step can lead to a breakdown in gameplay later on.</p>
        <div class="abstract-container">
    <p class="abstract-tagline">Abstract</p>
    <p>Reasoning is an essential skill to enable Large Language Models (LLMs) to interact with the world. As tasks become more complex, they demand increasingly sophisticated and diverse reasoning capabilities for sequential decision-making, requiring structured reasoning over the context history to determine the next best action.
    We introduce <strong><em>TALES</em></strong>, a diverse collection of synthetic and human-written text-adventure games designed to challenge and evaluate diverse reasoning capabilities.
    We present results over a range of LLMs, open- and closed-weights, performing a qualitative analysis on the top performing models.
    Despite an impressive showing on synthetic games, even the top LLM-driven agents fail to achieve 10% on games designed for human enjoyment</p>
</div>
</div>

<div id="tab2" class="tab-content">
<!-- Nested tabs for tab2 -->
<div class="nested-tabs">
    <button class="nested-tab-button active" onclick="openNestedTab(event, 'tab2-subtab1')">Textworld</button>
    <button class="nested-tab-button" onclick="openNestedTab(event, 'tab2-subtab2')">Textworld Express</button>
    <button class="nested-tab-button" onclick="openNestedTab(event, 'tab2-subtab3')">Alfworld</button>
    <button class="nested-tab-button" onclick="openNestedTab(event, 'tab2-subtab4')">Scienceworld</button>
    <button class="nested-tab-button" onclick="openNestedTab(event, 'tab2-subtab5')">Jericho</button>
</div>

<div id="tab2-subtab1" class="nested-tab-content active">
    <h2 id="tw_all_games">Scores for all Textworld games for Top 9 models</h2>
    <p><img src="assets/figs/textworld_all_games.png" alt="tw_allgames chart" /></p>
</div>

<div id="tab2-subtab2" class="nested-tab-content">
    <h2 id="twx_all_games">Scores for all Textworld Express games for Top 9 models</h2>
    <p><img src="assets/figs/textworld_express_all_games.png" alt="twx_allgames chart" /></p>
</div>

<div id="tab2-subtab3" class="nested-tab-content">
    <h2 id="alfworld_all_games">Scores for all Alfworld games for Top 9 models</h2>
    <p><img src="assets/figs/alfworld_all_games.png" alt="alfw_allgames chart" /></p>
</div>

<div id="tab2-subtab4" class="nested-tab-content">
    <h2 id="scienceworld_all_games">Scores for all Scienceworld games for Top 9 models</h2>
    <p><img src="assets/figs/scienceworld_all_games.png" alt="sciencew_allgames chart" /></p>
</div>

<div id="tab2-subtab5" class="nested-tab-content">
    <h2 id="jericho_all_games">Scores for all Jericho games for Top 9 models</h2>
    <p><img src="assets/figs/jericho_all_games.png" alt="jericho_allgames chart" /></p>
</div>
</div>

<div id="tab3" class="tab-content">
    <!-- Insert Tab 3 content here -->
    <h2>Breakdown of scores per framework</h2>
    <p><img src="assets/figs/all_framework_scores.png" alt="fws chart" /></p>
</div>

<div id="tab4" class="tab-content">
    <div class="nested-tabs">
        <button class="nested-tab-button active" onclick="openNestedTab(event, 'tab4-subtab1')">Textworld</button>
        <button class="nested-tab-button" onclick="openNestedTab(event, 'tab4-subtab2')">Textworld Express</button>
        <button class="nested-tab-button" onclick="openNestedTab(event, 'tab4-subtab3')">Alfworld</button>
        <button class="nested-tab-button" onclick="openNestedTab(event, 'tab4-subtab4')">Scienceworld</button>
        <button class="nested-tab-button" onclick="openNestedTab(event, 'tab4-subtab5')">Jericho</button>
    </div>
    
<div id="tab4-subtab1" class="nested-tab-content active">
    <div class="environment-container">
        <p class="environment-tagline">TextWorld</p>
        <p>TextWorld is a framework originally designed for training agents with Reinforcement Learning on text-based games. It can generate synthetic text-adventure games of varying complexity. In TALES, we integrate the "CookingWorld" games that were used as part of the <a href="https://competitions.codalab.org/competitions/21557">NeurIPS 2018 Competition</a>. The task involves following a recipe that requires finding ingredients and processing them according to said recipe. We selected one game per difficulty ranging from level 1 (with one location and a recipe of 1 ingredient) to level 10 (having 12 locations and a recipe with 3 ingredients). For all difficulties, the player receives 1 point after completing sub-goals related to the task in the game. Difficulty level 1 can be solved in 7 moves with a max score of 3, while level 10 requires 44 moves with a max score of 11.</p>
    </div>
</div>

<div id="tab4-subtab2" class="nested-tab-content">
    <div class="environment-container">
        <p class="environment-tagline">Textworld Express</p>
        <p>Textworld Express is a highly optimized re-implementation of many Textworld game scenarios that runs approximately three orders of magnitudes faster compared to the Textworld counterparts. 
        While token throughput is the major speed bottleneck in many LLM-based applications, we opt to use Textworld Express over Textworld for the performance improvement where applicable. 

        While significantly faster, an arguable drawback of using Textworld Express over Textworld is also in its stricter parser. 
        Textworld Express simplifies its parser for speed and thus does not allow for nearest-neighbor action phrases.</p>
    </div>
</div>

<div id="tab4-subtab3" class="nested-tab-content">
    <div class="environment-container">
        <p class="environment-tagline">Alfworld</p>
        <p>Alfworld is a multi-modal framework combining complementary visual and textual observations, where agents are asked to navigate and perform tasks in a household setting. All tasks provide only a terminal reward of 1 upon task completion. For <strong><em>TALES</em></strong>, we only use its textual modality as it has become the standard in the LLM literature when evaluated on.</p>
        
        <p>The Alfworld environments are unique in their lack of informative feedback.
        Where other environments have a predefined error message relating to the type of error, whether it is due to the parser not recognizing the command or the action not being possible, Alfworld has only one error message in the form of 'Nothing happened'. 
        In the original Alfworld framework, the visual component compensates for the insufficient text feedback. However, this lack of detailed information significantly increases the difficulty for agents that rely solely on text-based interactions.
        This difficulty is compounded upon by the limitation that an agent in Alfworld can only hold one object at a time. </p>
    </div>
</div>

<div id="tab4-subtab4" class="nested-tab-content">
    <div class="environment-container">
        <p class="environment-tagline">Scienceworld</p>
        <p>Scienceworld is a framework focused on the completion of elementary-level science curriculum tasks. 
        Notably for many of its tasks, Scienceworld emulates an open-world setting where the player can complete the task in different ways that do not follow one expected trajectory. 
        When it comes to heating objects, this part of the task can be completed by either the oven in the kitchen or the blast furnace in the workshop. 
        Similarly, Scienceworld also allows the player the freedom to reset the game on command.
        This is especially important as a number of Scienceworld games have dead states where it is no longer possible to complete the assigned task in that play-through.</p>
    </div>
</div>

<div id="tab4-subtab5" class="nested-tab-content">
    <div class="environment-container">
        <p class="environment-tagline">Jericho</p>
        <p>Jericho is a suite of 55 human-written, interactive fiction games. 
        We consider Jericho to be the most difficult framework due to the length and complexity of many of the games. Some can be completed within 17 steps while some others require over 500 steps.
        Those games also cover an extremely wide range of genres and styles and lack the consistency of many other text-game environment suites designed for evaluating agents.
        For example, '9:05' follows the morning of an ordinary office worker where 'Anchorhead' is a Lovecraftian Horror Story.</p>
    </div>
</div>
</div>

<div id="tab5" class="tab-content">
    <!-- Insert Tab 5 content here -->
    <h2></h2>
    <p>Funny LLM Fails to be added here.</p>
</div>

<div id="tab6" class="tab-content">
    <div class="table-container">
    <p>All LLM Scores can be found below. For the graphs in other visualizations, we used only the top 9 models. All scores are displayed as percentages.</p>
    <table class="model-scores">
        <thead>
        <tr>
            <th>Model</th>
            <th>Textworld</th>
            <th>Textworld Express</th>
            <th>Alfworld</th>
            <th>Scienceworld</th>
            <th>Jericho</th>
            <th>Overall</th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td><strong>claude-3.7-sonnet</strong></td>
            <td>97.3%</td>
            <td>91.3%</td>
            <td>83.3%</td>
            <td>76.5%</td>
            <td>12.3%</td>
            <td>52.1%</td>
        </tr>
        <tr>
            <td>claude-3.5-sonnet-latest</td>
            <td>95.5%</td>
            <td>81.6%</td>
            <td>75.0%</td>
            <td>82.3%</td>
            <td>9.5%</td>
            <td>50.0%</td>
        </tr>
        <tr>
            <td>o1</td>
            <td>97.8%</td>
            <td>70.0%</td>
            <td>28.3%</td>
            <td>80.1%</td>
            <td>10.6%</td>
            <td>44.1%</td>
        </tr>
        <tr>
            <td>gpt-4o</td>
            <td>83.6%</td>
            <td>80.6%</td>
            <td>56.7%</td>
            <td>61.4%</td>
            <td>5.5%</td>
            <td>40.2%</td>
        </tr>
        <tr>
            <td>claude-3.5-haiku</td>
            <td>94.9%</td>
            <td>79.8%</td>
            <td>26.7%</td>
            <td>67.3%</td>
            <td>4.9%</td>
            <td>39.3%</td>
        </tr>
        <tr>
            <td>Llama-3.1-405B-Instruct</td>
            <td>90.9%</td>
            <td>79.2%</td>
            <td>31.7%</td>
            <td>51.8%</td>
            <td>6.0%</td>
            <td>36.1%</td>
        </tr>
        <tr>
            <td>Llama-3.3-70B-Instruct</td>
            <td>69.6%</td>
            <td>77.2%</td>
            <td>15.0%</td>
            <td>55.1%</td>
            <td>4.4%</td>
            <td>32.6%</td>
        </tr>
        <tr>
            <td>Llama-3.1-70B-Instruct</td>
            <td>65.6%</td>
            <td>81.9%</td>
            <td>8.3%</td>
            <td>51.9%</td>
            <td>5.2%</td>
            <td>31.8%</td>
        </tr>
        <tr>
            <td>Qwen2.5-72B-Instruct</td>
            <td>76.5%</td>
            <td>83.8%</td>
            <td>36.7%</td>
            <td>35.0%</td>
            <td>2.8%</td>
            <td>30.5%</td>
        </tr>
        <tr>
            <td>Mistral-Large-Instruct-2407</td>
            <td>82.4%</td>
            <td>68.3%</td>
            <td>6.7%</td>
            <td>46.1%</td>
            <td>5.7%</td>
            <td>30.0%</td>
        </tr>
        <tr>
            <td>gpt-4o-mini</td>
            <td>56.5%</td>
            <td>73.6%</td>
            <td>0.0%</td>
            <td>27.2%</td>
            <td>1.7%</td>
            <td>21.6%</td>
        </tr>
        <tr>
            <td>Llama-4-Scout-17B-16E-Instruct</td>
            <td>41.1%</td>
            <td>68.4%</td>
            <td>0.0%</td>
            <td>27.0%</td>
            <td>1.8%</td>
            <td>19.6%</td>
        </tr>
        <tr>
            <td>Llama-4-Maverick-17B-128E-Instruct-FP8</td>
            <td>43.5%</td>
            <td>56.1%</td>
            <td>8.3%</td>
            <td>11.5%</td>
            <td>2.0%</td>
            <td>15.3%</td>
        </tr>
        <tr>
            <td>Mistral-Small-Instruct-2409</td>
            <td>56.1%</td>
            <td>27.3%</td>
            <td>0.0%</td>
            <td>24.4%</td>
            <td>1.4%</td>
            <td>14.7%</td>
        </tr>
        <tr>
            <td>Llama-3.1-8B-Instruct</td>
            <td>29.7%</td>
            <td>50.3%</td>
            <td>0.0%</td>
            <td>15.7%</td>
            <td>2.2%</td>
            <td>13.8%</td>
        </tr>
        <tr>
            <td>Qwen2.5-7B-Instruct</td>
            <td>27.7%</td>
            <td>45.6%</td>
            <td>0.0%</td>
            <td>12.6%</td>
            <td>0.7%</td>
            <td>11.6%</td>
        </tr>
        <tr>
            <td>Llama-3.2-3B-Instruct</td>
            <td>21.4%</td>
            <td>42.0%</td>
            <td>0.0%</td>
            <td>10.0%</td>
            <td>1.5%</td>
            <td>10.3%</td>
        </tr>
        <tr>
            <td>phi-4</td>
            <td>20.8%</td>
            <td>43.8%</td>
            <td>0.0%</td>
            <td>8.9%</td>
            <td>1.5%</td>
            <td>10.2%</td>
        </tr>
        <tr>
            <td>Mistral-Small-24B-Instruct-2501</td>
            <td>15.8%</td>
            <td>23.0%</td>
            <td>0.0%</td>
            <td>15.8%</td>
            <td>1.4%</td>
            <td>8.7%</td>
        </tr>
        <tr>
            <td>DeepSeek-R1-Distill-Llama-70B</td>
            <td>8.7%</td>
            <td>39.8%</td>
            <td>0.0%</td>
            <td>7.7%</td>
            <td>1.3%</td>
            <td>8.4%</td>
        </tr>
        <tr>
            <td>Ministral-8B-Instruct-2410</td>
            <td>10.9%</td>
            <td>22.8%</td>
            <td>0.0%</td>
            <td>2.3%</td>
            <td>0.4%</td>
            <td>4.6%</td>
        </tr>
        <tr>
            <td>Mistral-Small-3.1-24B-Instruct-2503</td>
            <td>2.5%</td>
            <td>10.3%</td>
            <td>0.0%</td>
            <td>10.5%</td>
            <td>0.8%</td>
            <td>4.5%</td>
        </tr>
        <tr>
            <td>Mixtral-8x22B-Instruct-v0.1</td>
            <td>17.1%</td>
            <td>8.4%</td>
            <td>0.0%</td>
            <td>4.0%</td>
            <td>0.4%</td>
            <td>3.7%</td>
        </tr>
        <tr>
            <td>Llama-3.2-1B-Instruct</td>
            <td>0.0%</td>
            <td>19.0%</td>
            <td>0.0%</td>
            <td>2.4%</td>
            <td>0.5%</td>
            <td>3.3%</td>
        </tr>
        <tr>
            <td>Phi-3-mini-128k-instruct</td>
            <td>2.7%</td>
            <td>9.4%</td>
            <td>0.0%</td>
            <td>2.4%</td>
            <td>0.3%</td>
            <td>2.2%</td>
        </tr>
        <tr>
            <td>Phi-3.5-MoE-instruct</td>
            <td>0.0%</td>
            <td>7.0%</td>
            <td>0.0%</td>
            <td>2.3%</td>
            <td>0.4%</td>
            <td>1.7%</td>
        </tr>
        <tr>
            <td>Phi-4-mini-instruct</td>
            <td>0.0%</td>
            <td>5.5%</td>
            <td>0.0%</td>
            <td>2.3%</td>
            <td>0.5%</td>
            <td>1.5%</td>
        </tr>
        <tr>
            <td>Mixtral-8x7B-Instruct-v0.1</td>
            <td>0.0%</td>
            <td>1.6%</td>
            <td>0.0%</td>
            <td>4.0%</td>
            <td>0.3%</td>
            <td>1.3%</td>
        </tr>
        <tr>
            <td>Phi-3.5-mini-instruct</td>
            <td>0.0%</td>
            <td>2.0%</td>
            <td>0.0%</td>
            <td>2.4%</td>
            <td>0.5%</td>
            <td>1.0%</td>
        </tr>
        <tr>
            <td>Phi-3-medium-128k-instruct</td>
            <td>0.0%</td>
            <td>0.0%</td>
            <td>0.0%</td>
            <td>2.3%</td>
            <td>0.3%</td>
            <td>0.7%</td>
        </tr>
        </tbody>
    </table>
    </div>
</div>
</div>