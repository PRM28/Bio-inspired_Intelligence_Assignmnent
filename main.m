clc;
clear;
rng(42);

%% Initialize Clock
% tic;  % Start timing the training process
load('trainedQAgent.mat', 'qAgent3', 'qtable3');
%% Create a Random MDP environment
gridx = 50;
gridy = 50;
GW = createGridWorld(gridx, gridy);

terminalState = sprintf('[%d,%d]', randi(gridx, 1), randi(gridy, 1));
GW.TerminalStates = {terminalState};
ns = numel(GW.States);
na = numel(GW.Actions);

% Update Reward Matrix
min_reward = -10;
max_reward = -1;
step_penalty = -0.1;  % New step penalty
terminal_reward = 50; % Increased terminal reward
max_m_dist = (gridx - 1) + (gridy - 1);

for stateIdx = 1:ns
    [x, y] = ind2sub([gridx, gridy], stateIdx); 
    terminal_coordinates = str2num(GW.TerminalStates{1});
    x_fin = terminal_coordinates(1);
    y_fin = terminal_coordinates(2);

    m_dist = abs(x_fin - x) + abs(y_fin - y);
    reward = min_reward + ((max_reward - min_reward) * (max_m_dist - m_dist) / max_m_dist) + step_penalty;
    GW.R(:, stateIdx, :) = reward;

    if m_dist == 1
        GW.R(stateIdx, state2idx(GW, GW.TerminalStates), :) = terminal_reward;
    end
end

% Defining Transition Matrix
% Defining Transition Matrix
GW.T = zeros(ns, ns, na);

% Update Transition Matrix
for stateIdx = 1:ns
    [x, y] = ind2sub([gridx, gridy], stateIdx); % Get the coordinates from the index

    if x > 1
        nextStateIdx = state2idx(GW, sprintf('[%d,%d]', x - 1, y));
        GW.T(stateIdx, nextStateIdx, 1) = 1; % Move North (up)
    end

    if x == 1
        nextStateIdx = state2idx(GW, sprintf('[%d,%d]', x, y));
        GW.T(stateIdx, nextStateIdx, 1) = 1;
    end

    if x < gridx
        nextStateIdx = state2idx(GW, sprintf('[%d,%d]', x + 1, y));
        GW.T(stateIdx, nextStateIdx, 2) = 1; % Move South (down)
    end

    if x == gridx
        nextStateIdx = state2idx(GW, sprintf('[%d,%d]', x, y));
        GW.T(stateIdx, nextStateIdx, 2) = 1;
    end

    if y < gridy
        nextStateIdx = state2idx(GW, sprintf('[%d,%d]', x, y + 1));
        GW.T(stateIdx, nextStateIdx, 3) = 1; % Move East (right)
    end

    if y == gridy
        nextStateIdx = state2idx(GW, sprintf('[%d,%d]', x, y));
        GW.T(stateIdx, nextStateIdx, 3) = 1;
    end

    if y > 1
        nextStateIdx = state2idx(GW, sprintf('[%d,%d]', x, y - 1));
        GW.T(stateIdx, nextStateIdx, 4) = 1; % Move West (left)
    end

    if y == 1
        nextStateIdx = state2idx(GW, sprintf('[%d,%d]', x, y));
        GW.T(stateIdx, nextStateIdx, 4) = 1;
    end
end
%% Creating environment

env2 = rlMDPEnv(GW);
env2.ResetFcn = @() state2idx(GW, sprintf('[%d,%d]', gridx, 1));

%% Create Q-learning Agent
qtable3 = rlTable(getObservationInfo(env2), getActionInfo(env2));
qtable3.Table = rand([ns, na]);  
qFcnAppr3 = rlQValueFunction(qtable3, getObservationInfo(env2), getActionInfo(env2));
qAgent3 = rlQAgent(qFcnAppr3);

% Adjust exploration and learning rate
qAgent3.AgentOptions.EpsilonGreedyExploration.Epsilon = 1.0;
qAgent3.AgentOptions.EpsilonGreedyExploration.EpsilonMin = 0.01;
qAgent3.AgentOptions.EpsilonGreedyExploration.EpsilonDecay = 0.0005;
qAgent3.AgentOptions.CriticOptimizerOptions.LearnRate = 0.05;

%% Training Agent
trainOpts = rlTrainingOptions;
trainOpts.MaxStepsPerEpisode = 500;
trainOpts.MaxEpisodes = 500;  % Increased number of episodes
trainOpts.StopTrainingCriteria = 'AverageReward';
trainOpts.StopTrainingValue = 50;
trainOpts.ScoreAveragingWindowLength = 50;
trainOpts.ParallelizationOptions.Mode = 'async';

% randomize terminal state params:
ep_rand_ts = 250;
list_ints = [];
rat_rand = trainOpts.MaxEpisodes / ep_rand_ts;
for i = 1:rat_rand
    list_ints  = [list_ints; i];
end


%Enabling training or disabling it
doTraining = true;

% Training Debugging

debugInfo = struct('Episode', [], 'Step', [], 'Action', [], 'State', [], 'Reward', []);
opt_sol = struct('Episode', [], 'Step', [], 'Action', [], 'State', [], 'Reward', [0]);
min_steps = inf;  % Initialize the minimum steps to a very high number

% Train Agent
if doTraining
    tot_reward = zeros([1, trainOpts.MaxEpisodes]);
    for episode = 1:trainOpts.MaxEpisodes
        
        % Randomize terminal state
        for i = list_ints
            if (episode/ep_rand_ts) == i
                env2.Model.TerminalStates = {sprintf('[%d,%d]', randi(gridx, 1), randi(gridy, 1))};
            end
        end

        state = reset(env2);  % Ensure state is numeric
        current_reward = 0;
        isDone = false;
        stepcount = 0;

        while ~isDone && stepcount < trainOpts.MaxStepsPerEpisode
            stepcount = stepcount + 1;

            % Obtaining Action info from agent
            if rand() > qAgent.AgentOptions.EpsilonGreedyExploration.Epsilon
                % Choose the action with the maximum Q-value
                [~, action] = max(qtable.Table(state, :));
                
            else
               % Choose a random action
                action = randi([1, 4], 1); % Random action between 1 and 4 
            end

            % Check transition probability
            nextStateIdx = find(GW.T(state, :, action));
            if ~isempty(nextStateIdx)
                actionValid = true;
            else
                actionValid = false;
            end

            if actionValid
                % Take step in environment
                [nextState, reward, isDone] = step(env2, action);

                % Update episode reward
                current_reward = current_reward + reward;

                % Update Q-table
                maximum_future_q = max(qtable.Table(nextState, :));
                current_q = qtable.Table(state, action);
                new_q = (1-qAgent.AgentOptions.CriticOptimizerOptions.LearnRate)*current_q + ...
                    qAgent.AgentOptions.CriticOptimizerOptions.LearnRate*(reward + qAgent.AgentOptions.DiscountFactor*maximum_future_q);
                qtable.Table(state, action) = new_q;

                % Log the relevant params for debugging and for plotting
                debugInfo.Action = [debugInfo.Action; action];
                debugInfo.State = [debugInfo.State; state];
                debugInfo.Reward = [debugInfo.Reward; reward];
                debugInfo.Step = [debugInfo.Step; stepcount];
                debugInfo.Episode = [debugInfo.Episode; episode];

                % Visualize the training process
                fprintf('Episode: %d, Step: %d, State: %d, Action: %d, Reward: %.2f, Next State: %d\n', ...
                episode, stepcount, state, action, current_reward, nextState);

                if isDone || state == state2idx(env2.Model, env2.Model.TerminalStates)
                    qtable.Table(state, action) = 0;  % Set Q-value to 0 for terminal state
                    isDone = true;
                end

                % Update state
                state = nextState;
            end
        end
        tot_reward(episode) = current_reward;

        % Track the minimum steps to reach the terminal state
        if isDone && stepcount < min_steps
            min_steps = stepcount;
            opt_sol.Reward = current_reward;
            opt_sol.Episode = episode;
            opt_sol.Step = debugInfo.Step;
            opt_sol.Action =  debugInfo.Action;
            opt_sol.State = debugInfo.State;
        end

        % Decaying Exploration 
        if qAgent.AgentOptions.EpsilonGreedyExploration.Epsilon > qAgent.AgentOptions.EpsilonGreedyExploration.EpsilonMin && episode < (4/5) * trainOpts.MaxEpisodes
           qAgent.AgentOptions.EpsilonGreedyExploration.Epsilon = qAgent.AgentOptions.EpsilonGreedyExploration.Epsilon - qAgent.AgentOptions.EpsilonGreedyExploration.EpsilonDecay;
        else 
           qAgent.AgentOptions.EpsilonGreedyExploration.Epsilon = qAgent.AgentOptions.EpsilonGreedyExploration.EpsilonMin; 
        end
    end
    
    % Save the trained agent
    save('trainedQAgent3.mat', 'qAgent2', 'qtable2');
else
    load('trainedQAgent3.mat');
end

% Calculate and display the total training time
training_time = toc;  % End timing the training process
fprintf('Total training time: %.2f seconds\n', training_time);

%% Creating an Environment for testing

% Create a Random MDP environment
gridx_test = 5;
gridy_test = 5;
GW_test = createGridWorld(gridx_test, gridy_test);

terminalState_test = sprintf('[%d,%d]', randi(gridx_test, 1), randi(gridy_test, 1));
GW_test.TerminalStates = {terminalState_test};
ns_test = numel(GW_test.States);
na_test = numel(GW_test.Actions);

% Update Reward Matrix
min_reward_test = -10;
max_reward_test = -1;
step_penalty_test = -0.1;  % New step penalty
terminal_reward_test = 50; % Increased terminal reward
max_m_dist_test = (gridx_test - 1) + (gridy_test - 1);

for stateIdx = 1:ns_test
    [x_test, y_test] = ind2sub([gridx_test, gridy_test], stateIdx); 
    terminal_coordinates_test = str2num(GW_test.TerminalStates{1});
    x_fin_test = terminal_coordinates_test(1);
    y_fin_test = terminal_coordinates_test(2);

    m_dist_test = abs(x_fin_test - x_test) + abs(y_fin_test - y_test);
    reward_test = min_reward_test + ((max_reward_test - min_reward_test) * (max_m_dist_test - m_dist_test) / max_m_dist_test) + step_penalty_test;
    GW_test.R(:, stateIdx, :) = reward_test;

    if m_dist_test == 1
        GW_test.R(stateIdx, state2idx(GW_test, GW_test.TerminalStates), :) = terminal_reward_test;
    end
end

% Defining Transition Matrix
GW_test.T = zeros(ns_test, ns_test, na_test);

% Update Transition Matrix
for stateIdx = 1:ns_test
    [x_test, y_test] = ind2sub([gridx_test, gridy_test], stateIdx); % Get the coordinates from the index

    if x_test > 1
        nextStateIdx = state2idx(GW_test, sprintf('[%d,%d]', x_test - 1, y_test));
        GW_test.T(stateIdx, nextStateIdx, 1) = 1; % Move North (up)
    end

    if x_test == 1
        nextStateIdx = state2idx(GW_test, sprintf('[%d,%d]', x_test, y_test));
        GW_test.T(stateIdx, nextStateIdx, 1) = 1;
    end

    if x_test < gridx_test
        nextStateIdx = state2idx(GW_test, sprintf('[%d,%d]', x_test + 1, y_test));
        GW_test.T(stateIdx, nextStateIdx, 2) = 1; % Move South (down)
    end

    if x_test == gridx_test
        nextStateIdx = state2idx(GW_test, sprintf('[%d,%d]', x_test, y_test));
        GW_test.T(stateIdx, nextStateIdx, 2) = 1;
    end

    if y_test < gridy_test
        nextStateIdx = state2idx(GW_test, sprintf('[%d,%d]', x_test, y_test + 1));
        GW_test.T(stateIdx, nextStateIdx, 3) = 1; % Move East (right)
    end

    if y_test == gridy_test
        nextStateIdx = state2idx(GW_test, sprintf('[%d,%d]', x_test, y_test));
        GW_test.T(stateIdx, nextStateIdx, 3) = 1;
    end

    if y_test > 1
        nextStateIdx = state2idx(GW_test, sprintf('[%d,%d]', x_test, y_test - 1));
        GW_test.T(stateIdx, nextStateIdx, 4) = 1; % Move West (left)
    end

    if y_test == 1
        nextStateIdx = state2idx(GW_test, sprintf('[%d,%d]', x_test, y_test));
        GW_test.T(stateIdx, nextStateIdx, 4) = 1;
    end
end

%% Show where the agent went for the opt_sol

% Plot the Grid World environment
plot(env2);
env2.Model.Viewer.ShowTrace = true;
env2.Model.Viewer.clearTrace;

% Extract states from opt_sol
visitedStates = opt_sol.State;

% Hold the current plot
hold on;

% Get the grid size
gridSizeX = gridx;
gridSizeY = gridy;

% Loop through the visited states and plot them
for i = 1:length(visitedStates)
    % Get the coordinates of the state
    [row, col] = ind2sub([gridSizeX, gridSizeY], visitedStates(i));
    
    % Adjust the coordinates to match the grid world
    x_coord = col - 0.5; % Adjust column index to match x-coordinate
    y_coord = gridSizeX - row + 0.5; % Adjust row index to match y-coordinate
    
    % Plot the state as a point on the grid
    plot(x_coord, y_coord, 'ro', 'MarkerSize', 3, 'MarkerFaceColor', 'r');
end

% Highlight the start and terminal states
startState = visitedStates(1);
terminalState = visitedStates(end);

% Plot start state
Start = str2num(env2.Model.CurrentState);
x_start = Start(2) - 0.5;
y_start = gridSizeX - Start(1) + 0.5;
plot(x_start, y_start, 'go', 'MarkerSize', 3, 'MarkerFaceColor', 'g');

% Plot terminal state
End =  str2num(env2.Model.TerminalStates);
x_end = End(2) - 0.5;
y_end = gridSizeX - End(2) + 0.5;
plot(x_end, y_end, 'bo', 'MarkerSize', 3, 'MarkerFaceColor', 'b');

% Add labels for clarity
legend({'Visited States', 'Start State', 'Terminal State'}, 'Location', 'northeastoutside');
title('Agent Path Through the Grid World');
xlabel('Grid Column');
ylabel('Grid Row');

% Release the plot hold
hold off;


%% Simulate the Agent's Performance with Random Terminal States
env_test = rlMDPEnv(GW_test);
plot(env_test);
env_test.Model.Viewer.ShowTrace = true;
env_test.Model.Viewer.clearTrace;

num_simulations = 1; % Number of random terminal hestate simulations
simulation_times = zeros(1, num_simulations); % Store time taken for each simulation

for simIdx = 1:num_simulations
    % Reset environment and set a new random terminal state
    env_test.Model.TerminalStates = {sprintf('[%d,%d]', randi(gridx_test, 1), randi(gridy_test, 1))};
    state = reset(env_test);
    
    % Initialize simulation clock
    sim_start_time = tic;
    
    isDone = false;
    while ~isDone
        % Choose the action with the maximum Q-value using getAction
        action = getAction(qAgent2, state);
        
        % Take step in environment
        [nextState, ~, isDone] = step(env_test, action{1});
        
        % Update state
        state = nextState;
        
        if state == state2idx(env_test.Model, env_test.Model.TerminalStates)
            isDone = true;
        end
    end
    
    % Record the time taken for this simulation
    simulation_times(simIdx) = toc(sim_start_time);
    
    fprintf('Simulation %d: Time taken = %.2f seconds\n', simIdx, simulation_times(simIdx));
end

% Calculate and display average simulation time
average_simulation_time = mean(simulation_times);
fprintf('Average time taken to find random target: %.2f seconds\n', average_simulation_time);
