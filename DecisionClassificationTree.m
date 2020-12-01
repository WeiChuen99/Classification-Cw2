function DecisionClassificationTree
    %% data loading and feature selection
    table = readtable('adult.csv');

    %attributes removed due to low predictor importance
    table = removevars(table, [1 2 3 4 9 10 12 13 14]);

    attribute_name = table.Properties.VariableNames;

    %% Label encoding for table
    table.marital_status = double(categorical(table.marital_status));
    table.occupation = double(categorical(table.occupation));
    table.relationship = double(categorical(table.relationship));
    table.census_income = double(categorical(table.census_income));
    table.census_income = table.census_income - 1; %convert values 1,2 to 0,1 for easier processing

    %% Data preparation for cross-validation
    k=10; % k-fold
    sampleSize = 3016; % size of each set divisible by 10
    lastn = 2; 
    table = table(1:end-lastn,:); %remove last 2 rows
    x = table2array(table); %convert table to array
    y = table.census_income;
    [r, ~] = size(x);
    numTestItems = round(r*0.1); %size of test set
    numTrainingItems = r - numTestItems; % leftover to be training set
    dataIndices = randperm(r); % shuffle the dataset 
    shuffled_data = x(dataIndices,:);
    x = shuffled_data;

    %% K-Fold cross validation
    for fold =1:k
        fprintf("Fold %d\n",fold);
        test_indices = 1+(fold-1)*sampleSize:fold*sampleSize;
        train_indices = [1:(fold-1)*sampleSize, fold*sampleSize+1:numTrainingItems];

        %% Training data preparation
        trainingData = x(train_indices,:);
        testData = x(test_indices,:);
        trainingData_x = trainingData(:,1:5);
        trainingData_y = trainingData(:,6);

        %% Tree creation and prediction
        tree = DecisionTreeLearning(trainingData_x,trainingData_y,1, attribute_name);
        tree_name = sprintf('Fold %d classification tree', fold);
        DrawDecisionTree(tree, tree_name); 

        test_x = testData(:,1:5); %test_x = removevars(table, 6);
        prediction = predict(test_x, tree);
        test_y = testData(:,6); %test_y = table.census_income;
        accuracy = evaluation(prediction, test_y);
        precision = precise(prediction, test_y,1);
        recalled = recalling(prediction, test_y,1);
        f1_score = f1_calc(precision, recalled);

        %% Accuracy storing and printing
        fprintf("accuracy = %.2f%%\n",accuracy);
        fprintf("precision = %.2f%%\n",precision);
        fprintf("Recall = %.2f%%\n",recalled);
        fprintf("F1 score = %.2f%%\n",f1_score);
        fprintf('-------------\n');

        accuracy_list(fold) = accuracy; % store accuracy of each fold
        precision_list(fold) = precision; % store accuracy of each fold
        recall_list(fold) = recalled; % store accuracy of each fold
        f1_score_list (fold)= f1_score;
    end
    fprintf("Average accuracy = %.2f%%\n",mean(accuracy_list));
    fprintf("Average precision = %.2f%%\n",mean(precision_list));
    fprintf("Average recall = %.2f%%\n",mean(recall_list));
    fprintf("Average F1 score = %.2f%%\n",mean(f1_score_list));
end


%% functions
function f1_score= f1_calc(precision, recall)
    f1_score =2 /( (1/precision)+ (1/recall));
end
function recalled = recalling(prediction, test_y,sampler)
    correct_prediction = 0;
    total_true_positive = 0;
    for i = 1:length(prediction)
        if test_y(i) == sampler
            total_true_positive = total_true_positive + 1;
            if prediction(i) == test_y(i)
                correct_prediction = correct_prediction + 1;
            end
        end
    end

    recalled = correct_prediction/total_true_positive * 100;
end

function precision = precise(prediction, test_y,sampler)
%calculate precision
    correct_prediction = 0;
    total_positive = 0;
    for i = 1:length(prediction)
        if prediction(i) == sampler
            total_positive = total_positive + 1;
            if prediction(i) == test_y(i)
                correct_prediction = correct_prediction + 1;
            end
        end
    end

    precision = correct_prediction/total_positive * 100;
end

function accuracy = evaluation(prediction, test_y)
%evaluate accuracy
    correct_prediction = 0;
    for i = 1:length(prediction)
        if prediction(i) == test_y(i)
            correct_prediction = correct_prediction + 1;
        end
    end

    accuracy = correct_prediction/length(test_y)*100;
end

function prediction = predict(test_x, tree) %table
%predicts input class
    %table = table2array(table);
    traverse = tree;
    prediction = zeros(length(test_x),1);

    for i = 1:length(test_x)
        while ~strcmp(traverse.op, '')
            if test_x(i, traverse.attribute) <= traverse.threshold
                traverse = traverse.kids{1}; 
            else
                traverse = traverse.kids{2};
            end
        end
        prediction(i) = traverse.class;
        traverse = tree;
    end
end
%%


function [tree] = DecisionTreeLearning(x,y,depth, attribute_name)
    tree = struct('op','','kids',[],'class',[],'attribute',0,'threshold', 0);
    min_gain = 0.075;
    
    [best_gain_attribute, best_gain_threshold, best_gain, left, right]=build_node(x,y);

    y_left=y(left);
    y_right=y(right);
    x_left=x(left,:); 
    x_right=x(right,:);

    %recursion and termination criteria:
    %if true, set current node as leaf node, prediction = majority of
    %class and return
    %else, initialize node variables and recurse
    if best_gain < min_gain || isempty(left) || isempty(right)
        tree.op = '';
        tree.attribute = 0;
        tree.threshold = 0;
        tree.kids = [];
        tree.class = mode(y);
        return;
    else
        tree.op = char(attribute_name{best_gain_attribute});
        tree.attribute = best_gain_attribute;
        tree.threshold = best_gain_threshold;
        tree.kids = cell(1,2);
        depth = depth+1;
        
        %check if pure, set pure side as leaf node and add prediction class
        if length(unique(y_left)) == 1
            tree.kids{1}.op= '';
            tree.kids{1}.kids= [];
            tree.kids{1}.class= mode(y_left);
            tree.kids{1}.attribute= 0;
            tree.kids{1}.threshold= 0;
        else
            tree.kids{1} = DecisionTreeLearning(x_left, y_left, depth, attribute_name);
        end
        if length(unique(y_right)) == 1
            tree.kids{2}.op= '';
            tree.kids{2}.kids= [];
            tree.kids{2}.class= mode(y_right);
            tree.kids{2}.attribute= 0;
            tree.kids{2}.threshold= 0;
        else
            tree.kids{2} = DecisionTreeLearning(x_right, y_right, depth, attribute_name);
        end
        depth = depth-1;
    end
end

function [best_gain_attribute,best_gain_threshold,best_gain,left,right] = build_node(x,y)
%loops through all columns to decide which attribute should be used to split
    [~, x_col] = size(x);
    number_attributes = x_col;
    best_gain_attribute = 1;
    best_gain = 0;
    for i = 1:number_attributes
        [~,best_gain_i,~,~]=split(x(:,i),y);
        if(best_gain_i>best_gain)
            best_gain = best_gain_i;
            best_gain_attribute = i;
        end
    end
    [best_gain_threshold,best_gain,left,right]=split(x(:,best_gain_attribute),y);
end

function [best_gain_threshold,best_gain,left,right] = split(x, y)
%loops through examples(datapoints) and determine best splitting point (threshhold)
    [x_row, ~] = size(x);
    number_examples = x_row;
    best_gain = 0;
    best_gain_threshold = 1;

    left = find(x<=x(1));
    right = find(x>x(1));
    
    x_max = max(x);
    
    x_unique_values = unique(x);
    if length(x_unique_values) == 1 %when attribute only has 1 value (no valid splitting point)
        majority_class = mode(y);
        left = find(y == majority_class); %take mode class as left
        right = find(y ~= majority_class); %take !mode class as right
        return;
    end
    
    %entropy before splitting
    entropy = calculate_entropy(y);

    for j = 1:length(x_unique_values)
        left_j = find(x<=x_unique_values(j));
        right_j = find(x>x_unique_values(j));
        
        if isempty(right_j) %needed when x_unique_values(j) = max(x)
            left_j =  find(x<x_unique_values(j));
            right_j = find(x == x_max);
        end
        
        %calculate entropy of left and right
        [entropy_l, p_sum_l, n_sum_l] = calculate_entropy(y(left_j));
        [entropy_r, p_sum_r, n_sum_r] = calculate_entropy(y(right_j));

        %calculate sum of remainder
        remainder_l = calculate_remainder(p_sum_l, n_sum_l, entropy_l, number_examples);
        remainder_r = calculate_remainder(p_sum_r, n_sum_r, entropy_r, number_examples);
        remainder_attribute = remainder_l + remainder_r;
        
        %calculate information gain
        gain = entropy - remainder_attribute;
        
        if(gain > best_gain)
            best_gain = gain; %take highest gain
            best_gain_threshold = j; %take highest gain threshold index
            left = left_j;
            right = right_j;
        end
    end
end

function [entropy, p_sum, n_sum] = calculate_entropy(y)
%calculate entropy
    p_sum = sum(y);
    pp = p_sum / length(y);
    if (pp == 0)
        pp_eq = 0;
    else
        pp_eq = -1*pp*log2(pp);
    end

    n_sum = length(y) - p_sum;
    pn = 1 - pp;
    if (pn == 0)
        pn_eq = 0;
    else
        pn_eq = -1*pn*log2(pn);
    end

    entropy = pp_eq + pn_eq;
    return
end

function remainder = calculate_remainder(p_sum, n_sum, entropy, number_examples)
%calculate remainder
    remainder = (p_sum + n_sum)*entropy/number_examples;
    return
end