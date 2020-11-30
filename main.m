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


%% K-Fold cross validation
for fold =1:k
    fprintf(" %d Fold\n",fold);
    test_indices = 1+(fold-1)*sampleSize:fold*sampleSize;
    train_indices = [1:(fold-1)*sampleSize, fold*sampleSize+1:numTrainingItems];
     
    %% Training data preparation
    trainingData = x(train_indices,:);
    testData = x(test_indices,:);
    trainingData_x = trainingData(:,1:5);
    trainingData_y = trainingData(:,6);
    
    %% Tree creation and prediction
    tree = DecisionClassficationTree(trainingData_x,trainingData_y,1, attribute_name);
    DrawDecisionTree(tree, ''); 
    
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
    
    accuracy_list(1,fold) = accuracy; % store accuracy of each fold
    precision_list(1,fold) = precision; % store accuracy of each fold
    recall_list(1,fold) = recalled; % store accuracy of each fold
    f1_score_list (1,fold)= f1_score;
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

