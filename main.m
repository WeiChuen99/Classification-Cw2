%% data loading and feature selection
table = readtable('adult.csv');

%attributes removed due to low predictor importance
table = removevars(table, [1 2 3 4 9 10 12 13 14]);

attribute_name = table.Properties.VariableNames;

%label encoding for table
table.marital_status = double(categorical(table.marital_status));
table.occupation = double(categorical(table.occupation));
table.relationship = double(categorical(table.relationship));
table.census_income = double(categorical(table.census_income));
table.census_income = table.census_income - 1; %convert values 1,2 to 0,1 for easier processing

%convert table to array
x = removevars(table, 6);
x = table2array(x);
y = table.census_income;
%%

%% tree creation and prediction 
tree = DeicisionClassficationTree(x,y,1, attribute_name);
DrawDecisionTree(tree, ''); 

prediction = predict(test_x, tree);
test_y = table.census_income;
accuracy = evaluation(prediction, test_y);
fprintf("accuracy = %.2f%%\n",accuracy);
%%

%% functions
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

function prediction = predict(table, tree)
    table = table2array(table);
    traverse = tree;
    prediction = zeros(length(table),1);

    for i = 1:length(table)
        while ~strcmp(traverse.op, '')
            if table(i, traverse.attribute) <= traverse.threshold
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