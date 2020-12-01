prompt = 'Select which tree to run: \nClassification(1)\nRegression(0)\nExit(e)\n\n';
user_input = input(prompt, 's');

while user_input ~= 'e'
    if user_input == '1'
        fprintf("Classification Tree selected\n");
        DecisionClassificationTree;
        fprintf("Classification Tree completed\n\n");
    elseif user_input == '0'
        fprintf("Regression Tree selected, estimated run time = 10mins\n");
        DecisionRegressionTree;
        fprintf("Regression Tree completed\n\n");
    else
        fprintf("Invalid input, please select again\n\n");
    end
    
    prompt = 'Select which tree to run: \nClassification(1)\nRegression(0)\nExit(e)\n\n';
    user_input = input(prompt, 's');
end

fprintf("Cancelled\n");