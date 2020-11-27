table = readtable('adult.csv');

table = removevars(table, [1 2 3 4 9 10 12 13 14]);
attribute_name = table.Properties.VariableNames;

table.marital_status = double(categorical(table.marital_status));
table.occupation = double(categorical(table.occupation));
table.relationship = double(categorical(table.relationship));

table.census_income = double(categorical(table.census_income));
table.census_income = table.census_income - 1;

table = table(1:5000,:);

x = removevars(table, 6);
x = table2array(x);
y = table.census_income;

% y_1 = find(y == 1);
% y_2 = find(y == 0);
% x = x([y_1 ; y_2(1:1250)]);
% y = y([y_1 ; y_2(1:1250)]);
tree = ID3(x,y,1,1, attribute_name);
DrawDecisionTree(tree); 

function [tree] = ID3(x,y,depth,flag, attribute_name)
    tree = struct('op','','kids',[],'class',[],'attribute',0,'threshold', 0);
    [x_row, x_col] = size(x);
    number_attributes = x_col;
    number_examples = x_row;
    entropy = calculate_entropy(x,y);
    
    best_gain = 0;
    best_gain_threshold = 0;
    best_gain_attribute = 0;
    min_node = 100;
    min_gain = 0.010;
    
    [best_gain_attribute, best_gain_threshold, best_gain, left, right]=build_node(x,y);

    
    y_left=y(left);
    y_right=y(right);
    x_left=x(left,:); 
    x_right=x(right,:);
    fprintf('Column = %d. SplitValue = %f. gain = %f.\n', ...
        best_gain_attribute, best_gain_threshold, best_gain);

    if best_gain < min_gain || isempty(left) || isempty(right)
        tree.op = char(attribute_name{best_gain_attribute});
        tree.attribute = best_gain_attribute;
        tree.threshold = best_gain_threshold;
        tree.class = mode(y);
        return;
    else
        tree.op = char(attribute_name{best_gain_attribute});
        tree.attribute = best_gain_attribute;
        tree.threshold = best_gain_threshold;
        tree.kids = cell(1,2);
        depth = depth+1;
        %recursion
        if length(unique(y_left)) == 1
            tree.kids{1}.op= '';
            tree.kids{1}.kids= [];
            tree.kids{1}.class= mode(y_left);
            tree.kids{1}.attribute= 0;
            tree.kids{1}.threshold= 0;
        else
            tree.kids{1} = ID3(x_left, y_left, depth, 1, attribute_name);
        end
        if length(unique(y_right)) == 1
            tree.kids{2}.op= '';
            tree.kids{2}.kids= [];
            tree.kids{2}.class= mode(y_left);
            tree.kids{2}.attribute= 0;
            tree.kids{2}.threshold= 0;
        else
            tree.kids{2} = ID3(x_right, y_right, depth, 0, attribute_name);
        end
        depth = depth-1;
    end
    
    
    %termination: 
    %pure(set pure side as leaf node and add prediction), 
    %gain is less than min gain(set current node as leaf node, prediction = majority of class), 
    %left/right empty(set current node as leaf node, prediction = majority of class)
end

function [best_gain_attribute,best_gain_threshold,best_gain,left,right] = build_node(x,y)
    [~, x_col] = size(x);
    number_attributes = x_col;
    best_gain_attribute = 1
    best_gain_threshold = 1;
    best_gain = 0;
    for i = 1:number_attributes
        [best_gain_threshold_i,best_gain_i,left,right]=split(x(:,i),y);
        if(best_gain_i>best_gain)
            best_gain = best_gain_i;
            best_gain_attribute = i;
        end
    end
    [best_gain_threshold,best_gain,left,right]=split(x(:,i),y);
end

function [best_gain_threshold,best_gain,left,right] = split(x, y)
    [x_row, ~] = size(x);
    number_examples = x_row;
    entropy = calculate_entropy(x,y);
    best_gain = 0;
    best_gain_threshold = 1;
    left = find(x>x(1));
    right = find(x<=x(1));
    
    x_min = min(x);
    x_max = max(x);
%     inc = (x_max - x_min)/1000; % length of increament
    
    for j = 1:number_examples
        left_j = find(x>x(j));
        right_j = find(x<=x(j));
        
        if isempty(left_j)
            left_j = find(x == x_max);
            right_j = find(x<x(j));
        end
        
        gain = 0;
        [entropy_l, p_sum_l, n_sum_l] = calculate_entropy(x(left_j), y(left_j));
        [entropy_r, p_sum_r, n_sum_r] = calculate_entropy(x(right_j), y(right_j));

        remainder_l = calculate_remainder(p_sum_l, n_sum_l, entropy_l, number_examples);
        remainder_r = calculate_remainder(p_sum_r, n_sum_r, entropy_r, number_examples);
        remainder_attribute = remainder_l + remainder_r;

        gain = entropy - remainder_attribute;
        fprintf("example: %d, gain: %.4f\n", j, gain);
        if(gain > best_gain)
            best_gain = gain; %take highest gain
            best_gain_threshold = j; %take highest gain threshold index
        end
    end
end

function [entropy, p_sum, n_sum] = calculate_entropy(x,y)
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
    remainder = (p_sum + n_sum)*entropy/number_examples;
    return
end

%         category_begining = 1;
%         category_index = 1;
%         remainder = 0;
% 
%         category_gain_eq = unique(x);
%         category_gain_eq = zeros(height(category_gain_eq), 4);
% 
%         for j = 1:number_examples - 1
%             if (table(j, i) ~= table(j+1, i))
%                 [category_gain_eq(category_index,1), category_gain_eq(category_index,2), ...
%                     category_gain_eq(category_index,3)] ...
%                     = calculate_entropy(table(category_begining:j, :));
% 
%                 category_begining = j+1;
%                 category_index = category_index+1;
%             elseif (j == number_examples - 1)
%                 [category_gain_eq(category_index,1), category_gain_eq(category_index,2), ...
%                     category_gain_eq(category_index,3)] ...
%                     = calculate_entropy(table(category_begining:j+1, :));
%             end
%         end
% 
%         for k = 1:height(category_gain_eq)
%             remainder = remainder + calculate_remainder(category_gain_eq(k,2), ...
%                 category_gain_eq(k,3), category_gain_eq(k,1), number_examples);
%         end
% 
%         gain = entropy - remainder;
% 
%         if(gain > best_gain)
%             best_gain = gain; %take highest gain
%             best_gain_threshold = j; %take highest gain threshold index
%             best_gain_attribute = i; %take highest gain attribute index
%         end