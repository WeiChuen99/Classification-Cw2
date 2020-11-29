function [tree] = DeicisionClassficationTree(x,y,depth, attribute_name)
    tree = struct('op','','kids',[],'class',[],'attribute',0,'threshold', 0);
    min_gain = 0.01;
    
    [best_gain_attribute, best_gain_threshold, best_gain, left, right]=build_node(x,y);

    y_left=y(left);
    y_right=y(right);
    x_left=x(left,:); 
    x_right=x(right,:);
    fprintf('Column = %d. SplitValue = %f. gain = %f.\n', ...
        best_gain_attribute, best_gain_threshold, best_gain);

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
            tree.kids{1} = DeicisionClassficationTree(x_left, y_left, depth, attribute_name);
        end
        if length(unique(y_right)) == 1
            tree.kids{2}.op= '';
            tree.kids{2}.kids= [];
            tree.kids{2}.class= mode(y_right);
            tree.kids{2}.attribute= 0;
            tree.kids{2}.threshold= 0;
        else
            tree.kids{2} = DeicisionClassficationTree(x_right, y_right, depth, attribute_name);
        end
        depth = depth-1;
    end
end

%loops through all columns to decide which attribute should be used to split
function [best_gain_attribute,best_gain_threshold,best_gain,left,right] = build_node(x,y) 
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

%loops through examples(datapoints) and determine best splitting point (threshhold)
function [best_gain_threshold,best_gain,left,right] = split(x, y)
    [x_row, ~] = size(x);
    number_examples = x_row;
    best_gain = 0;
    best_gain_threshold = 1;
%     left = find(x>x(1));
%     right = find(x<=x(1));
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
        fprintf("example: %d, gain: %.4f\n", j, gain);
        
        if(gain > best_gain)
            best_gain = gain; %take highest gain
            best_gain_threshold = j; %take highest gain threshold index
            left = left_j;
            right = right_j;
        end
    end
end

function [entropy, p_sum, n_sum] = calculate_entropy(y)
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