
function [class0, class1] = split_by_class(data, class_col)
    class0 = data(data(:, class_col) == 0, :);
    class1 = data(data(:, class_col) == 1, :);
end