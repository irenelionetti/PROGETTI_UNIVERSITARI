
function [training_set, validation_set] = split_training_validation(class0, class1, ratio)
    n_train = round(ratio * size(class0, 1));
    idx0 = randperm(size(class0, 1), n_train);
    idx1 = randperm(size(class1, 1), n_train);
    training_set = [class0(idx0, :); class1(idx1, :)];
    class0(idx0, :) = [];
    class1(idx1, :) = [];
    validation_set = [class0; class1];
end