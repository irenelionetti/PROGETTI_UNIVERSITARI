%Min-Max scaling normalization
function [normalized_data, min_val, max_val] = normalize_data(data, num_features, min_val, max_val)
    if nargin < 3
        min_val = min(data(:, 1:num_features));
        max_val = max(data(:, 1:num_features));
    end
    normalized_data = (data(:, 1:num_features) - min_val) ./ (max_val - min_val);
    normalized_data = [normalized_data, data(:, end)]; % Aggiungi la classe
end