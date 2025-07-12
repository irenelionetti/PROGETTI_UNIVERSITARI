function knn_model = train_knn(data, k, distance)
    knn_model = fitcknn(data(:, 1:end-1), data(:, end), 'NumNeighbors', k, 'Distance', distance,'IncludeTies',true, 'BreakTies', 'nearest');
end

