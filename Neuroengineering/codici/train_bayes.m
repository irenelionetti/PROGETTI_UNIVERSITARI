function bayes_model = train_bayes(data)
    bayes_model = fitcnb(data(:, 1:end-1), data(:, end));
end