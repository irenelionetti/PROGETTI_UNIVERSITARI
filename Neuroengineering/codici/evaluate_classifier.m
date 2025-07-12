function [out_train, out_test, out_val]=evaluate_classifier(model, training_set, validation_set, test_set)
    % Predizione sui dataset
    out_train = predict(model, training_set(:, 1:end-1));
    out_val = predict(model, validation_set(:, 1:end-1));
    out_test = predict(model, test_set(:, 1:end-1));
    
    % Training Set
    disp('Confusion Matrix for Training Set:');
    CM_train = confusionmat(training_set(:, end), out_train);
    disp(CM_train);
    evaluate_metrics(CM_train, 'Training Set');
    
    % Validation Set
    disp('Confusion Matrix for Validation Set:');
    CM_val = confusionmat(validation_set(:, end), out_val);
    disp(CM_val);
    evaluate_metrics(CM_val, 'Validation Set');
    
    % Test Set
    disp('Confusion Matrix for Test Set:');
    CM_test = confusionmat(test_set(:, end), out_test);
    disp(CM_test);
    evaluate_metrics(CM_test, 'Test Set');
end

function evaluate_metrics(CM, set_name)
    % Calcolo delle metriche
    TP = CM(1, 1);  % True Positives
    FN = CM(1, 2);  % False Negatives
    FP = CM(2, 1);  % False Positives
    TN = CM(2, 2);  % True Negatives

    % Metriche
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1_score = 2 * (precision * recall) / (precision + recall);
    accuracy = (TP + TN) / sum(CM(:));
    error_rate = (FP + FN) / sum(CM(:));
    FPR = FP / (FP + TN); % False Positive Rate
    FNR = FN / (TP + FN); % False Negative Rate

    % Mostra i risultati
    disp(['Metrics for ', set_name, ':']);
    fprintf('  Precision: %.2f\n', precision);
    fprintf('  Recall: %.2f\n', recall);
    fprintf('  F1-Score: %.2f\n', f1_score);
    fprintf('  Accuracy: %.2f\n', accuracy);
    fprintf('  Error Rate: %.2f\n', error_rate);
    fprintf('  FPR: %.2f\n', FPR);
    fprintf('  FNR: %.2f\n', FNR);
    
    % Plot della confusion matrix
    figure;
    confusionchart(CM);
    title(['Confusion Matrix - ', set_name]);
end
