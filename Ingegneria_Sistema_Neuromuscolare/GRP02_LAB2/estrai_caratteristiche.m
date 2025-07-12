function [features, labels] = estrai_caratteristiche(data, label_val, epoch_length)
    num_epochs = floor(size(data, 1) / epoch_length);
    num_channels = size(data, 2);
    
    data_epochs = reshape(data(1:num_epochs * epoch_length, :), epoch_length, num_epochs, []);
    features = [];
    labels = label_val * ones(num_epochs, 1);
    
    for epoch_idx = 1:num_epochs
        epoch_features = [];
        for channel_idx = 1:num_channels
            epoch_data = data_epochs(:, epoch_idx, channel_idx);
            rms_val = rms(epoch_data);
            mav_val = mean(abs(epoch_data));
            ssc_val = sum(diff(sign(diff(epoch_data))) ~= 0);
            wl_val = sum(abs(diff(epoch_data)));
            epoch_features = [epoch_features, rms_val, mav_val, ssc_val, wl_val];
        end
        features = [features; epoch_features];
    end
end
