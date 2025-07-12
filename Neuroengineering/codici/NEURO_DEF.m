%% Micro-electrode recordings (MER) acquired during bilateral subthalamic nucleus deep brain stimulation neurosurgery (STN-BNS)

clc
clear 
close all

%% Loading patient's data
% patient 20
D20=load("Data_Subj20.mat");

% patient 33 - 
D33=load("Data_Subj33.mat");

% patient 38 - 
D38=load("Data_Subj38.mat");

%% SIGNAL FILTERING
% Application of an IIR band-pass filter 
% General Parameters
fs = 20000; % Sampling frequency
fNy = fs / 2; %  Nyquist frequency
fc = 6000; % cutoff frequency observed by PSD

% IIR band-pass filter parameters
Wp = [200 6000] / fNy; % band-pass normalized
Ws = [100 6500] / fNy; % stop-band normalized
Rp = 1; % Ripple in banda-pass(dB)
Rs = 20; % Attenuation in the stopband (dB)

% Filter order and design
[n, Wn] = cheb1ord(Wp, Ws, Rp, Rs);
[b, a] = cheby1(n, Rp, Wn);

% Calculate and display the frequency response of the filter
figure();
freqz(b, a, 512, fs);
title('Frequency response of the band-pass IIR filter');
%% Epoch division with 50% overlap and parameter calculation on filtered signals
% Sampling frequency
epoch_length = fs; % epoch length in samples (1 second)
overlap_length = epoch_length / 2; % overlap length (50%)
step_length = epoch_length - overlap_length; % Step length between epochs
spike_threshold = 0.5; % Threshold for spike recording
% Structure for save results 
results_filtered = struct();
%Function to calculate spike parameters
calculate_spike_params = @(spike_times) struct( ...
    'SC', numel(spike_times) / (spike_times(end) - spike_times(1)), ...
    'SMAD', mean(abs(diff(spike_times))), ...
    'SSD', std(diff(spike_times)), ...
    'SF', 1 / mean(diff(spike_times)) ...
);

% Iteration over patients and hemispheres
patients = {'D20', 'D33', 'D38'};
hemispheres = {'LeftHemisphere', 'RightHemisphere'};

for p = 1:numel(patients)
    patient_name = patients{p};
    patient_data = eval(patient_name); % loading patient's data
    
    for h = 1:numel(hemispheres)
        hemisphere_name = hemispheres{h};
        emisfero = patient_data.Data.(hemisphere_name);
        
        % Retrieve the Target value
        target_value = emisfero.Target;

        % Original signals extraction
        MERs = emisfero.MERs;
        signal_names = fieldnames(MERs);
        num_signals = length(signal_names);
        
        % Prepare a cell array to store the results for each signal
        epoch_results = cell(num_signals, 1);
        
        % Iteration over signals 
        for sig = 1:num_signals
            signal_name = signal_names{sig};
            signal = double(MERs.(signal_name)); % Double convertion
            signal_length = length(signal); % signal length
            target_for_epoch = target_value(sig); % Target element associated with the i-th signal
            
            % Signal filtering
            filtered_signal = filtfilt(b, a, signal); %  IIR filter
            
            % Number of epochs for the signal considering the overlap
            num_epochs = floor((signal_length - overlap_length) / step_length);
            
            % Prepare a cell array to store the results for each signal
            epoch_params = cell(1, num_epochs);
            
            % Divide the filtered signal into epochs and calculate the parameters
            for ep = 1:num_epochs
                start_idx = (ep - 1) * step_length + 1;
                end_idx = start_idx + epoch_length - 1;

                % Ensure that 'end_idx' does not exceed the length of the signal
                if end_idx > signal_length
                    end_idx = signal_length;  % Set 'end_idx' to the last position of the signal
                end

                epoch = filtered_signal(start_idx:end_idx);
                
                % --- Parameter calculation ---
                kurt_val = kurtosis(epoch);
                cl_val = sum(abs(diff(epoch)));
                th_val = max(abs(epoch));
                pk_val = numel(findpeaks(epoch));
                ra_val = rms(epoch);
                ne_val = mean(epoch.^2);
                zc_val = sum(diff(sign(epoch)) ~= 0);
                
                % --- Spike recording and spike parameter calculation---
                spike_times = find(epoch > spike_threshold) / fs * 1000; % Spike times in ms
                
                if numel(spike_times) > 1 % Calculate if there are at least 2 epochs
                    spike_params = calculate_spike_params(spike_times);
                else
                    spike_params = struct('SC', NaN, 'SMAD', NaN, 'SSD', NaN, 'SF', NaN);
                end
                
                % --- Saving the parameters ---
                epoch_params{ep} = struct( ...
                    'Kurtosis', kurt_val, ...
                    'CL', cl_val, ...
                    'TH', th_val, ...
                    'PK', pk_val, ...
                    'RA', ra_val, ...
                    'NE', ne_val, ...
                    'ZC', zc_val, ...
                    'SpikeParams', spike_params, ...
                    'Target', target_for_epoch ...
                );
            end
            
            % Save the epoch results for the current signal
            epoch_results{sig} = epoch_params;
        end
        
        % Save the results for the current hemisphere
        results_filtered.(patient_name).(hemisphere_name) = epoch_results;
    end
end

% Saving the final results
save('ResultsFiltered.mat', 'results_filtered');

%% Parameters for patient selection
test_patient = 'D33'; % Patient to be used as test set
construction_patients = {'D38', 'D20'}; % Patient to be used as construction set

% Test set creation
test_set = [];
mainField = test_patient;
hemispheres = {'LeftHemisphere', 'RightHemisphere'};

for h = 1:numel(hemispheres)
    hemisphereField = hemispheres{h};
    cells = results_filtered.(mainField).(hemisphereField);
    for r = 1:numel(cells)
        subCells = cells{r};
        for c = 1:numel(subCells)
            epochStruct = subCells{c};
            epoch_features = [
                epochStruct.Kurtosis, ...
                epochStruct.CL, ...
                epochStruct.TH, ...
                epochStruct.PK, ...
                epochStruct.RA, ...
                epochStruct.NE, ...
                epochStruct.ZC, ...
                epochStruct.SpikeParams.SC, ...
                epochStruct.SpikeParams.SMAD, ...
                epochStruct.SpikeParams.SSD, ...
                epochStruct.SpikeParams.SF, ...
                epochStruct.Target
            ];
            test_set = [test_set; epoch_features];
        end
    end
end

% Construction set creation 
construction_set = [];
for p = 1:numel(construction_patients)
    patient = construction_patients{p};
    for h = 1:numel(hemispheres)
        hemisphereField = hemispheres{h};
        cells = results_filtered.(patient).(hemisphereField);
        for r = 1:numel(cells)
            subCells = cells{r};
            for c = 1:numel(subCells)
                epochStruct = subCells{c};
                epoch_features = [
                    epochStruct.Kurtosis, ...
                    epochStruct.CL, ...
                    epochStruct.TH, ...
                    epochStruct.PK, ...
                    epochStruct.RA, ...
                    epochStruct.NE, ...
                    epochStruct.ZC, ...
                    epochStruct.SpikeParams.SC, ...
                    epochStruct.SpikeParams.SMAD, ...
                    epochStruct.SpikeParams.SSD, ...
                    epochStruct.SpikeParams.SF, ...
                    epochStruct.Target
                ];
                construction_set = [construction_set; epoch_features];
            end
        end
    end
end

% Normalizzation
[construction_set_norm, min_val, max_val] = normalize_data(construction_set, 11);
[test_set_norm, ~, ~] = normalize_data(test_set, 11, min_val, max_val);

% classes splitting to individuate the majority class
[constr_classe0, constr_classe1] = split_by_class(construction_set_norm, 12);

% construction set splitting to build the training set and the validation set
% considering the 60% of the less rapresented class and the same number of
% elements of the majority class.
[training_set, validation_set] = split_training_validation(constr_classe0, constr_classe1, 0.6);

% Classifiers
k=round(sqrt(size(training_set,1)));
knn_model = train_knn(training_set, k,  'cityblock');
bayes_model = train_bayes(training_set);

% Predictions
[out_train, out_test, ~]=evaluate_classifier(knn_model, training_set, validation_set, test_set_norm);
train_error_knn = mean((training_set(:, end) - out_train).^2);  % Mean Square Error (MSE) for the training data
test_error_knn = mean((test_set(:,end) - out_test).^2);  % Mean Square Error (MSE) for the test data
error_diff_knn =  train_error_knn - test_error_knn;
disp(['Difference between training error and test error (Knn): ', num2str(error_diff_knn)]);


[out_train, out_test, out_val]=evaluate_classifier(bayes_model, training_set, validation_set, test_set_norm);
train_error_bay = mean((training_set(:, end)- out_train).^2);  % Mean Square Error (MSE) for the training data
test_error_bay = mean(( test_set(:,end)- out_test).^2);  % Mean Square Error (MSE) for the test data
error_diff_bay =  train_error_bay - test_error_bay;
disp(['Difference between training error and test error (Bayesian): ', num2str(error_diff_bay)]);

%% Representation of some signals to underline the difference between in-STN signals and out-STN signals
% Define the number of samples to take
num_samples = 100;

% Extract the signals for the first plot (m8)
emisfero = D20.Data.LeftHemisphere;
MERs_20l = emisfero.MERs;
signal_names = fieldnames(MERs_20l);
signal_name_m8 = signal_names{1};
signal_m8 = MERs_20l.(signal_name_m8);
signal_m8_double = double(signal_m8);
filt_signal_m8 = filtfilt(b, a, signal_m8_double);
signal_m8 = signal_m8(1:min(num_samples)); 
filt_signal_m8 = filt_signal_m8(1:min(num_samples));

% Extract the signals for the second plot (m2)
signal_name_m2 = signal_names{5};
signal_m2 = MERs_20l.(signal_name_m2);
signal_m2_double = double(signal_m2);
filt_signal_m2 = filtfilt(b, a, signal_m2_double);
signal_m2 = signal_m2(1:min(num_samples)); 
filt_signal_m2 = filt_signal_m2(1:min(num_samples));

% Calculate the common range (min and max values) for both signals
y_min = min([min(signal_m8), min(filt_signal_m8), min(signal_m2), min(filt_signal_m2)]);
y_max = max([max(signal_m8), max(filt_signal_m8), max(signal_m2), max(filt_signal_m2)]);

% First figure: m8 signal
figure;
hold on;
plot(signal_m8); % Plot the raw m8 signal
hold on;
plot(filt_signal_m8); % Plot the filtered m8 signal
legend('raw m8','filtered m8'); % Legend for the signals
title('Representation of the m8 signal (out STN) - patient D20 - Left hemisphere'); % Title of the plot
xlabel('Samples'); % X-axis label
ylabel('Amplitude [µV]'); % Y-axis label
axis([0 num_samples y_min y_max]); % Set the same range for the y-axis
hold off;

% Second figure: m2 signal
figure;
hold on;
plot(signal_m2); % Plot the raw m2 signal
hold on;
plot(filt_signal_m2); % Plot the filtered m2 signal
legend('raw m2','filtered m2'); % Legend for the signals
title('Representation of the m2 signal (in STN) - patient D20 - Left hemisphere'); % Title of the plot
xlabel('Sample Index'); % X-axis label
ylabel('Amplitude [µV]'); % Y-axis label
axis([0 num_samples y_min y_max]); % Set the same range for the y-axis
hold off;
