clc, clear all, close all;


%carica aperture e chiusure 
data_aperture=readmatrix('20_solo_apertura_group2.txt');
data_chiusure=readmatrix('20_chiusura_group2.txt');
time_aperture=data_aperture(:,1);
time_chiusure=data_chiusure(:,1);

%seleziono i canali effettivamente utilizzati
canali_presi=[2, 5, 6, 7, 9];

matrice_aperture=data_aperture(:,canali_presi);
matrice_chiusure=data_chiusure(:,canali_presi);

%% 2. Disporre i segnali ottenuti in una matrice per colonne. 
% Il risultato atteso sarà una matrice m × n dove m indica i campioni e n i canali. Disporre prima le contrazioni associate all’apertura della mano e poi quelle di chiusura. Fare in modo che la lunghezza temporale delle contrazioni incluse nella matrice sia la stessa. 

n_rows=min(size(matrice_chiusure,1), size(matrice_aperture,1));

% Creiamo un asse di riferimento normalizzato
x_aperture = linspace(1, n_rows, size(matrice_aperture,1)); 
x_chiusure = linspace(1, n_rows, size(matrice_chiusure,1)); 
x_new = 1:n_rows; % Nuovo asse uniforme

% Resampling usando interp1 con interpolazione lineare
aperture_resampled = interp1(x_aperture, matrice_aperture, x_new, 'linear', 'extrap');
chiusure_resampled = interp1(x_chiusure, matrice_chiusure, x_new, 'linear', 'extrap');

% Concatenazione verticale per ottenimento matrice di segnali grezzi aperture e poi chiusure
C = [aperture_resampled; chiusure_resampled];



%% 3. Creare una copia della matrice al punto 2 e sostituire i segnali grezzi con gli inviluppi dei segnali EMG mediante raddrizzamento e filtraggio passabasso a 10 Hz, ottenendo nuovamente una matrice di dimensione m × n

% Rappresentazione dei segnali grezzi e ricampionati di tutti i canali
% Grafico segnali aperture grezzo nei 5 canali
figure()
for i=1:5 
    vv =matrice_aperture (:,i)/range(matrice_aperture(:));  
    plot(time_aperture,vv+i);
    hold on
    xlabel('Tempo (s)');
    ylabel('Canali');
end
yticks(1:5)
yticklabels({'Canale 1','Canale 2','Canale 3','Canale 4','Canale 5'})
ylabel('Canali normalizzati');
title('Segnale Grezzo Aperture 5 canali');


%Grafico segnali chiusure grezzo nei 5 canali
figure()
for i=1:5 
    vv =matrice_chiusure (:,i)/range(matrice_chiusure(:)); 
    plot(time_chiusure,vv+i);
    hold on
    xlabel('Tempo (s)');
    ylabel('Canali');
end
title('Segnale Grezzo Chiusure 5 canali');
yticks(1:5)
yticklabels({'Canale 1','Canale 2','Canale 3','Canale 4','Canale 5'})
ylabel('Canali normalizzati');


% Grafico segnali chiusure ricampionati nei 5 canali
figure()
for i=1:5 
    vv =chiusure_resampled (:,i)/range(chiusure_resampled(:));   
    plot(x_new',vv+i);
    hold on
    xlabel('Campioni');
    ylabel('Canali normalizzati');
end
title('Segnale Ricampionato Chiusure 5 canali');
yticks(1:5)
yticklabels({'Canale 1','Canale 2','Canale 3','Canale 4','Canale 5'})
ylabel('Canali normalizzati');

% PSD dei canali migliore 
% apertura --> 3 (durante l'apertura mi aspetto si contraggano i muscoli dorsali dell'avambraccio, dove abbiamo posizionato la sonda numero 5)
% chisura --> 4, 5 (durante la chiusura si contraggono i muscoli ventrali dell'avambraccio, dove abbimo messo le sonde numero in realtà 6, 8)

% PSD del segnale delle aperture del canale 3
fs = 2048; % Frequenza di campionamento 
[Pxx_ap,f_ap] = pwelch(aperture_resampled(:,3)-mean(aperture_resampled(:,3)),hamming(1024),512,1024,fs); 
figure()
plot(f_ap,Pxx_ap/(max(Pxx_ap)))
axis([0 fs/2 0 1])
title('Stima della PSD delle aperture ricampionate, canale 3')
xlabel('Frequenza (Hz) ');
ylabel('PSD normalizzata');

% PSD del segnale di chiusura del canale 4
[Pxx_ch,f_ch] = pwelch(chiusure_resampled(:,4)-mean(chiusure_resampled(:,4)),hamming(1024),512,1024,fs); 
figure()
plot(f_ch,Pxx_ch/(max(Pxx_ch)))
axis([0 fs/2 0 1])
title('Stima della PSD delle chiusure ricampionate, canale 4')
xlabel('Frequenza (Hz) ');
ylabel('PSD normalizzata');

% filtraggio passa alto per rimuovere la continua --> drift 
[b,a] = cheby1(6, 0.5, 5/(fs/2), 'high');
figure;
freqz(b, a, 1024, fs);
title('Risposta in frequenza del filtro passa-alto (5 Hz)');
aperture_filt=filtfilt(b, a, aperture_resampled);
chiusure_filt=filtfilt(b, a, chiusure_resampled);

figure()
% Primo subplot a sinistra: aperture
subplot(1,2,1) 
for i = 1:5 
    vv = aperture_filt(:,i) / range(aperture_filt(:));   
    plot(x_new', vv + i);
    hold on
end
xlabel('Campioni');
ylabel('Canali normalizzati');
title('Segnale aperture filtrato (5 canali)');
yticks(1:5)
yticklabels({'Canale 1','Canale 2','Canale 3','Canale 4','Canale 5'})
ylabel('Canali normalizzati');

% Secondo subplot a destra: chiusure
subplot(1,2,2) 
for i = 1:5 
    vv = chiusure_filt(:,i) / range(chiusure_filt(:)); 
    plot(x_new', vv + i);
    hold on
end
xlabel('Campioni');
ylabel('Canali normalizzati');
title('Segnale chiusure filtrato (5 canali)');
yticks(1:5)
yticklabels({'Canale 1','Canale 2','Canale 3','Canale 4','Canale 5'})
ylabel('Canali normalizzati');


% Filtro passa-basso per ottenere INVILUPPO
ap_ch=C; %creare una copia della matrice al punto 2
fc = 10;    % Frequenza di taglio   
% Creo filtro passa-basso 
[b1, a1] = butter(6, fc / (fs / 2), 'low');

% Plot della risposta in frequenza del filtro
figure;
freqz(b, a, 1024, fs);
title('Risposta in frequenza del filtro passa-basso (10 Hz)');

% Rettificare applicando il valore assoluto
ap_ch_rect= abs(ap_ch) ;  % Rettifico

% Applicazione del filtro passa-basso per ottenere l'inviluppo
ap_ch_env=filtfilt(b1, a1, ap_ch_rect);

% Grafico di confronto per una colonna
figure;
plot(ap_ch(:,2), 'b', 'DisplayName', 'Segnale EMG grezzo'); hold on;
plot(ap_ch_env(:,2), 'r', 'LineWidth', 1.5, 'DisplayName', 'Inviluppo EMG');
legend;
xlabel('Campioni');
ylabel('Ampiezza (µV)'); 
title('Inviluppo del Segnale EMG');
grid on;



%% 4. Creare un vettore di label (0 e 1 oppure -1 e 1) di lunghezza m. Ogni label sarà associata alla tipologia di contrazione esaminata.
m = size(ap_ch_env, 1);
% Creazione del vettore di label
labels = zeros(m, 1);
labels(1:round(m/2)) = 1;  % Prima metà dei campioni -> APERTURA --> 1
labels(round(m/2)+1:end) = 0;  % Seconda metà dei campioni -> CHIUSURA --> 0
% aggiungo alla matrice che contiene gli inviluppi una colonna di label
A = [ap_ch_env, labels];


%% 5. Applicare in matlab un algoritmo di Support Vector Machine (SVM) per classificare i dati. Utilizzare le matrici costruite ai punti 2 e 3.
% Dividere i dati in training e test set secondo una partizione del 70-30%.
% Stimare precisione e accuratezza per le due tipologie di dati. 

% Matrice costruita al punto 2 --> C
% Matrice costruita al punto 3 --> ap_ch_env

% MATRICE 1: segnali grezzi (C)
X_raw = C;
y = labels;

% MATRICE 2: segnali con inviluppo (ap_ch_env)
X_env = ap_ch_env;


% Shuffle (mescola) i dati
N = size(X_raw, 1);                  % numero totale di epoche
idx = randperm(N);              % indici random

X_shuffled_raw = X_raw(idx, :);         % mescola le righe di X
y_shuffled = y(idx);            % mescola le label corrispondenti
X_shuffled_env=X_env(idx, :);

% Divide 70-30
nTrain = round(0.7 * N);

X_train_raw = X_shuffled_raw(1:nTrain, :);
X_train_env=X_shuffled_env(1:nTrain,:);
y_train = y_shuffled(1:nTrain);

X_test_raw = X_shuffled_raw(nTrain+1:end, :);
X_test_env = X_shuffled_env(nTrain+1:end, :);
y_test = y_shuffled(nTrain+1:end);


%% SVM --> applicato direttamente alle matrici senza estarre le features

% -------- SVM su segnali grezzi --------
% prova a vedre i risultati cambiando kernel e altri parametri
svm_model_raw = fitcsvm(X_train_raw, y_train, 'KernelFunction', 'linear', 'BoxConstraint', 1, ...
    'Standardize', true);
% Predizione
[estimated_class_raw, score_raw] = predict(svm_model_raw, X_test_raw);
% Valutazione
confMat_raw = confusionmat(y_test, estimated_class_raw);
accuracy_raw = sum(diag(confMat_raw)) / sum(confMat_raw(:));
precision_raw = confMat_raw(2,2) / sum(confMat_raw(:,2));

fprintf('\n--- Classificazione SVM su segnali grezzi ---\n');
fprintf('Accuratezza: %.2f%%\n', accuracy_raw * 100); %TP+TN/(TP+TN+FP+FN)
fprintf('Precisione: %.2f%%\n', precision_raw * 100); %TP/TP+TN


% -------- SVM su segnali inviluppati --------
svm_model_env = fitcsvm(X_train_env, y_train, 'KernelFunction', 'linear');
% Predizione
[estimated_class_env, score_env] = predict(svm_model_env, X_test_env);
% Valutazione
confMat_env = confusionmat(y_test, estimated_class_env);
accuracy_env = sum(diag(confMat_env)) / sum(confMat_env(:));
precision_env = confMat_env(2,2) / sum(confMat_env(:,2));

fprintf('\n--- Classificazione SVM su inviluppi ---\n');
fprintf('Accuratezza: %.2f%%\n', accuracy_env * 100);
fprintf('Precisione: %.2f%%\n', precision_env * 100);

%% 6. Applicare la Linear Discriminant Analysis (LDA) ai dati ottenuti ai punti 2 e 3 --> applicato direttamente alle matrici senza estarre le features


% Standardizzazione manuale dei dati (media 0, varianza 1)
%X_raw_standardized = zscore(X_raw);
%X_env_standardized = zscore(X_env);

% -------- LDA su segnali grezzi --------
lda_model_raw = fitcdiscr(X_train_raw, y_train,'DiscrimType', 'linear');
% Predizione
estimated_class_raw_lda = predict(lda_model_raw, X_test_raw);
% Valutazione
confMat_raw_lda = confusionmat(y_test, estimated_class_raw_lda);
accuracy_raw_lda = sum(diag(confMat_raw_lda)) / sum(confMat_raw_lda(:));
precision_raw_lda = confMat_raw_lda(2,2) / sum(confMat_raw_lda(:,2));

fprintf('\n--- Classificazione LDA su segnali grezzi ---\n');
fprintf('Accuratezza: %.2f%%\n', accuracy_raw_lda * 100);
fprintf('Precisione: %.2f%%\n', precision_raw_lda * 100);


% -------- LDA su segnali inviluppati --------
lda_model_env = fitcdiscr(X_train_env, y_train,'DiscrimType', 'linear');
% Predizione
estimated_class_env_lda = predict(lda_model_env, X_test_env);
% Valutazione
confMat_env_lda = confusionmat(y_test, estimated_class_env_lda);
accuracy_env_lda = sum(diag(confMat_env_lda)) / sum(confMat_env_lda(:));
precision_env_lda = confMat_env_lda(2,2) / sum(confMat_env_lda(:,2));

fprintf('\n--- Classificazione LDA su inviluppi ---\n');
fprintf('Accuratezza: %.2f%%\n', accuracy_env_lda * 100);
fprintf('Precisione: %.2f%%\n', precision_env_lda * 100);

%% SVM e LDA con FE su segnali inviluppati

% Separa i dati per apertura e chiusura
apertura_data = A(A(:, end) == 1, 1:end-1);  % Tutti i dati con etichetta 1 (apertura)
chiusura_data = A(A(:, end) == 0, 1:end-1);  % Tutti i dati con etichetta 0 (chiusura)
% Imposta la lunghezza dell'epoca
epoch_length = 512; %250 ms

% Estrai caratteristiche
[features_apertura_env, labels_apertura_env] = estrai_caratteristiche(apertura_data, 1, epoch_length);
[features_chiusura_env, labels_chiusura_env] = estrai_caratteristiche(chiusura_data, 0, epoch_length);

% 2. Combinare aperture e chiusure in una matrice unica di features e etichette
features_env = [features_apertura_env;features_chiusura_env];  % Matrice di caratteristiche (apertura + chiusura)
classes_env = [labels_apertura_env; labels_chiusura_env];    % Vettore di etichette (1 = apertura, 0 = chiusura)

% Shuffle (mescola) i dati
N = size(features_env, 1); 
idx = randperm(N);             
features_env_shuffled = features_env(idx, :); % mescola le righe 
classes_shuffled_env = classes_env(idx); % mescola le label corrispondenti

% 4. Divide 70-30
nTrain = round(0.7 * N);
features_train_env= features_env_shuffled(1:nTrain, :);
classes_train_env = classes_shuffled_env(1:nTrain);
features_test_env = features_env_shuffled(nTrain+1:end, :);
classes_test_env= classes_shuffled_env(nTrain+1:end);

% SVM
svm_model_env_feature = fitcsvm(features_train_env, classes_train_env, 'KernelFunction', 'linear', 'BoxConstraint', 1,'Standardize', true);
% Predizione
[estimated_class_env_feature, score_env_feature] = predict(svm_model_env_feature, features_test_env);

% LDA 
lda_model_env_feature = fitcdiscr(features_train_env, classes_train_env,'DiscrimType', 'linear');
% Predizione
estimated_class_env_feature_lda = predict(lda_model_env_feature, features_test_env);

% Valutazione SVM
confMat_env_feature= confusionmat(classes_test_env, estimated_class_env_feature);
accuracy_env_feature = sum(diag(confMat_env_feature)) / sum(confMat_env_feature(:));
precision_env_feature = confMat_env_feature(2,2) / sum(confMat_env_feature(:,2));

fprintf('\n--- Classificazione SVM su segnali inviluppati con feature extraction ---\n');
fprintf('Accuratezza: %.2f%%\n', accuracy_env_feature* 100);
fprintf('Precisione: %.2f%%\n', precision_env_feature * 100); 

figure;
conf_svm_env_feature = confusionchart(confMat_env_feature, {'Classe 0', 'Classe 1'});
title('Confusion Matrix SVM con FE- Test Set');

% Valutazione LDA
confMat_env_feature_lda = confusionmat(classes_test_env, estimated_class_env_feature_lda);
accuracy_env_feature_lda = sum(diag(confMat_env_feature_lda)) / sum(confMat_env_feature_lda(:));
precision_env_feature_lda = confMat_env_feature_lda(2,2) / sum(confMat_env_feature_lda(:,2));

fprintf('\n--- Classificazione LDA su inviluppi con feature extraction ---\n');
fprintf('Accuratezza: %.2f%%\n', accuracy_env_feature_lda * 100);
fprintf('Precisione: %.2f%%\n', precision_env_feature_lda * 100);

figure;
conf_lda_env_feature = confusionchart(confMat_env_feature_lda, {'Classe 0', 'Classe 1'});
title('Confusion Matrix LDA con FE - Test Set');

%% SVM con FE su segnali grezzi 
D=[C labels];
apertura_data_raw = D(D(:, end) == 1, 1:end-1);  % Tutti i dati con etichetta 1 (apertura)
chiusura_data_raw = D(D(:, end) == 0, 1:end-1);  % Tutti i dati con etichetta 0 (chiusura)

% Estrai caratteristiche
[features_apertura_raw, labels_apertura_raw] = estrai_caratteristiche(apertura_data_raw, 1, epoch_length);
[features_chiusura_raw, labels_chiusura_raw] = estrai_caratteristiche(chiusura_data_raw, 0, epoch_length);

% 2. Combinare aperture e chiusure in una matrice unica di features e etichette
features_raw = [features_apertura_raw;features_chiusura_raw];  % Matrice di caratteristiche (apertura + chiusura)
classes_raw = [labels_apertura_raw; labels_chiusura_raw];    % Vettore di etichette (1 = apertura, 0 = chiusura)

% 3. Shuffle 
N = size(features_raw, 1);                
idx = randperm(N);              
features_raw_shuffled = features_raw(idx, :);        
classes_shuffled_raw = classes_raw(idx);           

% 4. Divide 70-30
nTrain = round(0.7 * N);
features_train_raw= features_raw_shuffled(1:nTrain, :);
classes_train_raw = classes_shuffled_raw(1:nTrain);
features_test_raw = features_raw_shuffled(nTrain+1:end, :);
classes_test_raw= classes_shuffled_raw(nTrain+1:end);

% SVM
svm_model_raw_feature = fitcsvm(features_train_raw, classes_train_raw, 'KernelFunction', 'linear', 'BoxConstraint', 1, 'Standardize', true);
% Predizione
[estimated_class_raw_feature, score_raw_feature] = predict(svm_model_raw_feature, features_test_raw);

% LDA 
lda_model_raw_feature = fitcdiscr(features_train_raw, classes_train_raw,'DiscrimType', 'linear');
% Predizione
estimated_class_raw_feature_lda = predict(lda_model_raw_feature, features_test_raw);

% Valutazione SVM
confMat_raw_feature= confusionmat(classes_test_raw, estimated_class_raw_feature);
accuracy_raw_feature = sum(diag(confMat_raw_feature)) / sum(confMat_raw_feature(:));
precision_raw_feature = confMat_raw_feature(2,2) / sum(confMat_raw_feature(:,2));

fprintf('\n--- Classificazione SVM su segnali grezzi con feature extraction ---\n');
fprintf('Accuratezza: %.2f%%\n', accuracy_raw_feature* 100);
fprintf('Precisione: %.2f%%\n', precision_raw_feature * 100); 

figure;
conf_svm_raw_feature = confusionchart(confMat_raw_feature, {'Classe 0', 'Classe 1'});
title('Confusion Matrix SVM segnali raw - Test Set');

% Valutazione LDA
confMat_raw_feature_lda = confusionmat(classes_test_raw, estimated_class_raw_feature_lda);
accuracy_raw_feature_lda = sum(diag(confMat_raw_feature_lda)) / sum(confMat_raw_feature_lda(:));
precision_raw_feature_lda = confMat_raw_feature_lda(2,2) / sum(confMat_raw_feature_lda(:,2));

fprintf('\n--- Classificazione LDA su seganli grezzi con feature extraction ---\n');
fprintf('Accuratezza: %.2f%%\n', accuracy_raw_feature_lda * 100);
fprintf('Precisione: %.2f%%\n', precision_raw_feature_lda * 100);

figure;
conf_lda_raw_feature = confusionchart(confMat_raw_feature_lda, {'Classe 0', 'Classe 1'});
title('Confusion Matrix LDA con FE - Test Set');


%% 8. Classificatore con cosine similarity

win_length = round(0.25 * fs); % 250 ms
overlap = round(0.5 * win_length); % 50% overlap
step_size = win_length - overlap;

% uso i dati delle contrazioni massimali per creare dei prototipi come TEST
% Caricamento dati contrazioni massimali (apertura e chiusura)
data_apertura_max = readmatrix('massima_apertura_group2.txt');
data_chiusura_max = readmatrix('massima_chiusura_group2.txt');
canali_presi = [2, 5, 6, 7, 9];

% Selezione canali
apertura_max = data_apertura_max(:,canali_presi); % Tempo nelle righe, canali nelle colonne
chiusura_max = data_chiusura_max(:,canali_presi);

% Funzione per calcolare ARV correttamente su ogni canale (considerando il formato corretto)
compute_arv = @(segment) mean(abs(segment), 1);

% Suddivisione in finestre e calcolo ARV per ogni canale
arv_apertura = zeros(floor((size(apertura_max,1) - win_length) / step_size) + 1, length(canali_presi));
arv_chiusura = zeros(floor((size(chiusura_max,1) - win_length) / step_size) + 1, length(canali_presi));

for i = 1:step_size:size(apertura_max,1)-win_length
    arv_apertura(floor(i/step_size) + 1, :) = compute_arv(apertura_max(i:i+win_length-1, :));
end

for i = 1:step_size:size(chiusura_max,1)-win_length
    arv_chiusura(floor(i/step_size) + 1, :) = compute_arv(chiusura_max(i:i+win_length-1, :));
end

% Creazione dei prototipi (media sui valori ARV) --> di dimensione n=5 canali
prototype_apertura = mean(arv_apertura, 1);
prototype_chiusura = mean(arv_chiusura, 1);

% Caricamento dati di test (matrice dal punto 2)
data_test = C; % Matrice contenente i segnali di test

% Calcolo ARV per dati di test
arv_test = zeros(floor((size(data_test,1) - win_length) / step_size) + 1, length(canali_presi));
true_labels = zeros(size(arv_test,1), 1);  % inizializza vettore etichette vere

for i = 1:step_size:size(data_test,1)-win_length
    arv_test(floor(i/step_size) + 1, :) = compute_arv(data_test(i:i+win_length-1, :));
    finestra_etichette = labels(i:i+win_length-1);
    true_labels(floor(i/step_size) + 1) = mode(finestra_etichette);  % classe più frequente nella finestra
end

% Classificazione mediante cosine similarity--> calcolo cos sim tra ARV dei
% dati di test e ARV dei prototipi --> classifico in base a chi presenta
% cos maggiore
labels_test = zeros(size(arv_test,1),1);

for i = 1:size(arv_test,1) %per ogni riga
    cos_sim_apertura = dot(arv_test(i,:), prototype_apertura) / (norm(arv_test(i,:)) * norm(prototype_apertura));
    cos_sim_chiusura = dot(arv_test(i,:), prototype_chiusura) / (norm(arv_test(i,:)) * norm(prototype_chiusura));
    
    % Assegna la classe in base alla similarità più alta
    labels_test(i) = cos_sim_apertura > cos_sim_chiusura; % 1 per apertura, 0 per chiusura
end

% Visualizzazione dei risultati
fprintf('Classificazione completata.\n');
disp(['Numero di aperture classificate: ', num2str(sum(labels_test == 1))]);
disp(['Numero di chiusure classificate: ', num2str(sum(labels_test == 0))]);

accuracy = sum(labels_test == true_labels) / length(true_labels);
fprintf('Accuratezza classificazione: %.2f%%\n', accuracy * 100)