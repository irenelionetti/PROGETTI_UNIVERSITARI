clc, clear all, close all

% i primi 8 canali sono elettrodi su tricipite e gli ultimi 8 canali sono elettrodi su bicipite (segnali gruppo 3-10)

% Carico le matrici dei segnali
load('prova_bicipite_2.mat');
prova_bicipite_2=data;
load('prova_bicipite_4.mat');
prova_bicipite_4=data;
load('prova_bicipite_6.mat');
prova_bicipite_6=data;
load('prova_bicipite_8.mat');
prova_bicipite_8=data;
load('prova_tricipite_2.mat');
prova_tricipite_2=data;
load('prova_tricipite_4.mat');
prova_tricipite_4=data;
load('prova_tricipite_6.mat');
prova_tricipite_6=data;
load('prova_tricipite_8.mat');
prova_tricipite_8=data;


% Definizione dei nomi dei file e dei campi della struttura
nomi_file = {'prova_bicipite_2', 'prova_bicipite_4', 'prova_bicipite_6', 'prova_bicipite_8', ...
             'prova_tricipite_2', 'prova_tricipite_4', 'prova_tricipite_6', 'prova_tricipite_8'};

% Struttura per contenere i dati
segnali = struct();

% Caricamento dei dati
for i = 1:length(nomi_file)
    load([nomi_file{i}, '.mat']);  % Carica la variabile "data"
    segnali.(nomi_file{i}) = data; % Salva i dati nella struttura con il nome corrispondente
end

%% 1 e 7. Filtrare opportunamente i dati raccolti dal bicipite (non considerando i primi 2 s), filtrare opportunamente i dati raccolti dal tricipite (non considerando i primi 2 s)

fs = 2048; %Hz
N_remove = 2 * fs; % Numero di campioni da rimuovere

% Rimuove il transitorio di 2 s per tutti i segnali
campi = fieldnames(segnali);
for i = 1:length(campi)
    segnali.(campi{i}) = segnali.(campi{i})(:, N_remove+1:end);
end

% Creazione dell'asse temporale per il primo segnale (assumendo stessa lunghezza per tutti)
t = (0:size(segnali.prova_bicipite_2, 2)-1) / fs; 

% plot del segnale grezzo del bicipite 8 kg su tutti i canali 
MyPlot(figure,[1:length(segnali.prova_bicipite_8(1,:))]/fs,segnali.prova_bicipite_8,50);title('Segnale grezzo bicipite 8 kg su tutti i canali');xlabel('Tempo (s)');ylabel('Ampiezza (mV)');

% plot del segnale grezzo del tricipite 8 kg su tutti i canali 
MyPlot(figure,[1:length(segnali.prova_tricipite_8(1,:))]/fs,segnali.prova_tricipite_8,50);title('Segnale grezzo tricipite 8 kg su tutti i canali');xlabel('Tempo (s)');ylabel('Ampiezza (mV)');


% PSD del segnale grezzo del bicipite con 2 kg visto dal canale 9
[Pxx,f] = pwelch(segnali.prova_bicipite_2(9,:)-mean(segnali.prova_bicipite_2(9,:)),hamming(1024),512,2048,fs); 
figure()
plot(f,Pxx/(max(Pxx)))
axis([0 fs/2 0 1])
title('PSD del segnale grezzo del bicipite con 2 kg visto dal canale 9');
xlabel('Frequenza (Hz)');
ylabel('PSD normalizzata (adimensionale)');
%sicuramente presente interferenza di rete a 50 Hz, fare anche un passa-alto per artefatto da movimento

% PSD del segnale grezzo del tricipite con 2 kg visto dal canale 1
[Pxx,f] = pwelch(segnali.prova_tricipite_2(1,:)-mean(segnali.prova_tricipite_2(1,:)),hamming(1024),512,2048,fs); 
figure()
plot(f,Pxx/(max(Pxx)))
axis([0 fs/2 0 1])
title('PSD del segnale grezzo del tricipite con 2 kg visto dal canale 1')
xlabel('Frequenza (Hz)');
ylabel('PSD normalizzata (adimensionale)');


% RIMOZIONE INTERFERENZA DI RETE e sue armoniche dove nella PSD è presente un picco
% Definizione delle frequenze di interferenza da eliminare
disturbo_freqs = [50, 150, 250, 350];

%funzione rico per la rimozione dell'interferenza
for f_disturbo = disturbo_freqs
    [cMA, cAR] = rico(0.01, 2, f_disturbo, 1/fs); % Calcolo dei coefficienti del filtro per la frequenza desiderata
    
    % Applicazione del filtro a tutti i segnali
    for i = 1:length(campi)
        for ii = 1:size(segnali.(campi{i}), 1)
            segnali.(campi{i})(ii,:) = filtfilt(cMA, cAR, segnali.(campi{i})(ii,:));
        end
    end
end

% PSD del segnale del bicipite con 2 kg visto dal canale 9 senza interferenza di rete
[Pxx,f] = pwelch(segnali.prova_bicipite_2(9,:)-mean(segnali.prova_bicipite_2(9,:)),hamming(1024),512,2048,fs); 
figure()
plot(f,Pxx/(max(Pxx)))
axis([0 fs/2 0 1])
title('PSD del segnale del bicipite 2 kg - ch 9 - senza interferenza di rete')
xlabel('Frequenza (Hz)');
ylabel('PSD normalizzata (adimensionale)');

% PSD del segnale del tricipite con 2 kg visto dal canale 1 senza interferenza di rete
[Pxx,f] = pwelch(segnali.prova_tricipite_2(1,:)-mean(segnali.prova_tricipite_2(1,:)),hamming(1024),512,2048,fs); 
figure()
plot(f,Pxx/(max(Pxx)))
axis([0 fs/2 0 1])
title('PSD del segnale del tricipite 2 kg - ch 1 - senza interferenza di rete')
xlabel('Frequenza (Hz)');
ylabel('PSD normalizzata (adimensionale)');

%RIMOZIONE ARTEFATTO DA MOVIMENTO IN BASSA E ALTA FREQUENZA
% Creazione del filtro passa-alto 10 Hz (per rimuovere artefatti da movimento)
[bHPF, aHPF] = butter(2, 10/(fs/2), 'high');
figure;
freqz(bHPF, aHPF, 1024, fs);
title('Risposta in frequenza del filtro passa-alto (10 Hz)');

% Creazione del filtro passa-basso 350 Hz (per rimuovere rumore ad alta frequenza)
[bLPF, aLPF] = butter(2, 350/(fs/2), 'low');
figure;
freqz(bLPF, aLPF, 1024, fs);
title('Risposta in frequenza del filtro passa-basso (350 Hz)');

% Applica i filtri passa-alto e passa-basso a tutti i segnali
campi = fieldnames(segnali);
for i = 1:length(campi)
    for ii = 1:size(segnali.(campi{i}), 1)
        % Passa-alto
        segnali.(campi{i})(ii,:) = filtfilt(bHPF, aHPF, segnali.(campi{i})(ii,:));
        % Passa-basso
        segnali.(campi{i})(ii,:) = filtfilt(bLPF, aLPF, segnali.(campi{i})(ii,:));
    end
end


% PSD del segnale filtrato del bicipite con 2 kg visto dal canale 9
[Pxx,f] = pwelch(segnali.prova_bicipite_2(9,:)-mean(segnali.prova_bicipite_2(9,:)),hamming(1024),512,2048,fs); 
figure()
plot(f,Pxx/(max(Pxx)))
axis([0 fs/2 0 1])
title('PSD del segnale filtrato del bicipite 2 kg visto dal ch 9')
xlabel('Frequenza (Hz)');
ylabel('PSD normalizzata (adimensionale)');

% PSD del segnale filtrato del tricipite con 2 kg visto dal canale 1
[Pxx,f] = pwelch(segnali.prova_tricipite_2(1,:)-mean(segnali.prova_tricipite_2(1,:)),hamming(1024),512,2048,fs); 
figure()
plot(f,Pxx/(max(Pxx)))
axis([0 fs/2 0 1])
title('PSD del segnale filtrato del tricipite 2 kg visto dal ch 1');
xlabel('Frequenza (Hz)');
ylabel('PSD normalizzata (adimensionale)');


%grafico per tutti i canali bicipite FILTRATO
MyPlot(figure,[1:length(segnali.prova_bicipite_8(1,:))]/fs,segnali.prova_bicipite_8,50);title('Segnale filtrato contrazione bicipite 8 kg');xlabel('Tempo (s)');ylabel('Ampiezza (mV)');

%grafico per tutti i canali tricipite FILTRATO
MyPlot(figure,[1:length(segnali.prova_tricipite_8(1,:))]/fs,segnali.prova_tricipite_8,50);title('Segnale filtrato contrazione tricipite 8 kg');xlabel('Tempo (s)');ylabel('Ampiezza (mV)');

%% Creazione di strutture separate per bicipite e tricipite

%segnali bicipite conterrà solo i canali che rappresentano gli elettrodi
%sul bicipite, idem per tricipite
segnali_bicipite = struct();
segnali_tricipite = struct();

% % Separazione dei segnali
campi = fieldnames(segnali);
for i = 1:length(campi)
    segnali_tricipite.(campi{i}) = segnali.(campi{i})(1:8, :); % Primi 8 canali -> Tricipite
    segnali_bicipite.(campi{i}) = segnali.(campi{i})(9:16, :); % Ultimi 8 canali -> Bicipite
end

%% 2. Calcolo del Singolo Differenziale (SD) e Doppio Differenziale (DD)
segnali_SD_bicipite = struct();
segnali_DD_bicipite = struct();
segnali_SD_tricipite = struct();
segnali_DD_tricipite = struct();

campi_bi=fieldnames(segnali_bicipite);
campi_tri=fieldnames(segnali_tricipite);
for i = 1:length(campi_bi)
    % Bicipite
    SD_bic = diff(segnali_bicipite.(campi_bi{i}), 1, 1); % Singolo differenziale
    DD_bic = diff(SD_bic, 1, 1);  % Doppio differenziale
    segnali_SD_bicipite.(campi_bi{i}) = SD_bic;
    segnali_DD_bicipite.(campi_bi{i}) = DD_bic;

    % Tricipite
    SD_tri = diff(segnali_tricipite.(campi_tri{i}), 1, 1); % Singolo differenziale
    DD_tri = diff(SD_tri, 1, 1);  % Doppio differenziale
    segnali_SD_tricipite.(campi_tri{i}) = SD_tri;
    segnali_DD_tricipite.(campi_tri{i}) = DD_tri;
end

disp('Calcolo di SD e DD completato!');

%grafico tutti i canali per bicipite singolo differenziale e doppio differenziale contrazione 8 kg
MyPlot(figure,[1:length(segnali_SD_bicipite.prova_bicipite_8(1,:))]/fs,segnali_SD_bicipite.prova_bicipite_8,50);title('Segnali SD contrazione bicipite 8kg');xlabel('Tempo (s)');ylabel('Ampiezza (mV)');
MyPlot(figure,[1:length(segnali_DD_bicipite.prova_bicipite_8(1,:))]/fs,segnali_DD_bicipite.prova_bicipite_8,50);title('Segnali DD contrazione bicipite 8kg');xlabel('Tempo (s)');ylabel('Ampiezza (mV)');

%grafico per tutti i canali per tricipite singolo differenziale e doppio differenziale contrazione 8 kg
MyPlot(figure,[1:length(segnali_SD_tricipite.prova_tricipite_8(1,:))]/fs,segnali_SD_tricipite.prova_tricipite_8,50);title('Segnali SD contrazione tricipite 8kg');xlabel('Tempo (s)');ylabel('Ampiezza (mV)');
MyPlot(figure,[1:length(segnali_DD_tricipite.prova_tricipite_8(1,:))]/fs,segnali_DD_tricipite.prova_tricipite_8,50);title('Segnali DD contrazione tricipite 8kg');xlabel('Tempo (s)');ylabel('Ampiezza (mV)');


%% 3. Stimare la densità spettrale di potenza PSD per le diverse contrazioni
% con il metodo più opportuno. Considerare epoche di 250 ms e le tre
% tipologie di segnali: monopolare, SD, DD. Quali modifiche subiscono gli
% spettri a seguito dell'applizacione del filtro sapziale? Mostrare i
% risultati tramite un subplot da 4 righe (contrazioni) e 3 colonne
% (tipologia del segnale)

% Lunghezza epoca in campioni
epoch_length = 250e-3; % 250 ms
nperseg = epoch_length * fs; % Numero di campioni per epoca

% Definizione delle contrazioni
contrazioni = {'prova_bicipite_2', 'prova_bicipite_4', 'prova_bicipite_6', 'prova_bicipite_8'};
tipologie = {'Monopolare', 'Singolo Differenziale (SD)', 'Doppio Differenziale (DD)'};

% Selezione del canale rappresentativo 
ch_bic = 2;
ch_tri = 2;    

% Creazione della figura per i subplot (Bicipite)
figure;
sgtitle('PSD per diverse contrazioni col bicipite - elettrodi su bicipite');

for i = 1:length(contrazioni)
    % Ottieni i segnali del bicipite
    segnale_mono = segnali_bicipite.(contrazioni{i});
    segnale_SD = segnali_SD_bicipite.(contrazioni{i});
    segnale_DD = segnali_DD_bicipite.(contrazioni{i});
    
    % Calcola la PSD con Welch per ciascuna tipologia
    [Pxx_mono, f1] = pwelch(segnale_mono(ch_bic,:) - mean(segnale_mono(ch_bic,:)), hamming(nperseg), 0, 2048, fs);
    [Pxx_SD, f2] = pwelch(segnale_SD(ch_bic,:) - mean(segnale_SD(ch_bic,:)), hamming(nperseg), 0, 2048, fs);
    [Pxx_DD, f3] = pwelch(segnale_DD(ch_bic,:) - mean(segnale_DD(ch_bic,:)), hamming(nperseg), 0, 2048, fs);
    
    % Plot Monopolare
    subplot(4,3, (i-1)*3 + 1);
    plot(f1, Pxx_mono/(max(Pxx_mono)));
    title(['Contrazione: ', strrep(contrazioni{i}, '_', '\_'), ' - ', tipologie{1}]);
    xlabel('Frequenza (Hz)');
    ylabel('PSD Normalizzata');
    grid on;
    
    % Plot SD
    subplot(4,3, (i-1)*3 + 2);
    plot(f2, Pxx_SD/(max(Pxx_SD)));
    title(['Contrazione: ', strrep(contrazioni{i}, '_', '\_'), ' - ', tipologie{2}]);
    xlabel('Frequenza (Hz)');
    ylabel('PSD Normalizzata');
    grid on;
    
    % Plot DD
    subplot(4,3, (i-1)*3 + 3);
    plot(f3, Pxx_DD/(max(Pxx_DD)));
    title(['Contrazione: ', strrep(contrazioni{i}, '_', '\_'), ' - ', tipologie{3}]);
    xlabel('Frequenza (Hz)');
    ylabel('PSD Normalizzata');
    grid on;
end



% contrazioni del tricipite
% Definizione delle contrazioni
contrazioni = {'prova_tricipite_2', 'prova_tricipite_4', 'prova_tricipite_6', 'prova_tricipite_8'};
tipologie = {'Monopolare', 'Singolo Differenziale (SD)', 'Doppio Differenziale (DD)'};


% Creazione della figura per i subplot (Tricipite)
figure;
sgtitle('PSD per diverse contrazioni col tricipite - elettrodi sul Tricipite');

for i = 1:length(contrazioni)
    % Ottieni i segnali del tricipite
    segnale_mono = segnali_tricipite.(contrazioni{i});
    segnale_SD = segnali_SD_tricipite.(contrazioni{i});
    segnale_DD = segnali_DD_tricipite.(contrazioni{i});
    
    % Calcola la PSD con Welch per ciascuna tipologia
    [Pxx_mono, f1] = pwelch(segnale_mono(ch_tri,:) - mean(segnale_mono(ch_tri,:)), hamming(nperseg), 0, 2048, fs);
    [Pxx_SD, f2] = pwelch(segnale_SD(ch_tri,:) - mean(segnale_SD(ch_tri,:)), hamming(nperseg), 0, 2048, fs);
    [Pxx_DD, f3] = pwelch(segnale_DD(ch_tri,:) - mean(segnale_DD(ch_tri,:)), hamming(nperseg), 0, 2048, fs);
    
    % Plot Monopolare
    subplot(4,3, (i-1)*3 + 1);
    plot(f1, Pxx_mono/(max(Pxx_mono)));
    title(['Contrazione: ', strrep(contrazioni{i}, '_', '\_'), ' - ', tipologie{1}]);
    xlabel('Frequenza (Hz)');
    ylabel('PSD Normalizzata');
    grid on;
    
    % Plot SD
    subplot(4,3, (i-1)*3 + 2);
    plot(f2, Pxx_SD/(max(Pxx_SD)));
    title(['Contrazione: ', strrep(contrazioni{i}, '_', '\_'), ' - ', tipologie{2}]);
    xlabel('Frequenza (Hz)');
    ylabel('PSD Normalizzata');
    grid on;
    
    % Plot DD
    subplot(4,3, (i-1)*3 + 3);
    plot(f3, Pxx_DD/(max(Pxx_DD)));
    title(['Contrazione: ', strrep(contrazioni{i}, '_', '\_'), ' - ', tipologie{3}]);
    xlabel('Frequenza (Hz)');
    ylabel('PSD Normalizzata');
    grid on;
end

%con filtro spaziale acquisizione più selettiva, si nota uno spostamento verso le frequenze più alte->meno rumore


%% 4 e 5. Spectral Marching e metodo multicanale per stima della velocità di conduzione, diagramma a barre con error bar e fatigue plot per le 4 contrazioni bicipite
epoch_length = 250e-3; % 250 ms
start = 1.25;          % punto iniziale per la funzione delay
IED = 5e-3;            % distanza elettrodica
fprintf('\n');

prove = {'prova_bicipite_2', 'prova_bicipite_4', 'prova_bicipite_6', 'prova_bicipite_8'};
N = round(epoch_length * fs);   % campioni per epoca
min_epochs = 1000;                 % inizializzo il minimo numero di epoche ad un n uemro elevato

% Primo passaggio: determinare il numero massimo di epoche
for p = 1:length(prove)
    nome_prova = prove{p};
    n_epoch = floor(size(segnali_DD_bicipite.(nome_prova), 2) / N);
    if n_epoch < min_epochs
        min_epochs = n_epoch;
    end
end

% Prealloca le matrici: righe = epoche, colonne = prove
fmeanv   = nan(min_epochs, length(prove));
fmedianv = nan(min_epochs, length(prove));
rmsv     = nan(min_epochs, length(prove));
arv      = nan(min_epochs, length(prove));
cv       = nan(min_epochs, length(prove));

% Loop su ogni prova
for p = 1:length(prove)
    nome_prova = prove{p};
    fprintf('\n\nAnalisi di: %s\n', nome_prova);

    for i = 1:min_epochs
        fprintf('\n Processing epoch # %d', i);

        idx_start = 1 + (i - 1) * N;
        idx_end   = i * N;

        % Segnale singolo differenziale
        x = segnali_SD_bicipite.(nome_prova)(1, idx_start:idx_end);

        rmsv(i, p) = sqrt(1/length(x)*sum(abs(x).^2));
        arv(i, p) = mean(abs(x));
        fmeanv(i, p) = fmean(x, fs, N);
        fmedianv(i, p) = fmedian(x, fs, epoch_length);

        % Conduction Velocity (CV)
        xd1 = segnali_DD_bicipite.(nome_prova)(4, idx_start:idx_end); % il segnale cambia polarità --> tra i ch 2 e 3 probabilmente c'è la ZI
        xd2 = segnali_DD_bicipite.(nome_prova)(3, idx_start:idx_end);

        fft1r = real(fft(xd1)); fft1i = imag(fft(xd1));
        fft2r = real(fft(xd2)); fft2i = imag(fft(xd2));

        tmp = delay(fft1r, fft1i, fft2r, fft2i, start);
        tmp_secondi = tmp / fs;
        cv(i, p) = IED / tmp_secondi;
    end
end

% stima di CV con metodo multicanale per contrazione 8 kg del bicipite
addpath('.\CV_multich')

CV_multich_8=mle_CV_est(segnali_SD_bicipite.prova_bicipite_8, IED, fs);
CV_multich_6=mle_CV_est(segnali_SD_bicipite.prova_bicipite_6, IED, fs);
CV_multich_4=mle_CV_est(segnali_SD_bicipite.prova_bicipite_4, IED, fs);
CV_multich_2=mle_CV_est(segnali_SD_bicipite.prova_bicipite_2, IED, fs);
CV_multich= [CV_multich_2; CV_multich_4; CV_multich_6; CV_multich_8 ];

%Diagramma a barre con relativo error bar dove in ascissa sono riportato le
%informazioni sulle intensità della contrazione

media_cv = mean(cv);   % 1 x 4 -> media per ciascun livello di intensità
errore_cv = std(cv) ;  % 1 x 4 -> deviazione standard

% Etichette
labels = {'2 kg', '4 kg', '6 kg', '8 kg'}; 

% Crea il grafico a barre con barre di errore
figure()
bar(media_cv, 'FaceColor', '#87CEEB', 'EdgeColor', 'black'); 
hold on;
errorbar(1:length(media_cv), media_cv, errore_cv, '.', 'Color', 'black', 'LineWidth', 1.5); 

set(gca, 'XTick', 1:length(labels), 'XTickLabel', labels);
xlabel('Intensità Contrazione');
ylabel('Velocità di Conduzione (m/s)');
title('Velocità di Conduzione - Bicipite');
hold off;


% Disegno dei 4 Fatigue Plot per le contrazioni 2,4,6,8 kg
for p = 1:length(prove)
    figure;
    ticks = 0.6:0.1:1.9;hold on,
    axis([0 n_epoch 0.5 2.0]);axis('off');
    axes('Xlim', [0 n_epoch], 'Ylim', [0.5 2.0], 'YTick', ticks, 'YGrid', 'on');% Normalizzazione rispetto al valore iniziale
    plot(fmeanv(:,p) / fmeanv(1,p), 'r');hold on
    plot(fmedianv(:,p) / fmedianv(1,p),'y');hold on
    plot(rmsv(:,p) / rmsv(1,p), 'g');hold on
    plot(arv(:,p) / arv(1,p),'k');hold on
    plot(cv(:,p) / cv(1,p), 'b');hold off;
    legend('fmedia','fmediana','rms','arv','cv');title(['Fatigue plot - ', strrep(prove{p}, '_', '\_')]);
    xlabel('tempo (s)');
end 






%% 6. Realizzare 5 grafici per ARV, RMS, MDF, MNF e CV con le diverse contrazioni per bicipite
time = (1:min_epochs)';

% Inizializzazione strutture per i fit
fit_fmeanv   = zeros(min_epochs, length(prove));
fit_fmedianv = zeros(min_epochs, length(prove));
fit_rmsv     = zeros(min_epochs, length(prove));
fit_arv      = zeros(min_epochs, length(prove));
fit_cv       = zeros(min_epochs, length(prove));

% Calcolo dei fit per tutte le contrazioni
for p = 1:length(prove)
    % Fit lineare per ciascuna variabile
    fit_fmeanv(:,p)   = polyval(polyfit(time, fmeanv(:,p), 1), time);
    fit_fmedianv(:,p) = polyval(polyfit(time, fmedianv(:,p), 1), time);
    fit_rmsv(:,p)     = polyval(polyfit(time, rmsv(:,p), 1), time);
    fit_arv(:,p)      = polyval(polyfit(time, arv(:,p), 1), time);
    fit_cv(:,p)       = polyval(polyfit(time, cv(:,p), 1), time);
end

% Colori differenti per ogni prova
colori = {'r', 'g', 'b', 'k'};  % rosso, verde, blu, nero

figure('Name', 'Rette interpolanti - Tutte le variabili', 'Position', [100, 100, 1400, 300]);

variabili = {fit_rmsv, fit_arv, fit_cv, fit_fmeanv, fit_fmedianv};
nomi_variabili = {'RMS', 'ARV', 'CV', 'Fmean', 'Fmedian'};

for i = 1:5
    subplot(1, 5, i); hold on;
    for p = 1:length(prove)
        % Normalizzazione rispetto al primo valore
        base_val = variabili{i}(1, p);
        if base_val == 0 || isnan(base_val), base_val = 1; end
        plot(time, variabili{i}(:,p) / base_val, colori{p}, 'LineWidth', 1.5);
    end

    title(nomi_variabili{i});
    xlabel('Epoche');
    if i == 5
        legend(strrep(prove, '_', '\_'));
    end
end

%Commento: RMS e ARV crescono per tutte le prove, tranne per la contrazione
%con 2 kg che potrebbe non risultare una contrazione affaticante, la CV decresce per tutte le prove e MDF
%e MNF decrescono per tutte le prove tranne per la contrazione con 2 kg,
%dove probabilmente non si verificano fenomeni affaticanti.

%% 8. Sommare i segnali monopolari del bicipite durante una contrazione e quelli del tricipite durante un'altra contrazione in modo da simulare delle co-contrazioni

segnale_bicipite_monopolare = segnali_bicipite.prova_bicipite_8;  % bicipite contrazione 8 kg
segnale_tricipite_monopolare = segnali_bicipite.prova_tricipite_2;  % crosstalk tricipite su bicipite (elettrodi su bicipite mentre tricipite è attivo con contrazione 2 kg)

% Usiamo la lunghezza minima per adattare i segnali alla stessa lunghezza
min_len = min(length(segnale_bicipite_monopolare(1,:)), length(segnale_tricipite_monopolare(1,:)));


% Troncare i segnali alla lunghezza minima
segnale_bicipite_monopolare = segnale_bicipite_monopolare(:, 1:min_len);
segnale_tricipite_monopolare = segnale_tricipite_monopolare(:, 1:min_len);

% Sommare i segnali per ogni canale (sommare il canale i-esimo del bicipite con il canale i-esimo del tricipite)
segnale_co_contrazione = zeros(8, min_len);  % Preallocazione per il segnale di co-contrazione

for i = 1:8
    segnale_co_contrazione(i, :) = segnale_bicipite_monopolare(i, :) + segnale_tricipite_monopolare(i,:);
end


% Creare il grafico con 3 subplot
figure();

% Subplot 1: Segnale Bicipite
subplot(3, 1, 1);
plot(segnale_bicipite_monopolare(1,:), 'r', 'LineWidth', 1);
title('Bicipite (Monopolare)- canale 1');
xlabel('Campioni');
ylabel('Ampiezza [mV]');


% Subplot 2: Segnale Tricipite
subplot(3, 1, 2);
plot(segnale_tricipite_monopolare(1,:), 'b', 'LineWidth', 1);
title('Tricipite (Monopolare) - canale 1');
xlabel('Campioni');
ylabel('Ampiezza [mV]');

% Subplot 3: Somma dei segnali (Co-Contrazione)
subplot(3, 1, 3);
plot(segnale_co_contrazione(1,:), 'k', 'LineWidth', 1);
title('Co-Contrazione (Somma Bicipite + Tricipite)- canale 1');
xlabel('Campioni');
ylabel('Ampiezza [mV]');



%% 9. Ripetere i punti 2,3,4 per valutare l'effetto del crosstalk sui dati del muscolo target (bicipite)

%2.STIMA DEI SINGOLI E DOPPI DIFFERENZIALI

segnale_SD_cc = diff(segnale_co_contrazione, 1, 1);  % differenza tra righe adiacenti (canali)
segnale_DD_cc = diff(segnale_co_contrazione, 2, 1);  % differenza doppia (su righe)

%grafico tutti i canali per bicipite singolo differenziale e doppio differenziale contrazione 8 kg
MyPlot(figure,[1:length(segnale_SD_cc(1,:))]/fs,segnale_SD_cc,50);title('Segnali SD co-contrazione');xlabel('Tempo (s)');ylabel('Ampiezza (mV)');

MyPlot(figure,[1:length(segnale_DD_cc(1,:))]/fs,segnale_DD_cc,50);title('Segnali DD co-contrazione');xlabel('Tempo (s)');ylabel('Ampiezza (mV)');


%% 9.3 STIMA DELLA DENSITA' SPETTRALE DI POTENZA --> ripeto quanto fatto al punto 3 per il segnale con cross talk

% Lunghezza epoca in campioni
epoch_length = 250e-3; % 250 ms
nperseg = epoch_length * fs; % Numero di campioni per epoca

tipologie = {'Monopolare', 'Singolo Differenziale (SD)', 'Doppio Differenziale (DD)'};

% Selezione del canale rappresentativo 
ch= 2;
  
% Creazione della figura per i subplot
figure;
sgtitle('PSD per segnale di cocontrazione');

% Calcola la PSD con Welch per ciascuna tipologia
[Pxx_mono, f1] = pwelch(segnale_co_contrazione(ch,:) - mean(segnale_co_contrazione(ch,:)), hamming(nperseg), 0, 2048, fs);
[Pxx_SD, f2] = pwelch(segnale_SD_cc(ch,:) - mean(segnale_SD_cc(ch,:)), hamming(nperseg), 0, 2048, fs);
[Pxx_DD, f3] = pwelch(segnale_DD_cc(ch,:) - mean(segnale_DD_cc(ch,:)), hamming(nperseg), 0, 2048, fs);

% Plot Monopolare
subplot(1,3,1);
plot(f1, Pxx_mono/(max(Pxx_mono)));
title(['Segnale cocontrazione - ',tipologie{1}]);
xlabel('Frequenza (Hz)');
ylabel('PSD Normalizzata');
grid on;

% Plot SD
subplot(1,3,2);
plot(f2, Pxx_SD/(max(Pxx_SD)));
title(['Segnale co-contrazione - ', tipologie{2}]);
xlabel('Frequenza (Hz)');
ylabel('PSD Normalizzata');
grid on;

% Plot DD
subplot(1,3,3);
plot(f3, Pxx_DD/(max(Pxx_DD)));
title(['Segnale co-contrazione - ', tipologie{3}]);
xlabel('Frequenza (Hz)');
ylabel('PSD Normalizzata');
grid on;

%% 9.4 STIMA DELLA VELOCITA' DI CONDUZIONE E FATIGUE PLOT --> ripeto il punto 4 per il segnale con cross talk

start=1.25; %punto inziale della funzione delay mettere sempre se distanza elettrodica 5*10^-3
 IED=5*10^-3;
 fprintf('\n');
 n_epoch=floor(size(segnale_DD_cc(1,:),2)/(epoch_length*fs));
 N=round(epoch_length*fs); %campioni per ogni epoca
 fmeanv = zeros(n_epoch,1);
 fmedianv=zeros(n_epoch,1);
 rmsv=zeros(n_epoch,1);
 arv=zeros(n_epoch,1);
 cv = zeros(n_epoch,1);
 for i=1:n_epoch
   fprintf('\n Processing epoch # %d',i);
   %Isolare il segnale EMG relativo all'epoca corrente e al segnale
   %singolo differenziale
   x =segnale_SD_cc(1,1+(i-1)*N : (i*N),1); %uguale lab1 indicizzazione
   % calcolo rms
   rmsv(i) =std(x); %sqrt(1/length(x)*sum(abs(x).^2));
   arv(i)=mean(abs(x)); %valor rettificato medio  
   fmeanv(i) = fmean(x,fs,N);   
   fmedianv(i)=fmedian(x,fs,epoch_length);
  
   % calcolo CV   %Isolare i segnali EMG relativi all'epoca corrente e ai due segnali
   %doppi differenziali  
   xd1 = segnale_DD_cc(2,1+(i-1)*N : (i*N));
   xd2 = segnale_DD_cc(1,1+(i-1)*N : (i*N));   
   fft1r=real(fft(xd1));   fft1i=imag(fft(xd1));
   fft2r=real(fft(xd2));   fft2i=imag(fft(xd2));
   %Richiamare la funzione delay per trovare il tempo (IN CAMPIONI)   
   tmp = delay(fft1r,fft1i,fft2r,fft2i,start);%restituisce ritardo in campioni->
   %convertire il ritardo in secondi    
   tmp_secondi=tmp/fs; 
      %ottenere qui la velocità di conduzione (m/s)
   cv(i) = IED/tmp_secondi; 
 end
 fprintf('\n'); 
 
 % stima di CV con metodo multicanale 
CV_multich_cc=mle_CV_est(segnale_SD_cc, IED, fs)
 
%Diagramma a barre con relativo error bar dove in ascissa sono riportato le
%informazioni sulle intensità della contrazione

media_cv_cc = mean(cv);   
errore_cv_cc = std(cv) ;  


% Crea il grafico a barre con barre di errore
figure()
bar(media_cv_cc, 'FaceColor', '#87CEEB', 'EdgeColor', 'black'); 
hold on;
errorbar(1:length(media_cv_cc), media_cv_cc, errore_cv_cc, '.', 'Color', 'black', 'LineWidth', 1.5); 


xlabel('Co-contrazione');
ylabel('Velocità di Conduzione (m/s)');
title('Velocità di Conduzione - Segnale cocontrazione');
hold off;


 %-----------------------------------------------------------------------
% Disegno del Fatigue Plotfigure(),
figure()
ticks = 0.6:0.1:1.9;hold on,
axis([0 n_epoch 0.5 2.0]);axis('off');
axes('Xlim', [0 n_epoch], 'Ylim', [0.5 2.0], 'YTick', ticks, 'YGrid', 'on');% Normalizzazione rispetto al valore iniziale
plot(fmeanv / fmeanv(1), 'r');hold on
plot(fmedianv / fmedianv(1),'y');hold on
plot(rmsv / rmsv(1), 'g');hold on
plot(arv / arv(1),'k');hold on
plot(cv / cv(1), 'b');hold off;
legend('fmedia','fmediana','rms','arv','cv');title('Fatigue plot segnale cocontrazione');
xlabel('tempo (s)');

%% 10. Applicare tecniche di separazione delle sorgenti per cercare di risolvere il problema del crosstalk


segnale_bicipite= segnali_bicipite.prova_bicipite_8(1,:); %segnale sorgente di bicipite canale 1
crosstalk_1 = segnali_bicipite.prova_tricipite_2(1, :);  %segnale di crosstalk (elettrodi su bicipite e tricipite attivo)
segnale_tricipite = segnali_tricipite.prova_tricipite_8(1, :);  %segnale sorgente di tricipite canale 1
crosstalk_2= segnali_tricipite.prova_bicipite_2(1, :); %segnale di crosstalk (elettrodi su tricipite e bicipite attivo)


% Trova il minimo della lunghezza tra tre segnali
min_len = min([length(segnale_bicipite), length(crosstalk_1), length(segnale_tricipite), length(crosstalk_2)]);

% Troncare i segnali alla lunghezza minima
segnale_bicipite = segnale_bicipite(:, 1:min_len);
crosstalk_1 = crosstalk_1(:, 1:min_len);
segnale_tricipite = segnale_tricipite(:, 1:min_len);
crosstalk_2 = crosstalk_2(:, 1:min_len);

%segnali sorgente
input_signal=[segnale_bicipite; segnale_tricipite];

% Sommare i segnali per creare i segnali di co-contrazione, le miscele
segnale_con_crosstalk = zeros(2, min_len);  % Preallocazione per il segnale di co-contrazione

segnale_con_crosstalk(1, :) = segnale_bicipite + crosstalk_1;
segnale_con_crosstalk(2, :) = segnale_tricipite + crosstalk_2;


%applico la PCA come tecnica di separazione delle sorgenti
[coeff,s,l]=pca(segnale_con_crosstalk');

output=coeff*segnale_con_crosstalk; %PCA output, calcolare le componenti principali in funzione del tempo, stime delle sorgenti

for i=1:2;output(i,:)=output(i,:)/std(output(i,:));end

% Figure
% segnale 1 e 2
% devo cercare di matchare l'otput che più somiglia ai due segnali 
figure;
subplot(3,2,1);
plot(segnale_bicipite);aa=axis;%
title('Segnale bicipite originale')
subplot(3,2,2);
plot(segnale_tricipite, 'r');bb=axis;    % plot B
title('Segnale tricipite originale')

subplot(3,2,3);
for i=1:2;plot(segnale_con_crosstalk(i,:)/range(segnale_con_crosstalk(:))+i);hold on;end      % plot mixing 1
title('Miscele')
subplot(3,2,4);plot(segnale_con_crosstalk(1,:),segnale_con_crosstalk(2,:),'k.');
hold on;plot([0 coeff(1,1)],[0 coeff(1,2)],'r');plot([0 coeff(2,1)],[0 coeff(2,2)],'r');axis equal
title('Scatter plot e direzioni di PCs') % nuvola dei punti con i coeff stimati dalla PCA che vanno nella direzione di massima varianza e in quella ortogonale



% % Mettere in ordine le componenti in base alla correlazione con le sorgenti
RR=input_signal*output(1:2,:)'/length(input_signal);

I=1:2;
[~,J(1)]=max(abs(RR(1,:)));
[~,J(2)]=max(abs(RR(2,:)));
for i=1:2
     Y(i,:)=round(RR(I(i),J(i)))*output(J(i),:);
end

subplot(3,2,5);
plot(Y(1,:),'b'); axis(aa)
title('Componente 1') %Component 1

subplot(3,2,6);
plot(Y(2,:),'r');axis(bb)
title('Componente 2') %Component 2


mse = zeros(1,2); % Preallocazione

secondo = 1;
samples_to_plot = fs * secondo;

for i = 1:2
    % Calcolo dell'MSE
    mse(i) = mean((input_signal(i,:) - Y(i,:)).^2);

    % Plot
    figure;
    plot(input_signal(i,1:samples_to_plot), 'r'); hold on
    plot(Y(i,1:samples_to_plot), 'b')

    title(['Componente ', num2str(i), ' - MSE: ', num2str(mse(i), '%.4f')])
    legend('Segnale originale', ...
           'Componente ricostruita da PCA')
    xlabel('Campioni')
end

% Calcola la varianza dei segnali originali
var_signal = var(input_signal, 0, 2);  % varianza per ogni riga (canale)


% Calcola R^2 per ogni canale
R2 = 1 - mse ./ var_signal';

for i = 1:2
    fprintf('Segnale %d: Varianza originale = %.4f, MSE = %.4f, R^2 = %.4f\n', ...
        i, var_signal(i), mse(i), R2(i));
end
