clc
clear all
close all

%% Caricamento e organizzazione dei dati
fs = 2048;

% Caricamento segnali
load("InterferenceSignal_Fatigue_h3_M1.mat")
load("InterferenceSignal_Fatigue_h3_M2.mat")

% Crea una struttura per salvare i risultati
forces = [20, 40, 60, 80];
for i = 1:length(forces)
    f = forces(i);
    
    % Accesso dinamico alla variabile
    varName = sprintf('IntSig_h3_M1_Force%d', f);
    signal = eval(varName);  % ottieni la matrice 11x5x2048

    % Estrai M1: righe 1–5
    M1 = signal(1:5, :, :);  % (5×5×2048)
    
    % Estrai i 5 canali lungo la fibra (colonne) da riga 1 (lontana da M2)
    longitudinal_M1 = squeeze(M1(1, :, :));  % (5×2048)
    
    % Calcola singolo e doppio differenziale
    SD_M1 = diff(longitudinal_M1, 1, 1);  % (4×2048)
    DD_M1 = diff(longitudinal_M1, 2, 1);  % (3×2048)

    % Salva tutto in una struct per organizzazione
    Data(i).force = f;
    Data(i).longitudinal = longitudinal_M1;
    Data(i).SD = SD_M1;
    Data(i).DD = DD_M1;
end


%% Spectral Marching per stima della velocità di conduzione, diagramma a barre con error bar e fatigue plot per le 4 contrazioni
% Parametri
epoch_length = 0.250;  % secondi
start = 1.25;          % per delay()
IED = 5e-3;            % distanza elettrodica (5 mm)
fs = 2048;
N = round(epoch_length * fs);  % campioni per epoca

% Nomi per i livelli di forza
forces = [20, 40, 60, 80];
labels = {'20%', '40%', '60%', '80%'};
min_epochs = inf;

% Primo passaggio: trova il numero minimo di epoche disponibili
for i = 1:length(forces)
    DD = Data(i).DD;
    n_epoch = floor(size(DD, 2) / N);
    if n_epoch < min_epochs
        min_epochs = n_epoch;
    end
end

% Preallocazione
rmsv = nan(min_epochs, length(forces));
arv = nan(min_epochs, length(forces));
fmeanv = nan(min_epochs, length(forces));
fmedianv = nan(min_epochs, length(forces));
cv = nan(min_epochs, length(forces));

% Loop su ciascun livello di forza
for i = 1:length(forces)
    SD = Data(i).SD;  % 4×2048
    DD = Data(i).DD;  % 3×2048
    fprintf('\n\nAnalisi di Force %d%%\n', forces(i));

    for ep = 1:min_epochs
        idx_start = (ep - 1) * N + 1;
        idx_end = ep * N;

        x = SD(1, idx_start:idx_end);  % singolo canale SD

        % Parametri
        rmsv(ep, i) = rms(x);
        arv(ep, i) = mean(abs(x));
        fmeanv(ep, i) = fmean(x, fs, N);
        fmedianv(ep, i) = fmedian(x, fs, epoch_length);

        % CV da 2 canali DD
        xd1 = DD(2, idx_start:idx_end);
        xd2 = DD(1, idx_start:idx_end);

        fft1 = fft(xd1);
        fft2 = fft(xd2);

        tmp = delay(real(fft1), imag(fft1), real(fft2), imag(fft2), start);
        delay_seconds = tmp / fs;
        cv(ep, i) = IED / delay_seconds;
    end
end

% MEDIA E DEV.STD CV
media_cv = mean(cv);
errore_cv = std(cv);

% Grafico a barre CV
figure();
bar(media_cv, 'FaceColor', '#87CEEB', 'EdgeColor', 'black');
hold on;
errorbar(1:4, media_cv, errore_cv, '.', 'Color', 'black', 'LineWidth', 1.5);
set(gca, 'XTick', 1:4, 'XTickLabel', labels);
xlabel('Intensità Contrazione');
ylabel('Velocità di Conduzione (m/s)');
title('Velocità di Conduzione - Simulazione');
hold off;

% Fatigue plots normalizzati
for p = 1:length(forces)
    figure();
    ticks = 0.6:0.1:1.9;hold on,
    axis([0 n_epoch 0.5 2.0]);axis('off');
    axes('Xlim', [0 n_epoch], 'Ylim', [0.5 2.0], 'YTick', ticks, 'YGrid', 'on');% Normalizzazione rispetto al valore iniziale
    plot(fmeanv(:,p) / fmeanv(1,p), 'r');hold on
    plot(fmedianv(:,p) / fmedianv(1,p),'y');hold on
    plot(rmsv(:,p) / rmsv(1,p), 'g');hold on
    plot(arv(:,p) / arv(1,p),'k');hold on
    plot(cv(:,p) / cv(1,p), 'b');hold off;
    legend('fmedia','fmediana','rms','arv','cv');title(['Fatigue plot - ', num2str(forces(p)), '%']);
    xlabel('tempo (s)');
end 

%% Rette interpolanti

time = (1:min_epochs)';

% Inizializzazione strutture per i fit lineari
fit_fmeanv   = zeros(min_epochs, length(forces));
fit_fmedianv = zeros(min_epochs, length(forces));
fit_rmsv     = zeros(min_epochs, length(forces));
fit_arv      = zeros(min_epochs, length(forces));
fit_cv       = zeros(min_epochs, length(forces));

% Calcolo fit lineare per tutte le contrazioni
for p = 1:length(forces)
    fit_fmeanv(:,p)   = polyval(polyfit(time, fmeanv(:,p), 1), time);
    fit_fmedianv(:,p) = polyval(polyfit(time, fmedianv(:,p), 1), time);
    fit_rmsv(:,p)     = polyval(polyfit(time, rmsv(:,p), 1), time);
    fit_arv(:,p)      = polyval(polyfit(time, arv(:,p), 1), time);
    fit_cv(:,p)       = polyval(polyfit(time, cv(:,p), 1), time);
end

% Colori differenti per ogni livello di forza
colori = {'r', 'g', 'b', 'k'};  % rosso, verde, blu, nero

figure('Name', 'Rette interpolanti - Tutte le variabili', 'Position', [100, 100, 1400, 300]);

variabili = {fit_rmsv, fit_arv, fit_cv, fit_fmeanv, fit_fmedianv};
nomi_variabili = {'RMS', 'ARV', 'CV', 'Fmean', 'Fmedian'};

for i = 1:5
    subplot(1, 5, i); hold on;
    for p = 1:length(forces)
        % Normalizzazione rispetto al primo valore di ogni fit
        base_val = variabili{i}(1, p);
        if base_val == 0 || isnan(base_val)
            base_val = 1;
        end
        plot(time, variabili{i}(:,p) / base_val, colori{p}, 'LineWidth', 1.5);
    end

    title(nomi_variabili{i});
    xlabel('Epoche');
    if i == 5
        legend(arrayfun(@(x) sprintf('%d%% Forza', x), forces, 'UniformOutput', false), 'Location', 'best');
    end
end

%% Separazioni sorgenti con PCA
sorgente_1   = squeeze(IntSig_h3_M1_Force80(1,1,:));   % elettrodo M1
sorgente_2   = squeeze(IntSig_h3_M2_Force80(11,1,:));  % elettrodo M2
crosstalk_1  = squeeze(IntSig_h3_M2_Force20(1,1,:));   % crosstalk da M2 a M1
crosstalk_2  = squeeze(IntSig_h3_M1_Force20(11,1,:));  % crosstalk da M1 a M2

sorgente_1=sorgente_1';
sorgente_2=sorgente_2';
crosstalk_1=crosstalk_1';
crosstalk_2=crosstalk_2';

input_signal=[sorgente_1; sorgente_2];

for i=1:2
    input_signal(i,:)=input_signal(i,:)-mean(input_signal(i,:)); %tolgo la media
    input_signal(i,:)=input_signal(i,:)/std(input_signal(i,:));   % normalizzo
end
% faccio in modo che i 2 segnali siano ortogonali, rende i due segnali scorrelati 
input_signal(1,:)=input_signal(1,:)-(input_signal(1,:)*input_signal(2,:)')/(input_signal(2,:)*input_signal(2,:)')*input_signal(2,:);


%definisco il modello di mixing e il rumore
N=2;

%M=randn(N,2); %matrice di mixing casuale
M=[.25 .25;-0.1 1]; %matrice di mixing scelta dall'utente e fissa

%Miscele
segnale_con_crosstalk=M*input_signal; %noisy input generation

%applico la PCA come tecnica di separazione delle sorgenti
[coeff,s,l]=pca(segnale_con_crosstalk');

output=coeff*segnale_con_crosstalk; %PCA output, calcolare le componenti principali in funzione del tempo, stime delle sorgenti

for i=1:2;output(i,:)=output(i,:)/std(output(i,:));end



% % Mettere in ordine le componenti in base alla correlazione con le sorgenti
RR=input_signal*output(1:2,:)'/length(input_signal);

I=1:2;
[~,J(1)]=max(abs(RR(1,:)));
[~,J(2)]=max(abs(RR(2,:)));
for i=1:2
     Y(i,:)=round(RR(I(i),J(i)))*output(J(i),:);
end


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
