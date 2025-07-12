%% Error bar

%Bicipite

%cv2,cv4,cv6,cv8 sono da sostiuire con i vettori delle velocità di conduzione per diversi
%livelli di intansità del bicipite

%medio per diversi kg
media_cv_2=mean(cv2);
media_cv_4=mean(cv4);
media_cv_6=mean(cv6);
media_cv_8=mean(cv8);
error_cv_2=std(cv2)/sqrt(length(cv2)); %calcolo l'errore standard
error_cv_4=std(cv4)/sqrt(length(cv4));
error_cv_6=std(cv6)/sqrt(length(cv6));
error_cv_8=std(cv8)/sqrt(length(cv8));


CV = [media_cv_2, media_cv_4, media_cv_6, media_cv_8]; % Velocità di conduzione
errori = [error_cv_2, error_cv_4, error_cv_6, error_cv_8]; % Barre di errore
labels = {'2 kg', '4 kg', '6 kg ', '8 kg'}; % Etichette

% Creazione del grafico a barre
figure()
bar(CV, 'FaceColor', '#87CEEB', 'EdgeColor', 'black'); % Diagramma a barre
hold on;
errorbar(1:length(CV), CV, errori, '.', 'Color', 'black', 'LineWidth', 1.5); % Barre di errore

% Impostazioni visive
set(gca, 'XTick', 1:length(labels), 'XTickLabel', labels); % Etichette sull'asse x
xlabel('Intensità');
ylabel('Velocità di Conduzione');
title('Diagramma a Barre per la CV del bicipite');

hold off;

%Tricipite

%cv2,cv4,cv6,cv8 sono da sostiuire con i vettori delle velocità di conduzione per diversi
%livelli di intansità del tricipite

%medio per diversi kg
media_cv_2=mean(cv2);
media_cv_4=mean(cv4);
media_cv_6=mean(cv6);
media_cv_8=mean(cv8);
error_cv_2=std(cv2)/sqrt(length(cv2)); %calcolo l'errore standard
error_cv_4=std(cv4)/sqrt(length(cv4));
error_cv_6=std(cv6)/sqrt(length(cv6));
error_cv_8=std(cv8)/sqrt(length(cv8));


CV = [media_cv_2, media_cv_4, media_cv_6, media_cv_8]; % Velocità di conduzione
errori = [error_cv_2, error_cv_4, error_cv_6, error_cv_8]; % Barre di errore
labels = {'2 kg', '4 kg', '6 kg ', '8 kg'}; % Etichette

% Creazione del grafico a barre
figure()
bar(CV, 'FaceColor', '#87CEEB', 'EdgeColor', 'black'); % Diagramma a barre
hold on;
errorbar(1:length(CV), CV, errori, '.', 'Color', 'black', 'LineWidth', 1.5); % Barre di errore

% Impostazioni visive
set(gca, 'XTick', 1:length(labels), 'XTickLabel', labels); % Etichette sull'asse x
xlabel('Intensità');
ylabel('Velocità di Conduzione');
title('Diagramma a Barre per la CV del tricipite');

hold off;

