function fmedianv=fmedian(x,fsamp,epoc_len)
NFFT=fsamp*epoc_len;
x=x-mean(x);
[Pxx,f]=pwelch(x,boxcar(length(x)),0,NFFT,fsamp);
%periodogramma semplice finestrato con finestra rettangolare lunghezza pari ad x
Area=sum(Pxx/2); %calcolo la metà dell'area di tutta la stima
i=1; %inizializza indice vettore della Pxx
P=0; %inizializzatore di potenza
while P<=Area
    P=P+Pxx(i);
    i=i+1;
end
fmedianv=f(i);
end
