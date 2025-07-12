function fmeanv=fmean(x,fsamp,N)
NFFT=N;
x=x-mean(x);
[Pxx,f]=pwelch(x,boxcar(length(x)),0,NFFT,fsamp); %periodogramma semplice finestrato con finestra rettangolare lunghezza pari ad x
plot(f,Pxx/max(Pxx)); 
fmeanv=sum(f.*Pxx)/sum(Pxx);
end

