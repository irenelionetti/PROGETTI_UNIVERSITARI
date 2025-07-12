%shift del segnale in frequenza, permette di shiftare un segnale di una
%quantità qualunque che non deve essere per forza un campione discreto,
%altri metodi fatti a mano fanno traslare un segnale di una quantità pari a
%1 campione alla volta. 
function segt=freshift(seg,teta)

SEG=fft(seg);

f=fftshift([-0.5:1/(length(seg)):0.5-1/(length(seg))]);

SEGt=SEG.*exp(j*2*pi*teta*f);
segt=real(ifft(SEGt));

