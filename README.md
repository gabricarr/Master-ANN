# Master-ANN


TODO g++:
- Use the dataset you found for news
- add 3 new features, one for each output of the finbert model (look at the paper) (positive, neutre, negative)
- relunch the model with the new finbert datapoints



- Preparare assieme powerpoint per metasti
cose da citare:
- Abbiamo difficoltà nel creare nuovi dati.
-- Questo perchè non viene detto quali indicatori/features vengono usati e gli indicatori sono proprietà intellettuale dell azienda dove hanno lavorato i ricercatori

- Dobbiamo ricostruirli da zero per il mercato americano

- Il piano è adattare l'architettura di Master con la sentiment analysis dell altro paper

- Per questo aggiungiamo uno stage di 'sentiment extraction' e concateniamo gli autput di finbert alle features estratte dagli indicatori finanziari.




TODO gcc:
- usare un dataset di CSI300 per confrontare l'implementazione delle nostre feature con quelle originali e vedere chi fa meglio e come.





Osservazioni su test set:
- Nel modello il training va dal 2008 a maggio 2020, il validation da maggio 2020 a settembre 2021 e il test set da settembre 2021 a dicembre 2022.

- Ora, se osserviamo l'sp500 abbiamo un anno, il 2022, dove le tendenze cambiano, ma nel 2020 e inizio 2021 le tendenze sono più o meno come quelle del training set.

- Se calcoliamo le metriche nel validation set, allora tutto sembra andare bene e otteniamo buoni risultati, se invece usiamo il test set ora abbiamo metriche pessime e il modello non funziona (se fosse così facile predirre stock nessuno di noi sarebbe quì....)











Idee:
- Selezionare le best 10 dopo il training set e vedere come il guadagno si confronta con 10 scelte a caso.























