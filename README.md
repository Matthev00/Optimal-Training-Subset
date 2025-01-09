# Optimal Training Subset

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Opis 

Celem projektu jest znalezienie problemu klasyfikacji obrazów (np. zbiór FashionMNIST lub docelowo CIFAR-10 lub CIFAR-100 ) podzbioru najbardziej charakterystycznych przykładów z każdej klasy które są wystarczającym do zbudowania poprawnie działającego klasyfikatora. Czyli które zdjęcia są w stanie zapewnić najlepszą możliwą separację. 

## Algorytmy
Różne metody wybierania podzbioru:

- Strategia ewolucyjne $(\mu + \lambda)$ - krzyżowanie, mutacja, selekcja
- Strategia ewolucyjne $(\mu, \lambda)$ - krzyżowanie, mutacja, selekcja
- Strategia ewolucyjna $(1 + 1)$ - mutacja
- Algorytm wspinaczkowy
- Algorytm wspinaczkowy z tabu (z kolejką FIFO)

## Ocena algorytmów

### Funckja celu

W fukncji celu chcemy znaleść balans między dokładnością detekcji a wielkością zbioru

$$
J(S) = \alpha \cdot \text{Balanced Accuracy}(S) - \beta \cdot \frac{|S|}{|D|}
$$

Wartości $alphaα = 1$ i $beta = 1$.

### Ostateczne metryki

Ostatecznie zastosujemy **balanced accuracy** aby aby zweryfikować, czy funkcja celu nie poszła w skrajność


--------

