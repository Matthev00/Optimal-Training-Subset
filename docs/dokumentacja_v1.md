# Wstępna propozycja rozwiązania

- Różne metody wybierania podzbioru:
  - Strategie ewolucyjne (μ + λ)
  - Strategie ewolucyjne (μ, λ)
  - Strategia ewolycjna (1 + 1)
  - Algorytm wspinaczkowy
  - Algorytm wspinaczkowy z tabu

- Algorytmy te będą wykorzystywane z biblioteki [DEAP](https://deap.readthedocs.io/en/master/).
- Wszystkie eksperymenty będą przeprowadzane na **FashionMNIST**, a na koniec zostanie wykonana ewaluacja dwóch najlepszych technik na **CIFAR-100**.

---

## Modele

Po przetestowaniu modeli jak radzą sobie z detekcją (jeszcze przed dotrenowywaniem) na podstawie wyników wybraliśmy: 

- **Prosty Model własny** dla FashionMNIST:
  - Model z bardzo prostą architekturą sieci konwolucyjnej
  - Został wybrany aby szybko testować metody wybierania podzbiorów na mniej złożonym zbiorze danych.
  - Po wstępnych testach otrzymujemy około 90% accuracy na FashionMNIST

- **[MobileNetV3-Small](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html#torchvision.models.mobilenet_v3_small)** dla CIFAR-100:
  - Model z domyślnymi wagami z PyTorch.
  - Modyfikacja pierwszej warstwy, aby obsługiwała obrazy czarno-białe (1 kanał).
  - Zmiana ostatniej warstwy dla liczby klas (10).
  - Po wstępnych testach otrzymujemy około 60% accuracy na CIFAR-100
---

## Analiza danych
 W projekcie będziemy korzystać ze zbiorów: 
 - FashionMNIST 
 - CIFAR-100

 Zbiory te są juz podzielone na zbiory treninowe i testowe. Podział ten zachowuje zrównoważenie klasowe. Skorzystamy z tego podziału oraz dodatkowo ze zbioru tetowego wydzielmy zbiór walidacyjny. 

---

## Funkcja celu

Funkcja celu:
$$
J(S) = \alpha \cdot \text{Balanced Accuracy}(S) - \beta \cdot \frac{|S|}{|D|}
$$


Początkowe wartości $\alpha$ i $\beta$ zostały dobrane jako neutralne (1), aby zrównoważyć znaczenie jakości klasyfikacji i rozmiaru podzbioru. W przypadku, gdy jeden ze składników będzie dominował, rozważymy dostrojenie tych parametrów za pomocą eksperymentów.

---

## Metryki

Dodatkowo zastosujemy standardowe metryki, aby zweryfikować, czy funkcja celu nie poszła w skrajność:
- **Balanced Accuracy**.
- **AUC-ROC**.

---

## Sposób podsumowywania wyników

- Każdy eksperyment należy przeprowadzić kilkukrotnie dla każdego algorytmu i wyciągnąć średnią.
- Porównywać metody wybierania podzbiorów, korzystając z krzywej **ECDF** (Empirical Cumulative Distribution Function).
- Raportować wyniki, korzystając z **MLflow**.
- Porównanie wyników z eksperymentami zawartymi w literaturze naukowej 

---
