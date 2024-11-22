# Wstępna propozycja rozwiązania

- Podzielenie zbioru danych na **train/val/test** w proporcji **70:20:10**.
- Różne metody wybierania podzbioru typu:
  - Algorytmy ewolucyjne.
  - Strategie ewolucyjne.
  - Algorytm roju cząstek.
  - Może jakieś symulowane wyżażanie???????

- Algorytmy te będą wykorzystywane z biblioteki [DEAP](https://deap.readthedocs.io/en/master/).
- Wszystkie eksperymenty będą przeprowadzane na **FashionMNIST**, a na koniec zostanie wykonana ewaluacja dwóch najlepszych technik na **CIFAR-100**.

---

## Modele

- **[MobileNetV3-Small](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html#torchvision.models.mobilenet_v3_small)** dla FashionMNIST:
  - Model z domyślnymi wagami z PyTorch.
  - Modyfikacja pierwszej warstwy, aby obsługiwała obrazy czarno-białe (1 kanał).
  - Zmiana ostatniej warstwy dla liczby klas (10).

- **[EfficientNet-B0](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html#torchvision.models.efficientnet_b0)** dla CIFAR-100:
  - Model z domyślnymi wagami z PyTorch.
  - Zmiana ostatniej warstwy dla liczby klas (100).

---

## Analiza danych

TODO
---

## Funkcja celu

Funkcja celu:
$$
J(S) = \alpha \cdot \text{Balanced Accuracy}(S) - \beta \cdot \frac{|S|}{|D|}
$$

**Pytanie:** Jak dobrać wartości $\alpha$ i $\beta$?

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

---
