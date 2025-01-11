# Dokumentacja Końcowa

## Opis problemu
Dla problemu klasyfikacji obrazów (np. zbiór FashionMNIST lub docelowo CIFAR-10 lub CIFAR-100 ), należy znaleźć podzbiór najbardziej charakterystycznych przykładów z każdej klasy które są wystarczającym do zbudowania poprawnie działającego klasyfikatora. Czyli które zdjęcia są w stanie zapewnić najlepszą możliwą separację?

## Opis Rozwiązania

### Dane

### Funkcja celu 
$$
J(S) = \alpha \cdot \text{Balanced Accuracy}(S) - \beta \cdot \frac{|S|}{|D|}
$$
$\alpha = 1.0$  
$\beta = 0.5$

### Zastosowane modele CV

Po przetestowaniu modeli jak radzą sobie z detekcją(2-3 epoki na calym zbiorze treningowym) na podstawie wyników wybraliśmy: 

- **Prosty Model własny** dla FashionMNIST:
    ```py
    class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    ```

- **Trój kanałowy prosty model własny** dla Cifar-10
    ```python
    class CNN3Channel(nn.Module):
        def __init__(self):
            super(CNN3Channel, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64 * 4 * 4, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    ```

Decyzja o wyborze najprostszych modeli jest uzasadniona ich szybkością w trenowaniu oraz walidacji. Modele te osiągają wyniki na poziomie 85% na FashionMNIST i 50% na CIFAR-10 już po 2-3 epokach.

### Zastosowane algorytmy optymalizacyjne


### Przeprowadzone eksperymenty


### Zmierzone metryki
  - **Balanced Accuracy**.  
  - **Confusion Matrix**

### Analiza wyników FashionMNIST

### Analiza wyników Cifar-10

## Podsumowanie

## Zespół
Mateusz Ostaszewski 325203  
Michał Sadowski 325221  

