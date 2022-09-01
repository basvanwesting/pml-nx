# Neural

Working through the book [Programming Machine Learning, by Paolo Perrotta](https://pragprog.com/titles/pplearn/programming-machine-learning/)

Implement using [Elixir Nx](https://github.com/elixir-nx/nx) with Google's XLA backend, instead of Python as in the book.

Chapters are coded in the scripts folder.
Usage example: `mix run scripts/multi_digit.exs`

Compiling XLA (CPU backend):

```
brew install bazelisk

export USE_BAZEL_VERSION=4.2.1
bazelisk

mix deps.clean xla --build && XLA_BUILD=true mix deps.compile
```


# Results

```
# batch_size = 256, epochs = 10, lr = 1.0, n_hidden_nodes = 128, normalized
mix run scripts/multi_digit_3layer_batched.exs

2-0 > Loss: 0.1612035632133484, Accuracy: 91.5199966430664%
4-0 > Loss: 0.078809455037117, Accuracy: 95.04000091552734%
6-0 > Loss: 0.047872643917798996, Accuracy: 96.08000183105469%
8-0 > Loss: 0.0325293093919754, Accuracy: 96.45999908447266%
10-0 > Loss: 0.023789983242750168, Accuracy: 96.55999755859375%

_-val > Loss: 0.03812151029706001, Accuracy: 96.62000274658203%
_-test > Loss: 0.03812151029706001, Accuracy: 98.43999481201172%
```

```
# batch_size = 256, epochs = 10, lr = 0.01, n_hidden_nodes = 200
mix run scripts/multi_digit_3layer_batched.exs

1-0 > Loss: 2.395386219024658, Accuracy: 12.980000495910645%
1-1 > Loss: 2.3256468772888184, Accuracy: 13.75%
1-2 > Loss: 2.2861595153808594, Accuracy: 14.519999504089355%
...
10-232 > Loss: 0.12508173286914825, Accuracy: 92.9000015258789%
10-233 > Loss: 0.24870315194129944, Accuracy: 92.83999633789062%
10-234 > Loss: 0.2239966243505478, Accuracy: 92.88999938964844%

_-_ > Loss: 0.249597430229187, Accuracy: 92.88999938964844%
```


```
mix run scripts/multi_digit_3layer.exs

iteration, loss: {10000, 2.5443921089172363}
iteration, loss: {9999, 2.474839925765991}
iteration, loss: {9998, 2.413820743560791}
...
iteration, loss: {3, 0.144504576921463}
iteration, loss: {2, 0.1444987654685974}
iteration, loss: {1, 0.14449293911457062}

Success: 9321/10000 (93.21%)
```

```
mix run scripts/multi_digit.exs

iteration, loss: {200, 6.931463718414307}
iteration, loss: {199, 8.434455871582031}
iteration, loss: {198, 5.512046813964844}
...
iteration, loss: {3, 0.8599666357040405}
iteration, loss: {2, 0.8595183491706848}
iteration, loss: {1, 0.8590735197067261}

Success: 9032/10000 (90.32%)
