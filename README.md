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
mix run scripts/multi_digit_2layer.exs

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
