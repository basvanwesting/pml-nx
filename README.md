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
