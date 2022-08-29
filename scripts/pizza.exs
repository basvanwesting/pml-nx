defmodule Pizza do
  @precision      1.0e-4
  @lr             1.0e-3
  @max_iterations 10000

  def call do
    reservations = [13, 2,  14, 23, 13, 1,  18, 10, 26, 3,  3,  21, 7,  22, 2,  27, 6,  10, 18, 15, 9,  26, 8,  15, 10, 21, 5,  6,  13, 13]
    temperatures = [26, 14, 20, 25, 24, 12, 23, 18, 24, 14, 12, 27, 17, 21, 12, 26, 15, 21, 18, 26, 20, 25, 21, 22, 20, 21, 12, 14, 19, 20]
    tourists     = [9,  6,  3,  9,  8,  2,  9,  10, 3,  1,  3,  5,  3,  1,  4,  2,  4,  7,  3,  8,  6,  9,  10, 7,  2,  1,  7,  9,  4,  3]
    bias         = Enum.map(reservations, fn _ -> 1 end)
    pizzas       = [44, 23, 28, 60, 42, 5,  51, 44, 42, 9,  14, 43, 22, 34, 16, 46, 26, 33, 29, 43, 37, 62, 47, 38, 22, 29, 34, 38, 30, 28]

    x = Nx.tensor([bias, reservations, temperatures, tourists]) |> Nx.transpose()
    y = Nx.tensor([pizzas]) |> Nx.transpose()
    w = Nx.broadcast(0, {4, 1})

    train(x, y, w, @max_iterations, @lr)
    |> IO.inspect()
  end

  def train(_x, _y, w, 0, _lr), do: {w, 0}
  def train(x, y, w, iterations_left, lr) do
    current_loss = loss(x, y, w) |> Nx.to_number()
    IO.inspect({iterations_left, current_loss}, label: "iteration, loss")

    gradient = gradient(x, y, w)
    threshold = gradient
                |> Nx.abs()
                |> Nx.reduce_max()
                |> Nx.to_number()
                |> Kernel.*(lr)

    #IO.inspect({gradient, threshold}, label: "gradient, threshold")

    if threshold < @precision do
      {w, iterations_left}
    else
      new_w = Nx.subtract(w, Nx.multiply(gradient, lr))
      train(x, y, new_w, iterations_left - 1, lr)
    end
  end

  def predict(x, w) do
    Nx.dot(x, w)
  end

  def loss(x, y, w) do
    x
    |> Nx.dot(w)
    |> Nx.subtract(y)
    |> Nx.power(2)
    |> Nx.mean()
  end

  def gradient(x, y, w) do
    diff = x
    |> Nx.dot(w)
    |> Nx.subtract(y)

    x
    |> Nx.transpose()
    |> Nx.dot(diff)
    |> Nx.divide(Nx.axis_size(x, 0))
    |> Nx.multiply(2)
  end
end

Pizza.call()
