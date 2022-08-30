defmodule Police do
  import Nx.Defn

  @precision      1.0e-4
  @lr             1.0e-3
  @max_iterations 10000

  def call do
    reservations = [13, 2,  14, 23, 13, 1,  18, 10, 26, 3,  3,  21, 7,  22, 2,  27, 6,  10, 18, 15, 9,  26, 8,  15, 10, 21, 5,  6,  13, 13]
    temperatures = [26, 14, 20, 25, 24, 12, 23, 18, 24, 14, 12, 27, 17, 21, 12, 26, 15, 21, 18, 26, 20, 25, 21, 22, 20, 21, 12, 14, 19, 20]
    tourists     = [9,  6,  3,  9,  8,  2,  9,  10, 3,  1,  3,  5,  3,  1,  4,  2,  4,  7,  3,  8,  6,  9,  10, 7,  2,  1,  7,  9,  4,  3]
    polices      = [1,  0,  1,  1,  1,  0,  1,  1,  1,  0,  0,  1,  0,  1,  0,  1,  0,  0,  0,  1,  0,  1,  0,  1,  0,  1,  0,  0,  1,  0]

    x = [reservations, temperatures, tourists]
        |> Nx.tensor()
        |> Nx.transpose()

    bias = Nx.broadcast(1, {Nx.axis_size(x, 0), 1})
    x = Nx.concatenate([bias, x], axis: 1)

    y = Nx.tensor([polices]) |> Nx.transpose()
    w = Nx.broadcast(0, {4, 1})

    {w, _} = train(x, y, w, @max_iterations, @lr)
    |> IO.inspect()

    test(x, y, w)
  end

  def test(x, y, w) do
    total_examples = Nx.axis_size(x, 0)
    correct_results = classify(x, w)
                     |> Nx.equal(y)
                     |> Nx.sum()
                     |> Nx.to_number()

    success_percent = correct_results * 100 / total_examples
    IO.puts("Success: #{correct_results}/#{total_examples} (#{success_percent}%)")
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

  defn forward(x, w) do
    x
    |> Nx.dot(w)
    |> Nx.sigmoid()
  end

  defn classify(x, w) do
    x
    |> forward(w)
    |> Nx.round()
  end

  defn loss(x, y, w) do
    y_hat = forward(x, w)

    first_term = y * Nx.log(y_hat)
    second_term = Nx.subtract(1, y) * Nx.log(Nx.subtract(1, y_hat))

    Nx.negate(Nx.mean(Nx.add(first_term, second_term)))
  end

  defn gradient(x, y, w) do
    diff = x
    |> forward(w)
    |> Nx.subtract(y)

    x
    |> Nx.transpose()
    |> Nx.dot(diff)
    |> Nx.divide(Nx.axis_size(x, 0))
  end
end

Police.call()

