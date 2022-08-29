defmodule Pizza do
  import Nx.Defn

  @precision 1.0e-7

  def call do
    x = [13, 2, 14, 23, 13, 1, 18, 10, 26, 3, 3, 21, 7, 22, 2, 27, 6, 10, 18, 15, 9, 26, 8, 15, 10, 21, 5, 6, 13, 13]
    y = [33, 16, 32, 51, 27, 16, 34, 17, 29, 15, 15, 32, 22, 37, 13, 44, 16, 21, 37, 30, 26, 34, 23, 39, 27, 37, 17, 18, 25, 23]

    tensor = Nx.tensor([x, y], names: [:reservations, :pizzas])
    start_weight = 0.0
    max_iterations = 1000
    delta = 1.0

    train(tensor, max_iterations, start_weight, delta)
    |> IO.inspect()
  end

  def train(_tensor, 0, weight, _delta), do: {weight, 0}
  def train(tensor, iterations_left, weight, delta) do
    current_loss = loss(tensor, weight) |> Nx.to_number #|> IO.inspect(label: "current_loss")
    delta_up_loss = loss(tensor, weight + delta) |> Nx.to_number #|> IO.inspect(label: "delta_up_loss")
    delta_down_loss = loss(tensor, weight - delta) |> Nx.to_number #|> IO.inspect(label: "delta_down_loss")

    cond do
      delta_up_loss < current_loss -> train(tensor, iterations_left - 1, weight + delta, delta)
      delta_down_loss < current_loss -> train(tensor, iterations_left - 1, weight - delta, delta)
      iterations_left > 0 && delta > @precision -> train(tensor, iterations_left - 1, weight, delta / 2.0)
      true -> {weight, iterations_left}
    end
  end

  defn loss(tensor, weight) do
    tensor
    |> Nx.slice_along_axis(0, 1, axis: :reservations)
    |> Nx.multiply(weight)
    |> Nx.subtract(Nx.slice_along_axis(tensor, 1, 1, axis: :reservations))
    |> Nx.power(2)
    |> Nx.mean()
  end
end

Pizza.call()
