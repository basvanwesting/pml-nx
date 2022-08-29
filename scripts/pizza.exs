defmodule Pizza do
  @precision 1.0e-7

  def call do
    x = [13, 2, 14, 23, 13, 1, 18, 10, 26, 3, 3, 21, 7, 22, 2, 27, 6, 10, 18, 15, 9, 26, 8, 15, 10, 21, 5, 6, 13, 13]
    y = [33, 16, 32, 51, 27, 16, 34, 17, 29, 15, 15, 32, 22, 37, 13, 44, 16, 21, 37, 30, 26, 34, 23, 39, 27, 37, 17, 18, 25, 23]

    tensor = Nx.tensor([x, y], names: [:reservations, :pizzas])
    model = {0.0, 0.0}
    max_iterations = 1000
    lr = 1.0

    train(tensor, max_iterations, model, lr)
    |> IO.inspect()
  end

  def train(_tensor, 0, model, _lr), do: {model, 0}
  def train(tensor, iterations_left, model, lr) do
    current_loss = loss(tensor, model)
    IO.inspect({iterations_left, model, current_loss}, label: "iteration, model, loss")

    {next_model, next_loss} =
      model
      |> neighbour_models(lr)
      |> Enum.map(fn neighbour_model -> {neighbour_model, loss(tensor, neighbour_model)} end)
      |> Enum.min_by(&(elem(&1,1)))

    cond do
      next_loss < current_loss -> train(tensor, iterations_left - 1, next_model, lr)
      iterations_left > 0 && lr > @precision -> train(tensor, iterations_left - 1, model, lr / 2.0)
      true -> {model, iterations_left}
    end
  end

  def loss(tensor, {a, b}) do
    tensor
    |> Nx.slice_along_axis(0, 1, axis: :reservations)
    |> Nx.multiply(a)
    |> Nx.add(b)
    |> Nx.subtract(Nx.slice_along_axis(tensor, 1, 1, axis: :reservations))
    |> Nx.power(2)
    |> Nx.mean()
    |> Nx.to_number()
  end

  def neighbour_models({a, b}, lr) do
    [
      {a + lr, b},
      {a - lr, b},
      {a, b + lr},
      {a, b - lr},
    ]
  end
end

Pizza.call()
