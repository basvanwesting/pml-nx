defmodule SingleDigit do
  import Nx.Defn

  @lr             1.0e-5
  @max_iterations 100

  def call do
    IO.inspect(Nx.default_backend(), label: "Nx backend")

    {x_train, y_train} = load_training_data()
    {x_test,  y_test}  = load_testing_data()

    IO.inspect(x_train)
    IO.inspect(y_train)
    IO.inspect(x_test)
    IO.inspect(y_test)

    w = Nx.broadcast(0, {Nx.axis_size(x_train, 1), 1})

    IO.inspect(w)

    {w, _} = train(x_train, y_train, w, @max_iterations, @lr)

    IO.inspect(w)

    test(x_test, y_test, w)
  end

  def load_training_data do
    {train_images, train_labels} = Scidata.MNIST.download()
    #{test_images, test_labels} = Scidata.MNIST.download_test()

    {images_binary, images_type, images_shape} = train_images
    {labels_binary, labels_type, _labels_shape} = train_labels

    flat_shape = {elem(images_shape, 0), elem(images_shape, 2) * elem(images_shape, 3)}

    x_train =
      images_binary
      |> Nx.from_binary(images_type)
      |> Nx.reshape(flat_shape)

    y_train =
      labels_binary
      |> Nx.from_binary(labels_type)
      |> Nx.new_axis(-1)
      |> Nx.equal(5)

    bias = Nx.broadcast(1, {Nx.axis_size(x_train, 0), 1})
    x_train = Nx.concatenate([bias, x_train], axis: 1)

    #IO.inspect(x_train)
    #IO.inspect(y_train)

    {x_train, y_train}
  end

  def load_testing_data do
    {test_images, test_labels} = Scidata.MNIST.download_test()

    {images_binary, images_type, images_shape} = test_images
    {labels_binary, labels_type, _labels_shape} = test_labels

    flat_shape = {elem(images_shape, 0), elem(images_shape, 2) * elem(images_shape, 3)}

    x_test =
      images_binary
      |> Nx.from_binary(images_type)
      |> Nx.reshape(flat_shape)

    y_test =
      labels_binary
      |> Nx.from_binary(labels_type)
      |> Nx.new_axis(-1)
      |> Nx.equal(5)

    bias = Nx.broadcast(1, {Nx.axis_size(x_test, 0), 1})
    x_test = Nx.concatenate([bias, x_test], axis: 1)

    #IO.inspect(x_test)
    #IO.inspect(y_test)

    {x_test, y_test}
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
    #IO.inspect({gradient, threshold}, label: "gradient, threshold")

    new_w = Nx.subtract(w, Nx.multiply(gradient, lr))
    train(x, y, new_w, iterations_left - 1, lr)
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

SingleDigit.call()

