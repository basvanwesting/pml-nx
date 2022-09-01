defmodule MultiDigit3Layer do
  import Nx.Defn

  @lr             1.0e-2
  @max_iterations 10000
  @h_nodes        200

  def call do
    IO.inspect(Nx.default_backend(), label: "Nx backend")

    {x_train, y_train} = load_training_data()
    {x_test,  y_test}  = load_testing_data()

    IO.inspect(x_train)
    IO.inspect(y_train)
    IO.inspect(x_test)
    IO.inspect(y_test)

    n_input_variables = Nx.axis_size(x_train, 1)
    n_hidden_nodes = @h_nodes
    n_classes = Nx.axis_size(y_train, 1)

    {w1, w2} = initialize_weights(n_input_variables, n_hidden_nodes, n_classes)

    IO.inspect(w1)
    IO.inspect(w2)

    {w1, w2, _} = train(x_train, y_train, w1, w2, @max_iterations, @lr)

    IO.inspect(w1)
    IO.inspect(w2)

    test(x_test, y_test, w1, w2)
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
      |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))

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

    #IO.inspect(x_test)
    #IO.inspect(y_test)

    {x_test, y_test}
  end

  def initialize_weights(n_input_variables, n_hidden_nodes, n_classes) do
    w1_rows = n_input_variables + 1
    w2_rows = n_hidden_nodes + 1
    w1 = Nx.random_normal({w1_rows, n_hidden_nodes}) |> Nx.multiply(:math.sqrt(1 / w1_rows))
    w2 = Nx.random_normal({w2_rows, n_classes}) |> Nx.multiply(:math.sqrt(1 / w2_rows))

    {w1, w2}
  end

  def test(x, y, w1, w2) do
    total_examples = Nx.axis_size(x, 0)
    correct_results = classify(x, w1, w2)
                      |> Nx.equal(y)
                      |> Nx.sum()
                      |> Nx.to_number()

    success_percent = correct_results * 100 / total_examples
    IO.puts("Success: #{correct_results}/#{total_examples} (#{success_percent}%)")
  end

  def train(_x, _y, w1, w2, 0, _lr), do: {w1, w2, 0}
  def train(x, y, w1, w2, iterations_left, lr) do
    current_loss = loss(x, y, w1, w2) |> Nx.to_number()
    IO.inspect({iterations_left, current_loss}, label: "iteration, loss")

    {w1_gradient, w2_gradient} = back(x, y, w1, w2)
    #IO.inspect(w1_gradient, label: "w1_gradient")
    #IO.inspect(w2_gradient, label: "w2_gradient")

    new_w1 = Nx.subtract(w1, Nx.multiply(w1_gradient, lr))
    new_w2 = Nx.subtract(w2, Nx.multiply(w2_gradient, lr))
    train(x, y, new_w1, new_w2, iterations_left - 1, lr)
  end

  defn forward(x, w1, w2) do
    x
    |> prepend_bias()
    |> Nx.dot(w1)
    |> Nx.sigmoid()
    |> prepend_bias()
    |> Nx.dot(w2)
    |> softmax()
  end

  defn classify(x, w1, w2) do
    x
    |> forward(w1, w2)
    |> Nx.argmax(axis: 1)
    |> Nx.new_axis(-1)
  end

  defn loss(x, y, w1, w2) do
    y_hat = forward(x, w1, w2)

    Nx.multiply(y, Nx.log(y_hat))
    |> Nx.sum()
    |> Nx.divide(Nx.axis_size(y, 0))
    |> Nx.negate()
  end

  defn back(x, y, w1, w2) do
    h = x
        |> prepend_bias()
        |> Nx.dot(w1)
        |> Nx.sigmoid()

    y_hat = h
            |> prepend_bias()
            |> Nx.dot(w2)
            |> softmax()

    w2_gradient = Nx.dot(
      prepend_bias(h) |> Nx.transpose(),
      Nx.subtract(y_hat, y)
    ) |> Nx.divide(Nx.axis_size(x, 0))

    w1_gradient = Nx.dot(
      prepend_bias(x) |> Nx.transpose(),
      Nx.multiply(
        Nx.dot(
          Nx.subtract(y_hat, y),
          Nx.slice_along_axis(w2, 1, Nx.axis_size(w2, 0) - 1, axis: 0) |> Nx.transpose()
        ),
        sigmoid_gradient(h)
      )
    ) |> Nx.divide(Nx.axis_size(x, 0))

    {w1_gradient, w2_gradient}
  end

  defn sigmoid_gradient(sigmoid) do
    Nx.multiply(sigmoid, Nx.subtract(1, sigmoid))
  end

  defn softmax(t) do
    n = t
        |> Nx.exp()
        |> Nx.sum(axes: [1])
        |> Nx.new_axis(-1)

    t
    |> Nx.exp()
    |> Nx.divide(n)
  end

  defn prepend_bias(t) do
    bias = Nx.broadcast(1, {Nx.axis_size(t, 0), 1})
    Nx.concatenate([bias, t], axis: 1)
  end

end

MultiDigit3Layer.call()

