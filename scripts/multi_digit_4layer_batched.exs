defmodule MultiDigit4LayerBatched do
  import Nx.Defn

  @lr              1.0
  @n_hidden_nodes0 256
  @n_hidden_nodes1 64
  @batch_size      256
  @epochs          10

  def call do
    #IO.inspect(Nx.default_backend(), label: "Nx backend")

    {x_train,    y_train}    = load_training_data()
    {x_test_all, y_test_all} = load_testing_data()
    {x_train, x_test_all} = standardize_data(x_train, x_test_all)

    test_batch_size = Nx.axis_size(x_test_all, 0) |> div(2)
    [x_validate, x_test | _rest] = Nx.to_batched(x_test_all, test_batch_size) |> Enum.to_list()
    [y_validate, y_test | _rest] = Nx.to_batched(y_test_all, test_batch_size) |> Enum.to_list()

    #IO.inspect(x_train)
    #IO.inspect(y_train)
    #IO.inspect(x_validate)
    #IO.inspect(y_validate)
    #IO.inspect(x_test)
    #IO.inspect(y_test)

    weights = train(x_train, y_train, x_validate, y_validate, @n_hidden_nodes0, @n_hidden_nodes1, @epochs, @batch_size, @lr)

    report("_", "val", x_train, y_train, x_validate, y_validate, weights)
    report("_", "test", x_train, y_train, x_test, y_test, weights)
  end

  def load_training_data() do
    {train_images, train_labels} = Scidata.MNIST.download()

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

  def standardize_data(x_train, x_test) do
    average = Nx.mean(x_train)
    standard_deviation = Nx.standard_deviation(x_train)
    {
      Nx.subtract(x_train, average) |> Nx.divide(standard_deviation),
      Nx.subtract(x_test, average)  |> Nx.divide(standard_deviation)
    }
  end

  def train(x_train, y_train, x_test, y_test, n_hidden_nodes0, n_hidden_nodes1, epochs, batch_size, lr) do
    n_input_variables = Nx.axis_size(x_train, 1)
    n_classes = Nx.axis_size(y_train, 1)
    weights = initialize_weights(n_input_variables, n_hidden_nodes0, n_hidden_nodes1, n_classes)

    x_y_train_batches = Enum.zip(
      Nx.to_batched(x_train, batch_size),
      Nx.to_batched(y_train, batch_size)
    )

    for epoch <- 1..epochs,
      {{x_train_batch, y_train_batch}, batch_index} <- Enum.with_index(x_y_train_batches),
      reduce: weights do
      weights ->
        gradient = back(weights, x_train_batch, y_train_batch)
        weights = apply_gradient_to_weights(weights, gradient, lr)

        if rem(epoch, 2) == 0 && batch_index == 0 do
          report(epoch, batch_index, x_train_batch, y_train_batch, x_test, y_test, weights)
        end

        weights
    end
  end

  def initialize_weights(n_input_variables, n_hidden_nodes0, n_hidden_nodes1, n_classes) do
    w1_rows = n_input_variables + 1
    w2_rows = n_hidden_nodes0 + 1
    w3_rows = n_hidden_nodes1 + 1
    w1 = Nx.random_normal({w1_rows, n_hidden_nodes0}) |> Nx.multiply(:math.sqrt(1 / w1_rows))
    w2 = Nx.random_normal({w2_rows, n_hidden_nodes1}) |> Nx.multiply(:math.sqrt(1 / w2_rows))
    w3 = Nx.random_normal({w3_rows, n_classes})       |> Nx.multiply(:math.sqrt(1 / w3_rows))

    {w1, w2, w3}
  end

  def report(epoch, batch_index, x_train, y_train, x_test, y_test, weights) do
    current_loss = loss(weights, x_train, y_train) |> Nx.to_number()
    accuracy = classify(x_test, weights)
               |> Nx.equal(y_test)
               |> Nx.mean()
               |> Nx.multiply(100)
               |> Nx.to_number()

    IO.puts("#{epoch}-#{batch_index} > Loss: #{current_loss}, Accuracy: #{accuracy}%")
  end

  defn forward(x, {w1, w2, w3}) do
    x
    |> prepend_bias()
    |> Nx.dot(w1)
    |> Nx.sigmoid()
    |> prepend_bias()
    |> Nx.dot(w2)
    |> Nx.sigmoid()
    |> prepend_bias()
    |> Nx.dot(w3)
    |> softmax()
  end

  defn classify(x, weights) do
    x
    |> forward(weights)
    |> Nx.argmax(axis: 1)
    |> Nx.new_axis(-1)
  end

  defn loss(weights, x, y) do
    y_hat = forward(x, weights)

    Nx.multiply(y, Nx.log(y_hat))
    |> Nx.sum()
    |> Nx.divide(Nx.axis_size(y, 0))
    |> Nx.negate()
  end

  defn back(weights, x, y) do
    grad(weights, &loss(&1, x, y))
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

  defn apply_gradient_to_weights({w1, w2, w3}, {g1, g2, g3}, lr) do
    w1 = Nx.subtract(w1, Nx.multiply(g1, lr))
    w2 = Nx.subtract(w2, Nx.multiply(g2, lr))
    w3 = Nx.subtract(w3, Nx.multiply(g3, lr))
    {w1, w2, w3}
  end

end

MultiDigit4LayerBatched.call()

