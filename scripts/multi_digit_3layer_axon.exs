defmodule MultiDigit3LayerAxon do

  @n_hidden_nodes 128
  @batch_size     256
  @epochs         10

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

    n_input_variables = Nx.axis_size(x_train, 1)
    n_classes = Nx.axis_size(y_train, 1)

    train_batches = Enum.zip(
      Nx.to_batched(x_train, @batch_size) |> Enum.to_list,
      Nx.to_batched(y_train, @batch_size) |> Enum.to_list
    )
    validate_batches = [{x_validate, y_validate}]
    test_batches     = [{x_test,     y_test}]

    model =
      Axon.input("input", shape: {nil, n_input_variables})
      |> Axon.dense(@n_hidden_nodes, activation: :sigmoid)
      |> Axon.dense(n_classes)
      |> Axon.activation(:softmax)

    Axon.Display.as_table(model, Nx.template({1, n_input_variables}, :f32)) |> IO.puts

    model_state =
      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, :adam, from_logits: true)
      |> Axon.Loop.metric(:accuracy)
      |> Axon.Loop.validate(model, validate_batches)
      |> Axon.Loop.run(train_batches, %{}, epochs: @epochs)

    IO.puts("=== TEST ===")

    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(:accuracy)
    |> Axon.Loop.run(test_batches, model_state)

    IO.puts("")
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
      |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))

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

end

MultiDigit3LayerAxon.call()

