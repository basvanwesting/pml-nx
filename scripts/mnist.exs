defmodule MNIST do

  @input_shape  {nil, 1, 28, 28}
  @output_shape {nil, 10}

  @batch_size 32
  @epochs     20

  def call do
    #IO.inspect(Nx.default_backend(), label: "Nx backend")

    train_batches = Scidata.MNIST.download()      |> prepare_scidata_streams(@batch_size)
    test_batches  = Scidata.MNIST.download_test() |> prepare_scidata_streams(@batch_size)

    #{validate_batches, final_batches} = test_batches |> Enum.split(5000)

    model =
      Axon.input("input", shape: @input_shape)
      |> Axon.conv(16, kernel_size: {3, 3}, activation:  :relu)
      |> Axon.batch_norm()
      |> Axon.dropout()
      |> Axon.conv(32, kernel_size: {3, 3}, activation:  :relu)
      |> Axon.batch_norm()
      |> Axon.dropout()
      |> Axon.flatten()
      |> Axon.dense(512, activation: :relu)
      |> Axon.batch_norm()
      |> Axon.dropout()
      |> Axon.dense(128, activation: :relu)
      |> Axon.batch_norm()
      |> Axon.dropout()
      |> Axon.dense(10, activation: :softmax)

    #model =
      #Axon.input("input", shape: @input_shape)
      #|> Axon.conv(32, kernel_size: {3, 3}, activation:  :relu)
      #|> Axon.max_pool(kernel_size: {2, 2})
      #|> Axon.conv(64, kernel_size: {3, 3}, activation:  :relu)
      #|> Axon.max_pool(kernel_size: {2, 2})
      #|> Axon.flatten()
      #|> Axon.dense(64, activation:  :relu)
      #|> Axon.dense(10, activation:  :softmax)

    Axon.Display.as_table(model, Nx.template({1, 1, 28, 28}, :f32)) |> IO.puts

    model_state =
      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, :adam, from_logits: true)
      |> Axon.Loop.metric(:accuracy)
      |> Axon.Loop.validate(model, test_batches)
      |> Axon.Loop.run(train_batches, %{}, epochs: @epochs)

    #IO.puts("=== FINAL TEST ===")

    #model
    #|> Axon.Loop.evaluator()
    #|> Axon.Loop.metric(:accuracy)
    #|> Axon.Loop.run(final_batches, model_state)

    #IO.puts("")

    #{x1, y1} = hd(test_batches)
    #IO.inspect(y1)

    #Axon.predict(model, model_state, x1)
    #|> Nx.argmax(axis: 1)
    #|> Nx.new_axis(-1)
    #|> IO.inspect()
  end

  def prepare_scidata_streams(raw_data, batch_size) do
    {train_images, train_labels} = raw_data

    # Normalize and batch images
    {images_binary, images_type, images_shape} = train_images

    batched_images =
      images_binary
      |> Nx.from_binary(images_type)
      |> Nx.reshape(images_shape)
      |> IO.inspect(label: "reshaped binary images")
      |> Nx.divide(255)
      |> Nx.to_batched(batch_size)

    # One-hot-encode and batch labels
    {labels_binary, labels_type, _labels_shape} = train_labels

    batched_labels =
      labels_binary
      |> Nx.from_binary(labels_type)
      |> Nx.new_axis(-1)
      |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
      |> Nx.to_batched(batch_size)

    Stream.zip(batched_images, batched_labels)
  end

end

MNIST.call()

