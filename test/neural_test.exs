defmodule NeuralTest do
  use ExUnit.Case
  doctest Neural

  test "greets the world" do
    assert Neural.hello() == :world
  end
end
