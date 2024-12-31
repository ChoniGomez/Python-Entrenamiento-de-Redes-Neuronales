from perceptron_multilayer import MultiLayerPerceptron

# mlp es una red de 3 capas, 2 entradas, una capa con 2 neuronas y la salida con una neurona
mlp = MultiLayerPerceptron(layers=[2, 2, 1])
mlp.set_weights([[[-10, -10], [15, 15]], [[10, 10]]])
mlp.set_bias([[15, -10], [-15]])
mlp.print_network()

#Red Neuronal Multicapa para la compuerta XOR con 2 entradas
print(f"0 0 = {mlp.run([0, 0])[0]:.5f}")
print(f"0 1 = {mlp.run([0, 1])[0]:.5f}")
print(f"1 0 = {mlp.run([1, 0])[0]:.5f}")
print(f"1 1 = {mlp.run([1, 1])[0]:.5f}")