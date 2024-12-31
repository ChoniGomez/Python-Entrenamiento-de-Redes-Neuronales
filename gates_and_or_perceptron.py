from perceptron_monolayer import Perceptron
from activation import unit_step

# -------------- FUNCION SIGMOIDAL ------------------

#Se crea una instancia de la clase Perceptron para la compuerta AND
perceptron = Perceptron(inputs = 2)
perceptron.set_weights([2, 2])
perceptron.set_bias(-3)

#Imprimir los valores de 2 entradas con 5 decimales despues de la coma
print("ADN con Funcion Sigmoidal")
print(f"0 0 = {perceptron.run([0, 0]):.5f}")
print(f"0 1 = {perceptron.run([0, 1]):.5f}")
print(f"1 0 = {perceptron.run([1, 0]):.5f}")
print(f"1 1 = {perceptron.run([1, 1]):.5f}")

#Se crea una instancia de la clase Perceptron para la compuerta OR
perceptron = Perceptron(inputs = 2)
perceptron.set_weights([3, 3])
perceptron.set_bias(-1)

#Imprimir los valores de 2 entradas con 5 decimales despues de la coma
print("OR con Funcion Sigmoidal")
print(f"0 0 = {perceptron.run([0, 0]):.5f}")
print(f"0 1 = {perceptron.run([0, 1]):.5f}")
print(f"1 0 = {perceptron.run([1, 0]):.5f}")
print(f"1 1 = {perceptron.run([1, 1]):.5f}")


# -------------- FUNCION ESCALON UNITARIO ------------------

#Se crea una instancia de la clase Perceptron para la compuerta AND
perceptron_unit_step = Perceptron(inputs = 2, activation_function = unit_step)
perceptron_unit_step.set_weights([2, 2])
perceptron_unit_step.set_bias(-3)

#Imprimir los valores de 2 entradas con 5 decimales despues de la coma
print("ADN con Funcion Escalon Unitario")
print(f"0 0 = {perceptron_unit_step.run([0, 0]):.5f}")
print(f"0 1 = {perceptron_unit_step.run([0, 1]):.5f}")
print(f"1 0 = {perceptron_unit_step.run([1, 0]):.5f}")
print(f"1 1 = {perceptron_unit_step.run([1, 1]):.5f}")

#Se crea una instancia de la clase Perceptron para la compuerta OR
perceptron_unit_step = Perceptron(inputs = 2, activation_function = unit_step)
perceptron_unit_step.set_weights([3, 3])
perceptron_unit_step.set_bias(-1)

#Imprimir los valores de 2 entradas con 5 decimales despues de la coma
print("OR con Funcion Escalon Unitario")
print(f"0 0 = {perceptron_unit_step.run([0, 0]):.5f}")
print(f"0 1 = {perceptron_unit_step.run([0, 1]):.5f}")
print(f"1 0 = {perceptron_unit_step.run([1, 0]):.5f}")
print(f"1 1 = {perceptron_unit_step.run([1, 1]):.5f}")