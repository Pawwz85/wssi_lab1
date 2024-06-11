import numpy as np
from random import shuffle
from PIL import Image
from math import ceil, floor
from sklearn.datasets import load_iris

class IMGPainter:
    def __init__(self, size: tuple[int, int], background_color: tuple[int, int, int]):
        r, g, b = background_color
        x, y = size
        self.size = size
        self.img = np.array([[(r,g,b)]*y]*x).astype('int8')

    def draw_circle(self, pos: tuple[int, int], r: float, color: tuple[int, int, int], transparency: float = 0):
      x0, y0 = pos
      min_x = floor(x0 - r) if x0 > r else 0
      max_x = ceil(x0 + r) if x0 + r < self.size[0] else self.size[0]
      min_y = floor(y0 - r) if y0 > r else 0
      max_y = ceil(y0 + r) if y0 + r < self.size[1] else self.size[1]
      for x in range(min_x, max_x) :
         for y in range(min_y, max_y) :
          if (x - x0)**2 + (y - y0)**2 <= r**2:
            self.img[x][y] = np.array([*color])*(1 - transparency) +  self.img[x][y]*transparency

    def draw_line(self, pos1: tuple[int, int], pos2: tuple[int, int], color: tuple[int, int, int],  thickness: int, transparency: float = 0):
      x0, y0 = pos1
      x1, y1 = pos2
      if y0 == y1:
        A, B, C = (0, 1, -y0) # A * x + B * y + C = 0
      else:
        A = 1
        B = (x0 - x1)/(y1 - y0)
        C = -A*x0 - B*y0

      d = thickness/2
      min_x, max_x = ((min(x0, x1)), max(x0, x1))
      min_y, max_y = ((min(y0, y1)), max(y0, y1))
      min_x = floor(min_x - d) if min_x > d else 0
      max_x = ceil(max_x + d) if max_x + d < self.size[0] else self.size[0]
      min_y = floor(min_y - d) if min_y > d else 0
      max_y = ceil(max_y + d) if max_y + d < self.size[1] else self.size[1]
      for x in range(min_x, max_x) :
         for y in range(min_y, max_y) :
          if abs(x*A + y*B + C) < ((A**2 + B**2)**0.5) * thickness/2:
            self.img[x][y] = np.array([*color])*(1 - transparency) +  self.img[x][y]*transparency


class NNPainter:
  def __init__(self, nn = None, x_margin=None, y_margin=None,  neuron_radius = None, spacing = None, in_layer_spacing = None):
    self.__image_painter: None | IMGPainter = None
    self._x_margin = 50
    self._y_margin = 50
    self._neuron_radius = 50
    self._spacing_between_layers = 100
    self._spacing_in_layers = 30
    self._nn = nn
    self.set_params(x_margin = x_margin, y_margin=y_margin, neuron_radius= neuron_radius, spacing=spacing)

  def set_params(self, x_margin=None, y_margin=None, nn=None, neuron_radius = None, spacing = None, in_layer_spacing = None):
        if x_margin:
          self._x_margin = x_margin
        if y_margin:
          self._y_margin = y_margin
        if nn:
          self._nn = nn
        if neuron_radius:
          self._neuron_radius = neuron_radius
        if spacing:
          self._spacing_between_layers = spacing
        if in_layer_spacing:
          self._spacing_in_layers = in_layer_spacing
        return self

  def __create_neuron_layout(self, nn):
    architecture = nn.get_architecture()
    max_layer_size = max(architecture)
    result_list = []
    x = self._x_margin + self._neuron_radius
    column_size = max_layer_size * (2 * self._neuron_radius + self._spacing_in_layers) - self._spacing_in_layers
    for layer_size in architecture:
      offset = self._y_margin + (max_layer_size - layer_size)/max_layer_size*column_size/2
      y_step = self._spacing_in_layers + 2*self._neuron_radius
      ys = np.array([self._neuron_radius + i *y_step for i in range(layer_size)]) + offset
      result_list.append([(x, y) for y in ys])
      x += self._spacing_between_layers + 2*self._neuron_radius
    size = (x - self._spacing_between_layers - self._neuron_radius + self._x_margin, 2*self._y_margin + column_size)
    return result_list, size

  def __draw_neurons(self, img_painter: IMGPainter, locations):
      for i, layer in enumerate(locations):
        for pos in layer:
          color = (128, 128, 128) if i == 0 else (255, 165, 0) if i + 1 == len(locations) else (0, 191, 255)
          img_painter.draw_circle(pos, self._neuron_radius + 5, (0,0,0))
          img_painter.draw_circle(pos, self._neuron_radius, color)

  def __draw_connections(self, img_painter: IMGPainter, locations):
      for from_index in range(len(locations) - 1):
        for j, pos_from in enumerate(locations[from_index]):
          for i, pos_to in enumerate(locations[from_index + 1]):
            weight = self._nn._get_neuron(from_index, i).ws[j]
            origin = (pos_from[0] + self._neuron_radius + 5, pos_from[1])
            end = (pos_to[0] - self._neuron_radius - 5, pos_to[1])
            color = (256, 0, 0) if weight < 0 else (0,255,0)
            transparency = min(0.95, abs(weight))
            img_painter.draw_line(origin, end, color, 5, transparency)

  def draw(self):
      layout, size = self.__create_neuron_layout(self._nn)
      painter = IMGPainter(size, (255, 255, 255))
      self.__draw_neurons(painter, layout)
      self.__draw_connections(painter, layout)

      return painter.img

class GradientDescent:
  def __init__(self):
    self.gradient = None

  def optimise(self, raw_gradient):
    self.gradient = raw_gradient
    return self.gradient

class GradientDescentWithMomentum:
  def __init__(self, decay_factor = 0.9):
    self.gradient = None
    self.decay_factor = decay_factor

  def __update_gradient(self, raw_gradient, old_gradient):
    return [self.decay_factor * old_gradient[i] + raw_gradient[i] for i, _ in enumerate(raw_gradient)]
  def optimise(self, raw_gradient):
    if self.gradient: self.gradient = self.__update_gradient(raw_gradient[0], self.gradient[0]), self.__update_gradient(raw_gradient[1], self.gradient[1])
    else: self.gradient = raw_gradient
    return self.gradient

class Neuron:
    def __init__(self, n_inputs, bias = 0., weights = None):
        self.b = bias
        if weights: self.ws = np.array(weights)
        else: self.ws = (np.random.rand(n_inputs) - 0.5)*2
        self.gradient = np.zeros(np.shape(self.ws))
        self.bs_grad = 0.
        self.err = 0. # derivarife of overall network network error in respect with

    def _f(self, x): #activation function (here: leaky_relu)
        return max(x*.1, x)

    def _df(self, x): #dervitive of activation function
     if isinstance(x, np.ndarray):
        return np.array(list(map(lambda f: 1 if f>0 else 0.1, x)))
     else: return 1 if x>0 else 0.1

    def __call__(self, xs): #calculate the neuron's output: multiply the inputs with the weights and sum the values together, add the bias value,
                            # then transform the value via an activation function
        result = xs @ self.ws + self.b
        self.gradient = xs * self._df(result) # derivatife of neuron outpun in rescpect to weigths
        self.bs_grad = self._df(result) # ... to biases
        self.input_grad = self.ws * self._df(result) # ... to xs
        return self._f(result)


class NeuralNetwork:
  class SequentialBuilder:
    def __init__(self):
      self.__layers = []
      self.__optimiser = GradientDescent()
      self.__error_func = lambda y_true, y_pred: (y_pred- y_true)**2
      self.__derror_func = lambda y_true, y_pred: (y_pred- y_true)*2

    def add_layer(self, size: int, biases = None, weights = None):
      nr_of_layers = len(self.__layers)
      in_count = len(self.__layers[nr_of_layers - 1]) if nr_of_layers > 0 else 1

      if biases is None:
        biases = [0. for _ in range(size)]

      if weights is None:
        weights = [None for _ in range(size)]

      self.__layers.append([Neuron(in_count, biases[i], weights[i]) for i in range(size)])
      return self

    def set_optimiser(self, optimiser):
      self.__optimiser = optimiser
      return self

    def set_error_function(self, func):
      self.__error_func = func
      return self

    def build(self):
      return NeuralNetwork(self.__layers[1:], self.__optimiser, self.__error_func, self.__derror_func) # Drop input layer

  def __init__(self, layers: list[list[Neuron]], optimiser, error_function, derror_function):
    self.__nr_of_layers: int = len(layers)
    self.__layers: list[list[Neuron]] = layers
    self.__optimiser = optimiser
    self.__error_func = error_function
    self.__derror_func = derror_function
    self.__loss = 0.

  def __calculate_result_at_layer(self, layer_index: int, layer_input):
    result = [neuron(layer_input) for neuron in self.__layers[layer_index]]
    return np.array(result)

  def __propagate_forward(self, network_input):
    _in = network_input
    for i in range(len(self.__layers)):
      _in = self.__calculate_result_at_layer(i, _in)
    return _in

  def __calculate_gradient_at_layer(self, layer_index: int):
    # Someone could notice that we are not calculating the gradient of biases, but we can fully recover it later from weight gradient
    result = [neuron.input_grad * neuron.err for neuron in self.__layers[layer_index]] # TODO: here is a mistake, fix it immedietally
    return np.array(result)

  def __propagate_backward(self, error_value):
    _err = error_value # _err is  derivative of outputs on given layer in respect to error
    tensor_arr = [None] * len(self.__layers)
    for i in reversed(range(len(self.__layers))):
      for j, neuron in enumerate(self.__layers[i]): neuron.err = _err[j]
      tensor_arr[i] = self.__calculate_gradient_at_layer(i)
      if i > 0:
        _err = np.sum(tensor_arr[i], axis=0)

  def __retrieve_bias_gradient(self):
    return [np.array([neuron.bs_grad * neuron.err for neuron in layer]) for layer in self.__layers]

  def  __retriece_weights_gradient(self):
    return [np.array([neuron.gradient * neuron.err for neuron in layer]) for layer in self.__layers]

  def __calculate_gradients_for_single_item(self, x, y, derror_func):
    y_pred = self.predict(x)
    error = derror_func(y, y_pred)
    self.__propagate_backward(error)
    ws_grad = self.__retriece_weights_gradient()
    bias_grad = self.__retrieve_bias_gradient()
    self.__loss += self.__error_func(y, y_pred)
    return ws_grad, bias_grad

  def __calculate_gradients_for_batch(self, xs: list, ys: list, error_func):
    grads = [self.__calculate_gradients_for_single_item(x, ys[i], self.__derror_func) for i, x in enumerate(xs)]
    ws_grads = [item[0] for item in grads]
    bs_grads = [item[1] for item in grads]
    result_ws = ws_grads[0]
    result_bs = bs_grads[0]
    for g in ws_grads[1:]:
        for i in range(len(g)):
          result_ws[i] = result_ws[i] + g[i]
    for g in bs_grads[1:]:
        for i in range(len(g)):
          result_bs[i] = result_bs[i] + g[i]

    for i in range(len(result_ws)):
      result_ws[i] = result_ws[i]/len(xs)
      result_bs[i] = result_bs[i]/len(xs)
    self.__loss /= len(xs)
    return result_ws, result_bs

  def train_step(self, x: list, y: list, lr):
    self.__loss = 0.
    #grad = self.__optimiser.optimise(self.__calculate_gradient_for_batch(x,y, self.__error_func))
    ws_grad, bs_grad = self.__calculate_gradients_for_batch(x,y, self.__error_func)
    for layer_index, g in enumerate(ws_grad):
      for neuron_index, weight_gradient in enumerate(g):
        neuron = self._get_neuron(layer_index, neuron_index)
        neuron.ws -= lr*weight_gradient
    for layer_index, g in enumerate(bs_grad):
      for neuron_index, bias_gradient in enumerate(g):
        neuron = self._get_neuron(layer_index, neuron_index)
        neuron.b -= lr*bias_gradient
    print('loss:', self.__loss)

  def _get_neuron(self, layer_id, index):
    return self.__layers[layer_id][index]

  def predict(self, x):
    return self.__propagate_forward(x)

  def get_architecture(self):
    input_size = len(self.__layers[0][0].ws)
    return np.array([input_size] + [len(layer) for layer in self.__layers])


def shuffle_data(data):
  x, y = data
  indexes = [i for i in range(len(x))]
  shuffle(indexes)

  return np.array([x[i] for i in indexes]), np.array([y[i] for i in indexes])
def test():
  data = load_iris(return_X_y=True)
  data = shuffle_data(data)
  pivot = (len(data[0])//3)
  x, y = data

  validation_data = (x[0:pivot], y[0:pivot])
  training_data = (x[pivot:], y[pivot:])
  nn = NeuralNetwork.SequentialBuilder().add_layer(4).add_layer(4).add_layer(4).add_layer(3).set_optimiser(GradientDescentWithMomentum(0.9)).build()
  x_train, y_train = training_data
  y_train = np.array([[1 if label == i else 0 for i in range(3)] for label in y_train])

  correct_count = 0
  for i in range(50):
    nn.train_step(x_train, y_train, 0.01)
    correct_count = 0
    for j in range( len(validation_data)):
     #print(np.argmax(nn.predict(validation_data[0][j])), validation_data[1][j])
     if np.argmax(nn.predict(validation_data[0][j])) == validation_data[1][j]:
      correct_count += 1
    print('accuraccy: ', correct_count/len(validation_data) * 100, '%')

test()
