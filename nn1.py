import numpy as np
from PIL import Image
from math import ceil, floor
class IMGPainter:
    def __init__(self, size: tuple[int, int], background_color: tuple[int, int, int]):
        r, g, b = background_color
        x, y = size
        self.size = size
        self.img = np.array([[(r,g,b)]*y]*x, dtype='int8')

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

class Neuron:
    def __init__(self, n_inputs, bias = 0., weights = None):
        self.b = bias
        if weights: self.ws = np.array(weights)
        else: self.ws = (np.random.rand(n_inputs) - 0.5)*2

    def _f(self, x): #activation function (here: leaky_relu)
        return max(x*.1, x)

    def __call__(self, xs): #calculate the neuron's output: multiply the inputs with the weights and sum the values together, add the bias value,
                            # then transform the value via an activation function
        return self._f(xs @ self.ws + self.b)


class NeuralNetwork:
  class SequentialBuilder:
    def __init__(self):
      self.__layers = []

    def add_layer(self, size: int, biases = None, weights = None):
      nr_of_layers = len(self.__layers)
      in_count = len(self.__layers[nr_of_layers - 1]) if nr_of_layers > 0 else 1

      if biases is None:
        biases = [0. for _ in range(size)]

      if weights is None:
        weights = [None for _ in range(size)]

      self.__layers.append([Neuron(in_count, biases[i], weights[i]) for i in range(size)])
      return self

    def build(self):
      return NeuralNetwork(self.__layers[1:]) # Drop input layer

  def __init__(self, layers: list[list[Neuron]]):
    self.__nr_of_layers: int = len(layers)
    self.__layers: list[list[Neuron]] = layers

  def __calculate_result_at_layer(self, layer_index: int, layer_input):
    result = [neuron(layer_input) for neuron in self.__layers[layer_index]]
    return np.array(result)

  def __propagate_forward(self, network_input):
    _in = network_input
    for i in range(len(self.__layers)):
      _in = self.__calculate_result_at_layer(i, _in)
    return _in

  def _get_neuron(self, layer_id, index):
    return self.__layers[layer_id][index]

  def predict(self, x):
    return self.__propagate_forward(x)

  def get_architecture(self):
    input_size = len(self.__layers[0][0].ws)
    return np.array([input_size] + [len(layer) for layer in self.__layers])


nn = NeuralNetwork.SequentialBuilder().add_layer(3).add_layer(4).add_layer(4).add_layer(1).build()
painter = NNPainter(nn)
painter.set_params(spacing=200)
#print(nn.predict(np.array([0.1, 1.0, 2.5])))

print(nn.get_architecture())

t = painter.draw()
img = Image.fromarray(t, mode='RGB')
img.rotate(90, expand = 1)
