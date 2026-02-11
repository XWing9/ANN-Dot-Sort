import sys
import numpy as math
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtGui import QPainter, QColor, QPen, Qt
from PySide6.QtCore import QTimer

#Settings
num_dots = 50
learning_rate = 0.001
target_red = math.array([-2.0, 0.0])
target_blue = math.array([2.0, 0.0])

width, height = 800, 700
margin = 50
top_height = 350
bottom_height = height - top_height - 2 * margin

graphYScale = 3.0


#Dots
dots_pos = math.random.uniform(-4, 4, size=(num_dots, 2))
dots_color = math.random.choice([0, 1], size=num_dots) #0 red 1 blue
loss_history = []
frame_count = 0

#ANN
input_size = 3  #number of neurons
hidden_size = 8
output_size = 2

weight1 = math.random.randn(input_size, hidden_size) * 0.5 #weights between 1-2 layer
bias1 = math.zeros(hidden_size) #hidden layer bias
weight2 = math.random.randn(hidden_size, output_size) * 0.5 #weights between 2-3 layer
bias2 = math.zeros(output_size) #output layer bias

def forward(dotInfo):
    global weight1, bias1, weight2, bias2
    rawSum1 = math.dot(dotInfo, weight1) + bias1 #matrizes stuff
    activation1 = math.tanh(rawSum1) #aktivation funktion
    output = math.dot(activation1, weight2) + bias2
    return output, activation1

def compute_loss(prediction, dotTeam):
    loss = 0.0
    for i in range(len(dotTeam)):
        target = target_red if dotTeam[i] == 0 else target_blue
        loss += math.sum((prediction[i] - target) ** 2)
    return loss / len(dotTeam)

#ANN learning part
def backward(dotInfo, activation1, prediction, dotTeam):
    global weight1, bias1, weight2, bias2
    deltaWeight2 = math.zeros_like(weight2)
    deltaBias2 = math.zeros_like(bias2)
    deltaWeight1 = math.zeros_like(weight1)
    deltaBias1 = math.zeros_like(bias1)
    for i in range(len(dotTeam)):
        target = target_red if dotTeam[i] == 0 else target_blue
        error = 2 * (prediction[i] - target)
        deltaWeight2 += math.outer(activation1[i], error)
        deltaBias2 += error
        dA1 = math.dot(weight2, error)
        dZ1 = dA1 * (1 - activation1[i] ** 2)
        deltaWeight1 += math.outer(dotInfo[i], dZ1)
        deltaBias1 += dZ1
    deltaWeight2 /= len(dotTeam) # calc gradient values durch dotTeam values?
    deltaBias2 /= len(dotTeam)
    deltaWeight1 /= len(dotTeam)
    deltaBias1 /= len(dotTeam)
    weight2 -= learning_rate * deltaWeight2 #optimize the weights and values with new values
    bias2 -= learning_rate * deltaBias2
    weight1 -= learning_rate * deltaWeight1
    bias1 -= learning_rate * deltaBias1

# GUI CLASS
class ANNWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Moving Dots ANN")
        self.setGeometry(100, 100, width, height)

        # Timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)

    def update_animation(self):
        global dots_pos, loss_history, frame_count
        frame_count += 1

        # Prepare input
        dotInfo = math.column_stack([dots_pos, dots_color])
        prediction, activation1 = forward(dotInfo)
        loss = compute_loss(prediction, dots_color)
        loss_history.append(loss)
        backward(dotInfo, activation1, prediction, dots_color)

        # Move dots
        dots_pos += (prediction - dots_pos) * 0.05

        if len(loss_history) > width - 2*margin:
            loss_history = loss_history[-(width - 2*margin):]

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background:
        painter.fillRect(self.rect(), QColor(26, 26, 46))

        # dots
        painter.setPen(QPen(Qt.NoPen))
        for i, pos in enumerate(dots_pos):
            x = int((pos[0] + 5) / 10 * width)
            y = int(top_height/2 - pos[1] / 5 * top_height/2)
            
            color = QColor(233, 69, 96) if dots_color[i] == 0 else QColor(78, 204, 163)
            painter.setBrush(color)
            painter.drawEllipse(x-6, y-6, 12, 12)

        # target zones
        tx_red = int((target_red[0]+5)/10*width)
        ty_red = int(top_height/2 - target_red[1]/5*top_height/2)
        painter.setBrush(QColor(233, 69, 96, 60))
        painter.setPen(QPen(QColor(233, 69, 96), 2))
        painter.drawEllipse(tx_red-20, ty_red-20, 40, 40)

        tx_blue = int((target_blue[0]+5)/10*width)
        ty_blue = int(top_height/2 - target_blue[1]/5*top_height/2)
        painter.setBrush(QColor(78, 204, 163, 60))
        painter.setPen(QPen(QColor(78, 204, 163), 2))
        painter.drawEllipse(tx_blue-20, ty_blue-20, 40, 40)

        # Graph container
        painter.setPen(QPen(QColor(255, 255, 255, 50)))
        painter.setBrush(QColor(22, 33, 62))
        painter.drawRect(margin, top_height + margin, width - 2*margin, bottom_height)
        
        if len(loss_history) < 2:
            return
        
        scale_y = max(loss_history) * 1.1
        
        #loss line
        pen = QPen(QColor(78, 204, 163), 2)
        painter.setPen(pen)
        
        for i in range(1, len(loss_history)):
            x1 = margin + (i-1)
            y1 = top_height + margin + bottom_height - int(loss_history[i-1]/scale_y * bottom_height)
            x2 = margin + i
            y2 = top_height + margin + bottom_height - int(loss_history[i]/scale_y * bottom_height)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        #loss counter
        painter.setPen(QPen(Qt.white))
        current_loss = loss_history[-1]
        painter.drawText((width >> 2)+ margin, top_height + margin + 20, f"Loss: {current_loss:.4f}")

# MAIN
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ANNWindow()
    window.show()
    sys.exit(app.exec())