# 🧠 Simple Artificial Neural Network (ANN) – Dot Position Learner

A small artificial neural network project where colored dots learn to move toward predefined target positions based on their color.

This project was created to better understand the fundamentals of neural networks, loss functions, and parameter tuning.

---

## **📌 Overview **

The neural network receives:

- The current position of a dot
- The color of the dot

Based on this input, it predicts the correct target position.

Each color corresponds to a different target location.  
The network gradually learns to move dots to their correct positions by minimizing a loss function.

Movement happens step-by-step, allowing the learning process to be visualized in real time.

---

## ⚙️ How It Works

1. The network evaluates the dot's position and color.
2. It predicts the direction and magnitude of movement.
3. The dot moves by a small increment.
4. The loss is calculated based on the distance to the correct target.
5. Using backpropagation and gradient-based updates, the network adjusts its parameters.

Over time, the dots reach their correct positions more efficiently.

---

## 📊 Features

- Configurable training parameters
- Real-time GUI visualization

The graphical interface shows:
- The moving dots
- Their target positions
- A graph of the loss function during training

---

## 🎯 Purpose

This project was built purely for learning purposes to better understand:

- Neural network structure
- Loss functions
- Gradient-based learning
- Visualization of training behavior

It is intentionally simple and focused on clarity rather than performance or complexity.

---

## 🛠 Tech Stack

- Python
- Python extensions:
    numPy for math
    PySide6 for GUI
    
