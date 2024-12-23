
# Autonomous Drone Navigation with Deep Reinforcement Learning and Depth Maps  

In recent years, drones have revolutionized industries such as aerial imaging, logistics, and search-and-rescue operations. However, enabling drones to navigate autonomously without colliding with obstacles remains a significant challenge, especially in dynamic environments. This project presents an advanced solution by combining **Deep Reinforcement Learning (DRL)** and **depth maps** to enhance drone navigation and obstacle avoidance.  

Our model integrates the **Dueling Double Deep Q-Network (D3QN)** and **ResNet-8** architectures to enable accurate movement commands and robust steering angle predictions. To further enhance real-time performance, the models are optimized using **ONNX**, achieving a significant boost in inference speed. Tested in various simulated scenarios, the system demonstrates remarkable robustness in adverse weather conditions and superior obstacle detection capabilities compared to existing methods.  

---

## Table of Contents  

- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [File Descriptions](#file-descriptions)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Results and Performance](#results-and-performance)  
- [Future Improvements](#future-improvements)  
- [Acknowledgments](#acknowledgments)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Overview  

The increasing utilization of drones in various domains has led to a pressing need for reliable autonomous navigation systems. Current solutions face challenges such as detecting small obstacles (e.g., wires, tree branches), operating in poor weather or lighting conditions, and maintaining real-time inference speeds.  

This project aims to overcome these hurdles through a novel combination of:  
- **Deep Reinforcement Learning**: Leveraging the D3QN algorithm for movement prediction.  
- **Depth Maps**: Using the MiDaS algorithm to generate accurate depth information from single RGB images.  
- **ONNX Optimization**: Compressing models to improve inference speed without significant accuracy loss.  

Key contributions include:  
1. Enhanced accuracy and speed for collision avoidance.  
2. Robust performance in diverse environmental conditions.  
3. System validation in the AirSim simulation environment with varied speeds and weather.  

---

## Features  

- **Dynamic Obstacle Avoidance**: Handles small and complex obstacles effectively.  
- **Environment Adaptability**: Performs well in different lighting and weather conditions.  
- **Depth Map Integration**: Converts RGB images to depth maps for improved obstacle detection.  
- **Optimized Performance**: ONNX compression boosts inference speed.  
- **Simulated Testing**: Thorough validation in the AirSim simulation environment.  

---

## Installation  

### Prerequisites  

- **Python 3.8+**  
- Required libraries: Install via `requirements.txt`.  

```bash  
pip install -r requirements.txt  
```  

### Clone the Repository  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/your-username/your-repo-name.git  
   cd your-repo-name  
   ```  

2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

---

## File Descriptions  

### 1. `agent.py`  
Defines the **D3QN agent**, implementing:  
- Double Q-learning with Dueling Architecture for stable training.  
- Key functionalities like action selection (`choose_action`) and learning (`learn`).  

### 2. `main.py`  
The main script for training and evaluating the model.  
- Handles data loading, model training, and evaluation in AirSim.  
- Integrates depth and grayscale image processing for the two networks.  

### 3. `network.py`  
Contains the architectures for:  
- **D3QN**: Predicts movement commands based on depth images.  
- **ResNet-8**: Estimates collision probability and steering angle from grayscale images.  

### 4. `replay_memory.py`  
Manages experience replay, a key DRL component:  
- Stores state-action-reward sequences for sampling during training.  

### 5. `utils.py`  
A collection of utility functions for preprocessing and analysis:  
- Image transformations (e.g., RGB to depth maps).  
- Performance metrics and visualization utilities.  

---

## Usage  

### Training  

1. Prepare your training datasets and place them in the appropriate directory structure.
2. Configure hyperparameters in `main.py` as needed.  
3. Train the model:  
   ```bash  
   python main.py --train  
   ```  

### Testing  

1. Load pre-trained models in `main.py`.  
2. Evaluate performance:  
   ```bash  
   python main.py --evaluate  
   ```  
3. Test the trained model in the AirSim simulation environment:
   ```bash  
   python main.py --test  
   ```  

---

## Project Structure  

```
your-repo-name/  
├── agent.py                # D3QN Agent Implementation  
├── main.py                 # Main Script for Training and Testing  
├── network.py              # Neural Network Architectures  
├── replay_memory.py        # Replay Memory Management  
├── utils.py                # Utility Functions  
├── requirements.txt        # Required Python Libraries  
├── LICENSE                 # License File  
└── README.md               # Project Documentation  
```  

---

## Results and Performance  

### Highlights  

- **Accuracy**: Increased by 2% compared to baseline models.  
- **Inference Speed**: Improved by 25% with ONNX optimization.  
- **Environmental Robustness**: Reliable performance across challenging weather conditions (e.g., fog, rain, and night).  

### AirSim Testing  

- **Obstacle Avoidance Rate**: 98.6%  
- **Inference Speed Gain**: Enabled higher drone speeds with negligible accuracy loss (1.4%).  

---

## Future Improvements  

- Integrating real-world drone testing for field validation.  
- Enhancing small obstacle detection using LiDAR and stereo cameras.  
- Exploring ensemble models to improve robustness further.  

---

## Acknowledgments  

This project was made possible thanks to the following:  
- **Udacity and Real Collisions Datasets**: Provided essential data for training and testing.  
- **AirSim Simulation Environment**: A robust platform for testing autonomous drone navigation.  
- **ONNX Developers**: For enabling efficient model compression and optimization.  
- **Research Community**: For pioneering advancements in deep reinforcement learning and obstacle avoidance.  

Special thanks to:

- The AirSim team for providing a robust simulation environment.
- OpenAI and the broader machine learning community for inspiration and foundational research.
- The developers of MiDaS and ONNX for tools enabling efficient depth mapping and model optimization.

---

## Contributing  

We welcome contributions from the community! Here’s how you can get involved:  
1. Fork the repository.  
2. Create a feature branch:  
   ```bash  
   git checkout -b feature-name  
   ```  
3. Commit your changes:  
   ```bash  
   git commit -m "Add feature description"  
   ```  
4. Push to the branch:  
   ```bash  
   git push origin feature-name  
   ```  
5. Open a pull request.

---

## License  

This project is based on code originally written by Phil Tabor under the MIT License.
Modifications have been made by Omid Aghdaei in 2024.

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
