###  `NeuraDance` 

A deep learning project for recognizing **dance forms** using images and videos, powered by **MediaPipe** and **TensorFlow/Keras**.

---

## Directory Structure

```
NeuraDance/
├── DanceForms/           # Folder containing input dance videos (e.g., Bharatanatyam, Kathak, Mohiniyattam, etc.)
├── DataGen/              # Generated .npy pose data from videos
├── ImagesToPredict/      # Images used for prediction using the trained model
├── Model1/               # Folder to store models or experiments
├── genData.py            # Script to extract pose data from videos using MediaPipe
├── labels.npy            # Numpy array of corresponding labels for training data
├── model.h5              # Trained Keras model
├── predict.py            # Predicts the dance form from images
├── trainModel1.py        # Training script (Version 1)
├── trainModel2.py        # Training script (Version 2)

```

---

## Setup

Before running any scripts, make sure the following dependencies are installed:

```bash
pip install opencv-python mediapipe numpy tensorflow

```

---

## Step-by-Step Workflow

### 1. **Generate Training Data from Videos**

Extract pose landmarks from videos using **MediaPipe** and store them as `.npy` files.

```bash
python genData.py

```

- This will read all `.mp4` videos from `DanceForms/`
- For each video, a corresponding `.npy` pose sequence is saved in `DataGen/`

---

### 2. **Train the Dance Recognition Model**

You can choose one of the training scripts to train your model using the generated pose data.

```bash
python trainModel1.py   # First model
# OR
python trainModel2.py   # Made few changes
```

- Make sure `labels.npy` is present or created in the training script.
- The trained model is saved as `model.h5`.

---

### 3. **Predict Dance Form from Image**

Once the model is trained, you can run predictions on still images using:

```bash
python predict.py

```

- Place your images in the `ImagesToPredict/` folder before running.
- It will use the trained `model.h5` to classify the dance form.

---

## Notes

- `genData.py` uses **MediaPipe Pose** to extract body landmarks from each frame.
- The model learns based on relative body joint positions.
- You can modify `trainModel*.py` to use sequences (for video classification) or still frames.
